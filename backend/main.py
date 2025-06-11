import logging
import json
import zlib
import asyncio
from contextlib import asynccontextmanager
import numpy as np
import gymnasium as gym
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from PIL import Image
import io
import os
import uuid
import boto3
import redis.asyncio as redis
from dotenv import load_dotenv
from botocore.config import Config
import tarfile
import tempfile
import os
import backoff
from in_memory import InMemoryStore


from pydantic import BaseModel
from typing import List

class ScoreModel(BaseModel):
    id: str  # stringified ObjectId
    session_id: str
    player: str
    score: float
    timestamp: float

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("car_racing_ws")

# MongoDB setup
MONGO_HOST = os.getenv("MONGO_HOST", "mongodb://localhost:27017")
mongo_client = AsyncIOMotorClient(MONGO_HOST)
db = mongo_client["carracing"]
sessions_collection = db["sessions"]

# Redis setup
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
redis_client = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)

# AWS S3 setup
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET")
s3 = boto3.client("s3", region_name=AWS_REGION, config=Config(max_pool_connections=50))

IN_MEMORY_STORE = InMemoryStore()

# FastAPI setup

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application lifespan...")
    # STARTUP
    try:
        await mongo_client.server_info()
        logger.info("✅ MongoDB connected successfully at startup")
    except Exception as e:
        logger.info(f"{MONGO_HOST}")
        logger.error(f"❌ MongoDB connection failed at startup: {e}")

    try:
        s3.list_buckets()
        logger.info(f"✅ Connected to S3 and verified access to bucket '{S3_BUCKET}'")
    except Exception as e:
        logger.error(f"❌ S3 connection failed at startup: {e}")

    async def check_redis():
        try:
            return await redis_client.ping()
        except Exception as e:
            logger.error(f"❌ Redis ping failed: {e}")
            raise

    try:
        pong = await check_redis()
        logger.info(f"✅ Redis connected: {pong}")
    except Exception as e:
        logger.error(f"❌ Redis connection failed at startup: {e}")

    yield

    # SHUTDOWN (optional)
    logger.info("Shutting down application lifespan...")
    try:
        await redis_client.close()
        logger.info("✅ Redis connection closed cleanly")
    except Exception as e:
        logger.warning(f"⚠️ Redis failed to close cleanly: {e}")

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def store_frame(obs, action, reward, done, session_id, frame):
    logger.info(f"Storing frame {frame} for session {session_id}...")
    try:
        img = Image.fromarray(obs)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        image_key = f"car-racing-data/{session_id}/{frame}.png"
        s3.upload_fileobj(buffer, S3_BUCKET, image_key)
        logger.info(f"✅ Uploaded image frame {frame} to S3")

        metadata = {
            "frame": frame,
            "action": action,
            "reward": float(reward),
            "done": done,
        }
        metadata_bytes = io.BytesIO(json.dumps(metadata).encode("utf-8"))
        json_key = f"car-racing-data/{session_id}/{frame}.json"
        s3.upload_fileobj(metadata_bytes, S3_BUCKET, json_key)
        logger.info(f"✅ Uploaded metadata for frame {frame} to S3")

    except Exception as e:
        logger.warning(f"❌ Failed to upload frame or metadata to S3: {e}")

async def upload_session_bulk(frames, session_id):
    logger.info(f"Uploading session {session_id} as a single archive...")
    def create_and_upload_archive():
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as temp_file:
            try:
                with tarfile.open(temp_file.name, 'w:gz') as tar:
                    for frame_json in frames:
                        data = json.loads(frame_json)
                        obs_array = np.array(data["obs"], dtype=np.uint8)
                        
                        # Add image to archive
                        img = Image.fromarray(obs_array)
                        img_buffer = io.BytesIO()
                        img.save(img_buffer, format="PNG")
                        
                        img_info = tarfile.TarInfo(name=f"{data['frame']}.png")
                        img_info.size = len(img_buffer.getvalue())
                        img_buffer.seek(0)
                        tar.addfile(img_info, img_buffer)
                        
                        # Add metadata to archive
                        metadata = {
                            "frame": data["frame"],
                            "action": data["action"],
                            "reward": data["reward"],
                            "done": data["done"],
                        }
                        metadata_bytes = json.dumps(metadata).encode('utf-8')
                        metadata_info = tarfile.TarInfo(name=f"{data['frame']}.json")
                        metadata_info.size = len(metadata_bytes)
                        tar.addfile(metadata_info, io.BytesIO(metadata_bytes))
                
                # Single S3 upload call
                archive_key = f"car-racing-sessions/{session_id}.tar.gz"
                s3.upload_file(temp_file.name, S3_BUCKET, archive_key)
                logger.info(f"✅ Uploaded session {session_id} as single archive ({len(frames)} frames)")
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
                    logger.info(f"✅ Temporary archive file cleaned up")
    
    create_and_upload_archive()

@app.websocket("/ws/play")
async def play_game(websocket: WebSocket):
    logger.info("WebSocket connection initiated...")
    await websocket.accept()
    name = "Anonymous"
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    obs, _ = env.reset()
    sent_frames = 0
    total_reward = 0.0
    action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    session_id = str(uuid.uuid4())

    h, w, _ = obs.shape
    await websocket.send_text(json.dumps({"type": "init", "width": w, "height": h}))
    logger.info(f"Sent init (w={w}, h={h})")

    await redis_client.expire(f"session:{session_id}", 600)
    logger.info(f"Redis session {session_id} set to expire in 600 seconds")

    async def game_loop():
        nonlocal obs, sent_frames, action, total_reward
        logger.info("Game loop started...")
        try:
            while True:
                current_obs = obs.copy()
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

                await IN_MEMORY_STORE.rpush(
                    f"session:{session_id}",
                    json.dumps({
                        "frame": sent_frames,
                        "obs": current_obs.tolist(),
                        "action": action.tolist(),
                        "reward": float(reward),
                        "done": terminated or truncated
                    })
                )
                await redis_client.expire(f"session:{session_id}", 600)
                
                raw = obs.tobytes()
                comp = zlib.compress(raw, level=3)
                await websocket.send_bytes(comp)

                await websocket.send_text(json.dumps({
                    "type": "frame_meta",
                    "reward": float(total_reward),
                    "frame": sent_frames
                }))
                

                sent_frames += 1

                if terminated or truncated:
                    await websocket.send_text(json.dumps({
                        "type": "done",
                        "score": float(total_reward)
                    }))
                    logger.info(f"Game terminated or truncated. Final score: {total_reward}")

                    await sessions_collection.insert_one({
                        "session_id": session_id,
                        "player": name,
                        "score": float(total_reward),
                        "timestamp": asyncio.get_event_loop().time(),
                    })
                    logger.info(f"✅ Final score recorded for {name} (session={session_id})")

                    obs, _ = env.reset()
                    total_reward = 0.0
                    sent_frames = 0

                await asyncio.sleep(1 / 30)
        except Exception as e:
            logger.warning(f"Game loop stopped for '{name}': {e}")

    game_task = asyncio.create_task(game_loop())

    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)

            if data.get("type") == "start":
                name = data.get("name", "Anonymous")
                logger.info(f"Driver '{name}' connected")

            elif data.get("type") == "action":
                action = np.array(data["action"], dtype=np.float32)
                logger.info(f"Received action from '{name}': {action}")

            else:
                logger.warning(f"Unknown type from '{name}': {data.get('type')}")
    except WebSocketDisconnect:
        logger.info(f"WebSocket closed by '{name}'")
    finally:
        logger.info("Cleaning up after WebSocket disconnect...")

        logger.info(f"Pulling frames from Redis for session {session_id}")
        frames = None
        exists = await redis_client.exists(f"session:{session_id}")
        if exists:
            for attempt in range(3):
                try:
                    frames = await asyncio.wait_for(
                        IN_MEMORY_STORE.lrange(f"session:{session_id}", 0, -1),
                        timeout=5.0
                    )
                    logger.info(f"Successfully pulled frames from Redis (attempt {attempt + 1})")
                    break
                except asyncio.TimeoutError:
                    frames = None
                    logger.warning(f"Retrying Redis pull (attempt {attempt + 1})...")
            else:
                logger.error("❌ Failed to pull frames from Redis after retries")

        if frames is not None and len(frames) != 0:
            await upload_session_bulk(frames=frames, session_id=session_id)
            await redis_client.delete(f"session:{session_id}")
            logger.info(f"✅ Redis session {session_id} cleaned up after upload")
            game_task.cancel()
            env.close()
        else:
            logger.error(f"Frames could not be read from Redis for session {session_id}")

def clean_score(score_doc):
    logger.info(f"Cleaning score document: {score_doc}")
    return {
        "id": str(score_doc["_id"]),
        "session_id": score_doc["session_id"],
        "player": score_doc["player"],
        "score": score_doc["score"],
        "timestamp": score_doc["timestamp"],
    }

@app.get("/api/scores", response_model=List[ScoreModel])
async def get_scores(limit: int = 10):
    logger.info(f"Fetching top {limit} scores from MongoDB...")
    raw_scores = await sessions_collection.find().sort("score", -1).limit(limit).to_list(length=limit)
    logger.info(f"Fetched {len(raw_scores)} scores from MongoDB")
    return [clean_score(score) for score in raw_scores]
