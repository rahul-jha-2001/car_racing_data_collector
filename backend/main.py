import logging
import json
import zlib
import asyncio
import numpy as np
import gymnasium as gym
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import uuid
from dotenv import load_dotenv

# Load env variables from .env file
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
logger.info(f"Mongo host: {MONGO_HOST}")
mongo_client = AsyncIOMotorClient(MONGO_HOST)
db = mongo_client["carracing"]
sessions_collection = db["sessions"]
frames_collection = db["frames"]

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def verify_mongo_connection():
    try:
        await mongo_client.server_info()
        logger.info("✅ MongoDB connected successfully at startup")
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed at startup: {e}")

@app.websocket("/ws/play")
async def play_game(websocket: WebSocket):
    await websocket.accept()
    name = "Anonymous"
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    obs, _ = env.reset()
    sent_frames = 0
    total_reward = 0.0
    action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    session_id = str(uuid.uuid4())

    # Send frame dimensions
    h, w, _ = obs.shape
    await websocket.send_text(json.dumps({"type": "init", "width": w, "height": h}))
    logger.info(f"Sent init (w={w}, h={h})")

    async def store_frame(doc):
        try:
            await frames_collection.insert_one(doc)
        except Exception as e:
            logger.warning(f"❌ Failed to insert frame: {e}")

    async def game_loop():
        nonlocal obs, sent_frames, action, total_reward
        try:
            while True:
                current_obs = obs.copy()
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

                asyncio.create_task(store_frame({
                    "session_id": session_id,
                    "frame": sent_frames,
                    "obs": current_obs.tolist(),
                    "action": action.tolist(),
                    "reward": float(reward),
                    "done": terminated or truncated,
                }))

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

                await asyncio.sleep(1 / 60)
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

            else:
                logger.warning(f"Unknown type from '{name}': {data.get('type')}")
    except WebSocketDisconnect:
        logger.info(f"WebSocket closed by '{name}'")
    finally:
        game_task.cancel()
        env.close()
        logger.info(f"Env closed for '{name}'")
