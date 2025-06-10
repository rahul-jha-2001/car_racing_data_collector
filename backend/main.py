import logging
import json
import zlib
import asyncio
from contextlib import asynccontextmanager
import numpy as np
import gymnasium as gym
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from PIL import Image
import io
import os
import uuid
import aioboto3
import redis.asyncio as redis
from dotenv import load_dotenv
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError
import tarfile
import tempfile
import backoff
from dataclasses import dataclass
from typing import List, Optional
import time

from pydantic import BaseModel

class ScoreModel(BaseModel):
    id: str  # stringified ObjectId
    session_id: str
    player: str
    score: float
    timestamp: float

@dataclass
class AppConfig:
    """Configuration class for all app settings"""
    mongo_host: str = "mongodb://localhost:27017"
    redis_host: str = "localhost"
    redis_port: int = 6379
    aws_region: str = "us-east-1"
    s3_bucket: Optional[str] = None
    session_ttl: int = 600
    frame_rate: int = 30
    max_retries: int = 3
    upload_timeout: float = 30.0

# Load environment variables
load_dotenv()

# Configuration
config = AppConfig(
    mongo_host=os.getenv("MONGO_HOST", "mongodb://localhost:27017"),
    redis_host=os.getenv("REDIS_HOST", "localhost"),
    redis_port=int(os.getenv("REDIS_PORT", "6379")),
    aws_region=os.getenv("AWS_REGION", "us-east-1"),
    s3_bucket=os.getenv("S3_BUCKET"),
    session_ttl=int(os.getenv("SESSION_TTL", "600")),
    frame_rate=int(os.getenv("FRAME_RATE", "30")),
)

# Validate required configuration
if not config.s3_bucket:
    raise ValueError("S3_BUCKET environment variable is required")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("car_racing_ws")

# MongoDB setup
mongo_client = AsyncIOMotorClient(config.mongo_host)
db = mongo_client["carracing"]
sessions_collection = db["sessions"]

# Redis setup
redis_client = redis.Redis(
    host=config.redis_host, 
    port=config.redis_port, 
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5,
    retry_on_timeout=True
)

# AWS S3 setup (async)
s3_session = aioboto3.Session()

@backoff.on_exception(backoff.expo, Exception, max_tries=config.max_retries)
async def check_redis():
    """Check Redis connectivity with retry logic"""
    try:
        pong = await redis_client.ping()
        return pong
    except Exception as e:
        logger.error(f"Redis connection check failed: {e}")
        raise

@backoff.on_exception(backoff.expo, ClientError, max_tries=config.max_retries)
async def check_s3():
    """Check S3 connectivity with retry logic"""
    try:
        async with s3_session.client(
            's3', 
            region_name=config.aws_region,
            config=Config(max_pool_connections=50)
        ) as s3:
            await s3.list_buckets()
        return True
    except (ClientError, NoCredentialsError) as e:
        logger.error(f"S3 connection check failed: {e}")
        raise

async def store_frame_metadata(session_id: str, frame: int, action: list, reward: float, done: bool):
    """Store lightweight frame metadata in Redis"""
    try:
        metadata = {
            "frame": frame,
            "action": action,
            "reward": reward,
            "done": done,
            "timestamp": time.time()
        }
        
        await redis_client.rpush(
            f"session:{session_id}",
            json.dumps(metadata)
        )
        await redis_client.expire(f"session:{session_id}", config.session_ttl)
        
    except redis.RedisError as e:
        logger.error(f"Failed to store frame metadata in Redis: {e}")
        raise

async def upload_frame_to_s3(obs: np.ndarray, session_id: str, frame: int):
    """Upload individual frame to S3 asynchronously"""
    try:
        img = Image.fromarray(obs)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        image_key = f"car-racing-data/{session_id}/{frame}.png"
        
        async with s3_session.client(
            's3', 
            region_name=config.aws_region,
            config=Config(max_pool_connections=50)
        ) as s3:
            await asyncio.wait_for(
                s3.upload_fileobj(buffer, config.s3_bucket, image_key),
                timeout=config.upload_timeout
            )
            
    except asyncio.TimeoutError:
        logger.error(f"S3 upload timeout for frame {frame} in session {session_id}")
        raise
    except ClientError as e:
        logger.error(f"S3 upload failed for frame {frame}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error uploading frame {frame}: {e}")
        raise

async def upload_session_bulk(frames: List[str], session_id: str):
    """Upload entire session as single archive file with proper error handling"""
    temp_file_path = None
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as temp_file:
            temp_file_path = temp_file.name
            
            with tarfile.open(temp_file.name, 'w:gz') as tar:
                for frame_json in frames:
                    try:
                        data = json.loads(frame_json)
                        
                        # Add metadata to archive
                        metadata = {
                            "frame": data["frame"],
                            "action": data["action"],
                            "reward": data["reward"],
                            "done": data["done"],
                            "timestamp": data.get("timestamp", time.time())
                        }
                        metadata_bytes = json.dumps(metadata).encode('utf-8')
                        metadata_info = tarfile.TarInfo(name=f"{data['frame']}.json")
                        metadata_info.size = len(metadata_bytes)
                        tar.addfile(metadata_info, io.BytesIO(metadata_bytes))
                        
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Skipping malformed frame data: {e}")
                        continue
            
            # Upload to S3 asynchronously
            archive_key = f"car-racing-sessions/{session_id}.tar.gz"
            async with s3_session.client(
                's3', 
                region_name=config.aws_region,
                config=Config(max_pool_connections=50)
            ) as s3:
                await asyncio.wait_for(
                    s3.upload_file(temp_file.name, config.s3_bucket, archive_key),
                    timeout=config.upload_timeout
                )
                
            logger.info(f"✅ Uploaded session {session_id} as archive ({len(frames)} frames)")
            
    except asyncio.TimeoutError:
        logger.error(f"S3 archive upload timeout for session {session_id}")
        raise
    except ClientError as e:
        logger.error(f"S3 archive upload failed for session {session_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating archive for session {session_id}: {e}")
        raise
    finally:
        # Always clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except OSError as e:
                logger.warning(f"Failed to clean up temp file {temp_file_path}: {e}")

class GameSession:
    """Encapsulate game session state and logic"""
    
    def __init__(self, session_id: str, player_name: str = "Anonymous"):
        self.session_id = session_id
        self.player_name = player_name
        self.env = None
        self.obs = None
        self.total_reward = 0.0
        self.sent_frames = 0
        self.action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.action_lock = asyncio.Lock()
        self._running = False
        
    async def initialize(self):
        """Initialize the game environment"""
        try:
            self.env = gym.make("CarRacing-v3", render_mode="rgb_array")
            self.obs, _ = self.env.reset()
            logger.info(f"Game session {self.session_id} initialized for player {self.player_name}")
        except Exception as e:
            logger.error(f"Failed to initialize game session: {e}")
            raise
    
    async def get_action(self):
        """Thread-safe action getter"""
        async with self.action_lock:
            return self.action.copy()
    
    async def set_action(self, new_action: np.ndarray):
        """Thread-safe action setter"""
        async with self.action_lock:
            self.action[:] = new_action
    
    async def step(self):
        """Execute one game step"""
        if not self.env:
            raise RuntimeError("Game session not initialized")
            
        current_obs = self.obs.copy()
        action = await self.get_action()
        
        try:
            self.obs, reward, terminated, truncated, _ = self.env.step(action)
            self.total_reward += reward
            
            # Store metadata (lightweight)
            await store_frame_metadata(
                self.session_id,
                self.sent_frames,
                action.tolist(),
                float(reward),
                terminated or truncated
            )
            
            # Optionally upload frame to S3 in background (for real-time backup)
            # asyncio.create_task(upload_frame_to_s3(current_obs, self.session_id, self.sent_frames))
            
            self.sent_frames += 1
            
            return current_obs, reward, terminated, truncated
            
        except Exception as e:
            logger.error(f"Game step failed: {e}")
            raise
    
    async def reset(self):
        """Reset the game session"""
        if self.env:
            try:
                await self.save_score()
                self.obs, _ = self.env.reset()
                self.total_reward = 0.0
                self.sent_frames = 0
                logger.info(f"Game session {self.session_id} reset")
            except Exception as e:
                logger.error(f"Failed to reset game session: {e}")
                raise
    
    async def save_score(self):
        """Save the final score to MongoDB"""
        try:
            await sessions_collection.insert_one({
                "session_id": self.session_id,
                "player": self.player_name,
                "score": float(self.total_reward),
                "timestamp": time.time(),
            })
            logger.info(f"✅ Score saved for {self.player_name}: {self.total_reward}")
        except Exception as e:
            logger.error(f"Failed to save score: {e}")
            raise
    
    def cleanup(self):
        """Clean up resources"""
        if self.env:
            try:
                self.env.close()
                logger.info(f"Game environment closed for session {self.session_id}")
            except Exception as e:
                logger.warning(f"Error closing environment: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # STARTUP
    startup_errors = []
    
    try:
        await mongo_client.server_info()
        logger.info("✅ MongoDB connected successfully")
    except Exception as e:
        error_msg = f"❌ MongoDB connection failed: {e}"
        logger.error(error_msg)
        startup_errors.append(error_msg)

    try:
        await check_s3()
        logger.info("✅ S3 connected successfully")
    except Exception as e:
        error_msg = f"❌ S3 connection failed: {e}"
        logger.error(error_msg)
        startup_errors.append(error_msg)

    try:
        pong = await check_redis()
        logger.info(f"✅ Redis connected: {pong}")
    except Exception as e:
        error_msg = f"❌ Redis connection failed: {e}"
        logger.error(error_msg)
        startup_errors.append(error_msg)
    
    if startup_errors:
        logger.warning(f"Application started with {len(startup_errors)} connection issues")
    else:
        logger.info("✅ All services connected successfully")

    yield

    # SHUTDOWN
    try:
        await redis_client.close()
        logger.info("✅ Redis connection closed")
    except Exception as e:
        logger.warning(f"⚠️ Redis failed to close cleanly: {e}")
    
    try:
        mongo_client.close()
        logger.info("✅ MongoDB connection closed")
    except Exception as e:
        logger.warning(f"⚠️ MongoDB failed to close cleanly: {e}")

app = FastAPI(
    title="Car Racing WebSocket API",
    description="Real-time car racing game with data collection",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {"status": "healthy", "services": {}}
    
    # Check MongoDB
    try:
        await mongo_client.server_info()
        health_status["services"]["mongodb"] = "connected"
    except Exception:
        health_status["services"]["mongodb"] = "disconnected"
        health_status["status"] = "degraded"
    
    # Check Redis
    try:
        await redis_client.ping()
        health_status["services"]["redis"] = "connected"
    except Exception:
        health_status["services"]["redis"] = "disconnected"
        health_status["status"] = "degraded"
    
    # Check S3
    try:
        await check_s3()
        health_status["services"]["s3"] = "connected"
    except Exception:
        health_status["services"]["s3"] = "disconnected"
        health_status["status"] = "degraded"
    
    return health_status

@app.websocket("/ws/play")
async def play_game(websocket: WebSocket):
    await websocket.accept()
    
    session_id = str(uuid.uuid4())
    game_session = None
    game_task = None
    
    logger.info(f"New WebSocket connection established: {session_id}")
    
    try:
        game_session = GameSession(session_id)
        await game_session.initialize()
        
        h, w, _ = game_session.obs.shape
        await websocket.send_text(json.dumps({
            "type": "init", 
            "width": w, 
            "height": h,
            "session_id": session_id
        }))
        
        async def game_loop():
            """Main game loop with proper error handling"""
            try:
                while True:
                    try:
                        current_obs, reward, terminated, truncated = await game_session.step()
                        
                        # Send compressed frame data
                        raw = game_session.obs.tobytes()
                        comp = zlib.compress(raw, level=3)
                        await websocket.send_bytes(comp)

                        # Send frame metadata
                        await websocket.send_text(json.dumps({
                            "type": "frame_meta",
                            "reward": float(game_session.total_reward),
                            "frame": game_session.sent_frames - 1,
                            "current_reward": float(reward)
                        }))

                        if terminated or truncated:
                            await websocket.send_text(json.dumps({
                                "type": "done",
                                "score": float(game_session.total_reward)
                            }))
                            
                            await game_session.reset()

                        # Frame rate control
                        await asyncio.sleep(1 / config.frame_rate)
                        
                    except WebSocketDisconnect:
                        logger.info(f"WebSocket disconnected during game loop: {session_id}")
                        break
                    except Exception as e:
                        logger.error(f"Error in game loop: {e}")
                        # Try to continue unless it's a critical error
                        if "environment" in str(e).lower():
                            break
                        await asyncio.sleep(0.1)  # Brief pause before continuing
                        
            except asyncio.CancelledError:
                logger.info(f"Game loop cancelled for session: {session_id}")
            except Exception as e:
                logger.error(f"Game loop stopped unexpectedly: {e}")

        game_task = asyncio.create_task(game_loop())

        # Handle WebSocket messages
        while True:
            try:
                msg = await websocket.receive_text()
                data = json.loads(msg)

                if data.get("type") == "start":
                    player_name = data.get("name", "Anonymous")
                    game_session.player_name = player_name
                    logger.info(f"Player '{player_name}' started session {session_id}")

                elif data.get("type") == "action":
                    action_data = data.get("action", [0.0, 0.0, 0.0])
                    new_action = np.array(action_data, dtype=np.float32)
                    await game_session.set_action(new_action)

                else:
                    logger.warning(f"Unknown message type: {data.get('type')}")
                    
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: {session_id}")
                break
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON received: {e}")
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                break

    except Exception as e:
        logger.error(f"Critical error in WebSocket handler: {e}")
    finally:
        # Cleanup
        if game_task:
            game_task.cancel()
            try:
                await game_task
            except asyncio.CancelledError:
                pass
        
        if game_session:
            game_session.cleanup()
        
        # Process session data - FIXED LOGIC
        await process_session_data(session_id)
        
        logger.info(f"Session {session_id} cleanup completed")

async def process_session_data(session_id: str):
    """Process and upload session data with proper error handling"""
    try:
        # FIXED: Check if session exists before trying to fetch
        exists = await redis_client.exists(f"session:{session_id}")
        if exists:
            logger.info(f"Processing session data for {session_id}")
            
            frames = None
            for attempt in range(config.max_retries):
                try:
                    frames = await asyncio.wait_for(
                        redis_client.lrange(f"session:{session_id}", 0, -1),
                        timeout=5.0
                    )
                    break
                except asyncio.TimeoutError:
                    logger.warning(f"Redis fetch timeout (attempt {attempt + 1}/3)")
                    if attempt == config.max_retries - 1:
                        logger.error(f"Failed to fetch session data after {config.max_retries} attempts")
                        return
                except redis.RedisError as e:
                    logger.error(f"Redis error fetching session data: {e}")
                    return

            if frames and len(frames) > 0:
                try:
                    await upload_session_bulk(frames, session_id)
                    # Clean up Redis data after successful upload
                    await redis_client.delete(f"session:{session_id}")
                    logger.info(f"✅ Session {session_id} processed and cleaned up")
                except Exception as e:
                    logger.error(f"Failed to upload session bulk data: {e}")
            else:
                logger.warning(f"No frames found for session {session_id}")
        else:
            logger.warning(f"Session {session_id} not found in Redis")
            
    except Exception as e:
        logger.error(f"Error processing session data: {e}")

def clean_score(score_doc):
    """Clean and format score document"""
    return {
        "id": str(score_doc["_id"]),
        "session_id": score_doc["session_id"],
        "player": score_doc["player"],
        "score": score_doc["score"],
        "timestamp": score_doc["timestamp"],
    }

@app.get("/api/scores", response_model=List[ScoreModel])
async def get_scores(limit: int = 10):
    """Get top scores with error handling"""
    try:
        if limit <= 0 or limit > 100:
            raise HTTPException(status_code=400, detail="Limit must be between 1 and 100")
            
        raw_scores = await sessions_collection.find().sort("score", -1).limit(limit).to_list(length=limit)
        return [clean_score(score) for score in raw_scores]
    except Exception as e:
        logger.error(f"Error fetching scores: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch scores")

@app.get("/api/metrics")
async def get_metrics():
    """Get basic application metrics"""
    try:
        total_sessions = await sessions_collection.count_documents({})
        
        # Get Redis info if available
        redis_info = {}
        try:
            redis_info = await redis_client.info()
        except Exception:
            redis_info = {"status": "unavailable"}
        
        return {
            "total_sessions": total_sessions,
            "redis_info": {
                "connected_clients": redis_info.get("connected_clients", "unknown"),
                "used_memory_human": redis_info.get("used_memory_human", "unknown"),
            },
            "config": {
                "frame_rate": config.frame_rate,
                "session_ttl": config.session_ttl,
                "max_retries": config.max_retries,
            }
        }
    except Exception as e:
        logger.error(f"Error fetching metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch metrics")