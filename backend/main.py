import logging
import json
import zlib
import asyncio

import numpy as np
import gymnasium as gym
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# — Configure logging —
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("car_racing_ws")

# — FastAPI setup —
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/play")
async def play_game(websocket: WebSocket):
    await websocket.accept()
    name = "Anonymous"
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    obs, _ = env.reset()
    sent_frames = 0
    action = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Default action

    # Send frame dimensions
    h, w, _ = obs.shape
    await websocket.send_text(json.dumps({"type": "init", "width": w, "height": h}))
    logger.info(f"Sent init (w={w}, h={h})")

    total_reward = 0.0

    async def game_loop():
        nonlocal obs, sent_frames, action, total_reward
        try:
            while True:
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

                # Send compressed frame
                raw = obs.tobytes()
                comp = zlib.compress(raw, level=3)
                await websocket.send_bytes(comp)

                # Send reward metadata
                await websocket.send_text(json.dumps({
                    "type": "frame_meta",
                    "reward": float(total_reward),
                    "frame": sent_frames
                }))

                sent_frames += 1
                logger.debug(f"→ Sent frame #{sent_frames} to '{name}' (reward={total_reward:.2f})")

                if terminated or truncated:
                    await websocket.send_text(json.dumps({
                        "type": "done",
                        "score": float(total_reward)
                    }))
                    obs, _ = env.reset()
                    total_reward = 0.0
                    sent_frames = 0
                    logger.info(f"Episode ended and reset for '{name}'")

                await asyncio.sleep(1 / 60)
        except Exception as e:
            logger.info(f"Game loop stopped for '{name}': {e}")


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


