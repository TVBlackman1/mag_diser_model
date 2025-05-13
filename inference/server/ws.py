import asyncio
import json
import numpy as np
import torch
import websockets

from inference.load_model import load_model
from config.env_config import FIELD_SIZE
from utils.checks import is_target_reached, is_collision

env, agent = load_model()

HOST = "localhost"
PORT = 8765

async def handle_client(websocket):
    print("🎮 Client connected.")
    try:
        async for message in websocket:
            request = json.loads(message)
            command = request.get("command")

            if command == "init_episode":
                env.reset()
                response = {
                    "field_size": FIELD_SIZE,
                    "drone": env.drone_pos.tolist(),
                    "target": env.target_pos.tolist(),
                    "obstacles": [obs.tolist() for obs in env.obstacles]
                }
                print("🆕 New episode initialized")

            elif command == "step":
                obs = np.array(request["observation"], dtype=np.float32)
                with torch.no_grad():
                    action = agent.actor(torch.tensor(obs).unsqueeze(0)).squeeze(0).numpy()
                dx, dy = action.tolist()

                response = {
                    "dx": dx,
                    "dy": dy,
                    "drone_pos": env.drone_pos.tolist(),
                    "goal_reached": bool(is_target_reached(env.drone_pos, env.target_pos)),
                    "collision": bool(is_collision(env.drone_pos, env.obstacles))
                }

            else:
                print("⚠️ Unknown command:", command)
                response = {"error": "Unknown command"}

            await websocket.send(json.dumps(response))

    except websockets.ConnectionClosed:
        print("🔌 Client disconnected (clean)")
    except Exception as e:
        print("💥 Connection error:", e)
    finally:
        print("📴 Session ended")

async def main():
    async with websockets.serve(handle_client, HOST, PORT):
        print(f"🟢 WebSocket server listening on ws://{HOST}:{PORT}")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
