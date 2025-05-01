import asyncio

import torch
import websockets
import json
import numpy as np

from inference.load_model import load_model
from config.env_config import FIELD_SIZE

env, agent = load_model()

async def handle_client(websocket):
    print("Client connected.")

    async for message in websocket:
        data = json.loads(message)
        command = data.get("command")

        if command == "init_episode":
            env.reset()
            response = {
                "field_size": FIELD_SIZE,
                "drone": env.drone_pos.tolist(),
                "target": env.target_pos.tolist(),
                "obstacles": [obs.tolist() for obs in env.obstacles]
            }
            await websocket.send(json.dumps(response))

        elif command == "step":
            obs = np.array(data["observation"], dtype=np.float32)
            with torch.no_grad():
                action = agent.actor(torch.tensor(obs).unsqueeze(0)).squeeze(0).numpy()
            dx, dy = action.tolist()
            await websocket.send(json.dumps({"dx": dx, "dy": dy}))

        elif command == "goal_reached":
            print("✅ Target reached by drone.")
            await websocket.send(json.dumps({"ack": "goal_received"}))

        else:
            print("Unknown command:", command)
            await websocket.send(json.dumps({"error": "Unknown command"}))

async def main():
    async with websockets.serve(handle_client, "localhost", 8765):
        print("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
