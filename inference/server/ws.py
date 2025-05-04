import socket
import threading
import json
import torch
import numpy as np

from inference.load_model import load_model
from config.env_config import FIELD_SIZE
from utils.checks import is_target_reached, is_collision

env, agent = load_model()

HOST = "localhost"
PORT = 12345

def handle_client(conn):
    print("🎮 Client connected.")
    buffer = ""
    while True:
        try:
            data = conn.recv(4096).decode()
            if not data:
                break

            buffer += data
            if not buffer.endswith("\n"):
                continue  # Wait for end of full message

            request = json.loads(buffer.strip())
            buffer = ""

            command = request.get("command")

            if command == "init_episode":
                env.reset()
                response = {
                    "field_size": FIELD_SIZE,
                    "drone": env.drone_pos.tolist(),
                    "target": env.target_pos.tolist(),
                    "obstacles": [obs.tolist() for obs in env.obstacles]
                }

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
                print("Unknown command:", command)
                response = {"error": "Unknown command"}

            conn.sendall((json.dumps(response) + "\n").encode())

        except Exception as e:
            print("💥 Connection error:", e)
            break

    conn.close()
    print("🔌 Client disconnected.")

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"🟢 TCP server listening on {HOST}:{PORT}")

    while True:
        conn, _ = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn,))
        thread.start()

if __name__ == "__main__":
    main()
