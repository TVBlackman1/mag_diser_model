import socket
import json

HOST = "localhost"
PORT = 12345

def send_json(sock, obj):
    msg = json.dumps(obj) + "\n"
    sock.sendall(msg.encode())

def receive_json(sock, timeout=5.0):
    sock.settimeout(timeout)
    buffer = b""
    while not buffer.endswith(b"\n"):
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("Server closed the connection")
        buffer += chunk
    return json.loads(buffer.decode())

def test_init_episode():
    with socket.create_connection((HOST, PORT), timeout=5) as sock:
        send_json(sock, {"command": "init_episode"})
        response = receive_json(sock)

        assert "field_size" in response
        assert "drone" in response
        assert "target" in response
        assert "obstacles" in response

def test_step():
    with socket.create_connection((HOST, PORT), timeout=5) as sock:
        send_json(sock, {"command": "init_episode"})
        _ = receive_json(sock)

        obs = [0.0, 1.0, 1.0, 0.5]
        send_json(sock, {"command": "step", "observation": obs})
        response = receive_json(sock)

        assert "dx" in response
        assert "dy" in response
        assert "drone_pos" in response
        assert "goal_reached" in response
        assert "collision" in response
