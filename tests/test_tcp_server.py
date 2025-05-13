import asyncio
import json
import pytest
import websockets

HOST = "localhost"
PORT = 8765
WS_URL = f"ws://{HOST}:{PORT}"

async def send_and_receive(ws, obj):
    await ws.send(json.dumps(obj))
    response = await ws.recv()
    return json.loads(response)

@pytest.mark.asyncio
async def test_init_episode():
    async with websockets.connect(WS_URL) as ws:
        response = await send_and_receive(ws, {"command": "init_episode"})

        assert "field_size" in response
        assert "drone" in response
        assert "target" in response
        assert "obstacles" in response

@pytest.mark.asyncio
async def test_step():
    async with websockets.connect(WS_URL) as ws:
        await send_and_receive(ws, {"command": "init_episode"})

        obs = [0.0, 1.0, 1.0, 0.5]
        response = await send_and_receive(ws, {"command": "step", "observation": obs})

        assert "dx" in response
        assert "dy" in response
        assert "drone_pos" in response
        assert "goal_reached" in response
        assert "collision" in response
