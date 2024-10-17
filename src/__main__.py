import asyncio
from io import BytesIO
import websockets

from ASR.ASR import ASR

_ASR = ASR("tiny", "auto","int8")

async def handler(websocket):
    while True:
        try:
            # Receiving binary data directly from the client
            data = await websocket.recv()
            #Handle the audion data with Whisper
            _ASR.process_audio(data)
            # Optionally, send an acknowledgment back to the client
            await websocket.send("Chunk received")
        except websockets.ConnectionClosed:
            print("Connection closed")
            break

# Start WebSocket server
start_server = websockets.serve(handler, "127.0.0.1", 3000)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
