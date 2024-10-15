import asyncio
import json
from io import BytesIO
import websockets
from whisper import transcribe
import numpy as np
from aiohttp import web
import aiohttp_cors
import uuid
# A buffer to hold audio chunks
audio_chunks = []
rooms_list = []

async def handler(websocket, path):
    global audio_chunks
    while True:
        try:
            # Receiving binary data directly from the client
            data = await websocket.recv()

            audio_chunks.append(data)  # Collect binary data chunks
            audio_binary_io = BytesIO(b''.join(audio_chunks))
            # array_data = np.frombuffer(data, dtype=np.int8)  # Check if the data is audio
            transcribe(audio_binary_io)

            # Optionally, send an acknowledgment back to the client
            await websocket.send("Chunk received")
        except websockets.ConnectionClosed:
            print("Connection closed")
            break


async def createRoomUUID(request):
    roomID = str(uuid.uuid4())
    if roomID not in rooms_list:
        rooms_list.append(roomID)
        return web.Response(
            text=roomID
        )
    else:
        createRoomUUID
    

# Start WebSocket server

start_server = websockets.serve(handler, "127.0.0.1", 3000)
app = web.Application()
app.router.add_get("/createRoomUUID", createRoomUUID)
cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })
for route in list(app.router.routes()):
    cors.add(route)
web.run_app(app)
print("Backend is up and running on 127.0.0.1:3000")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
