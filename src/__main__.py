import asyncio
import json
from io import BytesIO
import socketio
from whisper import transcribe
import numpy as np
from aiohttp import web
import aiohttp_cors
import uuid
# A buffer to hold audio chunks
audio_chunks = []
rooms_list = []
username = ""
sio = socketio.AsyncServer(
    async_mode='aiohttp',
    cors_allowed_origins=['http://localhost:3000']
)
app = web.Application()
sio.attach(app)


# async def handler(websocket, path):
#     global audio_chunks
#     while True:
#         try:
#             # Receiving binary data directly from the client
#             data = await websocket.recv()

#             audio_chunks.append(data)  # Collect binary data chunks
#             audio_binary_io = BytesIO(b''.join(audio_chunks))
#             # array_data = np.frombuffer(data, dtype=np.int8)  # Check if the data is audio
#             transcribe(audio_binary_io)

#             # Optionally, send an acknowledgment back to the client
#             await websocket.send("Chunk received")
#         except websockets.ConnectionClosed:
#             print("Connection closed")
#             break




async def create_room_uuid(request):
    print("received request for create_room_uuid")
    roomID = str(uuid.uuid4())
    print(f"room_id: {roomID}")
    if roomID not in rooms_list:
        rooms_list.append(roomID)
        return web.Response(
            text=roomID
        )
    else:
        create_room_uuid()

@sio.on("BE-create-username")
async def create_username(sid, _username):
    await sio.save_session(sid, _username)
    print(f"{_username} saved for session {sid}")
    await sio.emit("username-confirmation", f"Username '{_username}' saved", to=sid)

@sio.on("BE-get-session")
async def send_session(sid):
    session_data = await sio.get_session(sid)
    print(f"sid: {sid}, session data: {session_data}")
    await sio.emit("FE-session-data", session_data, to=sid)
    

cors = aiohttp_cors.setup(app)
resource = cors.add(app.router.add_resource("/createRoomUUID"))
cors.add(resource.add_route("GET", create_room_uuid), {
    "*": aiohttp_cors.ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_headers="*",
    )
})


if __name__ == "__main__":
    web.run_app(app, port=8080, host="localhost")