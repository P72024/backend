import asyncio
import json
from io import BytesIO
import socketio
from whisper import transcribe
import numpy as np
from aiohttp import web
import aiohttp_cors
import uuid
import pprint
# A buffer to hold audio chunks
audio_chunks = []
rooms_list = []
username = ""
socket = socketio.AsyncServer(
    async_mode='aiohttp',
    cors_allowed_origins=['http://localhost:3000']
)
app = web.Application()
socket.attach(app)


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

@socket.on("BE-create-username")
async def create_username(sid, _username):
    await socket.save_session(sid, _username)
    print(f"{_username} saved for session {sid}")
    await socket.emit("username-confirmation", f"Username '{_username}' saved", to=sid)


# TODO: return
@socket.on("BE-get-session")
async def send_session(sid):
    session_data = await socket.get_session(sid)
    print(f"frontend requested session data. sending... \nsid: {sid}, session data: {session_data}")
    await socket.emit("FE-session-data", session_data, to=sid)
    

@socket.on("BE-enter-room")
async def enter_room(sid, room_uuid):
    print(f"sid: {sid} enter room: {room_uuid}")
    async with socket.session(sid) as session:
        session["room_uuid"] = room_uuid["uuid"]
    await socket.enter_room(sid, room=room_uuid["uuid"])
    return "OK"
    

async def send_text(sid, text):
    session_data = await socket.get_session(sid)
    await socket.emit("FE-receive-text", data=text, room=session_data["room_uuid"])
                   

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