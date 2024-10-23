import asyncio
import json
from io import BytesIO
import socketio
from whisper import transcribe
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
import av
from aiohttp import web
import aiohttp_cors
import uuid
import pprint
# A buffer to hold audio chunks
peerconnections = {}
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

class AudioRelayTrack(MediaStreamTrack):
    """
    This track relays audio from an incoming audio track to the outgoing track.
    """
    kind = "audio"

    def __init__(self, track):
        super().__init__()  # don't forget this!
        self.track = track

    async def recv(self):
        # Receive the next frame of audio from the incoming track
        frame = await self.track.recv()
        return frame

@socket.on("BE-send-offer")
async def send_offer(sid, offer):
    peerconnections[sid] = RTCPeerConnection()

    # This will store the audio relay track
    audio_relay_track = None

    pc = peerconnections[sid]

    @pc.on("track")
    async def on_track(track):
        if track.kind == "audio":
            print("Received audio track")
            # Create a relay track that sends audio to the other connection
            audio_relay_track = AudioRelayTrack(track)
            for other_pc_sid, other_pc in peerconnections.items():
                other_pc.addTrack(audio_relay_track)

    session_description = RTCSessionDescription(sdp=offer["sdp"], type=offer["type"])
    await peerconnections[sid].setRemoteDescription(session_description)
    answer = await peerconnections[sid].createAnswer()
    await peerconnections[sid].setLocalDescription(answer)
    if peerconnections[sid].localDescription is not None:
        await socket.emit("FE-send-answer", {
            "sdp": peerconnections[sid].localDescription.sdp,
            "type": peerconnections[sid].localDescription.type,
        })


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


@socket.on("BE-send-room-uuid")
async def send_room_uuid(sid):
    if not rooms_list:
        return "no-rooms-available"
    else: 
     return rooms_list[0]


# TODO: return
@socket.on("BE-get-session")
async def send_session(sid):
    session_data = await socket.get_session(sid)
    print(f"frontend requested session data. sending... \nsid: {sid}, session data: {session_data}")
    await socket.emit("FE-session-data", session_data, to=sid)
    await send_text(sid, "transcription from whisper")
    

@socket.on("BE-enter-room")
async def enter_room(sid, room_uuid):
    print(f"sid: {sid} enter room: {room_uuid}")
    async with socket.session(sid) as session:
        session["room_uuid"] = room_uuid["uuid"]
    await socket.enter_room(sid, room=room_uuid["uuid"])
    return "OK"
    


async def send_text(sid, text):
    session_data = await socket.get_session(sid)
    print("sending text....\n")
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