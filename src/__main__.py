import asyncio
import json
from io import BytesIO
from faster_whisper import WhisperModel
import socketio
import wave
from whisper import transcribe
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCIceCandidate
from aiortc.contrib.media import MediaRelay, MediaBlackhole
from aiohttp import web
import aiohttp_cors
import uuid
import time
import pprint
import threading
import librosa
import soundfile

from ASR.ASR import ASR
# A buffer to hold audio chunks
peerconnections = {}
relayers = {}
audio_chunks = []
rooms_list = []
username = ""
socket = socketio.AsyncServer(
    async_mode='aiohttp',
    cors_allowed_origins=['http://localhost:3000']
)
app = web.Application()
socket.attach(app)

_ASR = ASR("base", device="cpu", compute_type="int8")

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

class AudioTrack(MediaStreamTrack):
    """
    A media stream track that records audio to a file.
    """
    kind = "audio"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.audio_file = wave.open("temp.wav", "wb")
        self.audio_file.setnchannels(1)  # mono
        self.audio_file.setsampwidth(4)  # 32-bit audio
        self.audio_file.setframerate(16000)  # 48kHz

    async def recv(self):
        #print("recv")
        frame = await self.track.recv()

        converted_audio_data = frame.to_ndarray().astype(np.float32)

        resampled_audio = librosa.resample(converted_audio_data, orig_sr=48000, target_sr=16000)

        max_val = np.max(np.abs(resampled_audio))
        if max_val > 1.0:
            audio_resampled_normalized = resampled_audio / max_val
        else:
            audio_resampled_normalized = resampled_audio

        self.audio_file.writeframes(audio_resampled_normalized.tobytes())

        # Convert frame to bytes
        #audio_data = frame.to_ndarray().tobytes()

        #converted_audio_data = frame.to_ndarray().astype(np.float32)
        #converted_audio_data /= 32768.0

        #self.audio_file.writeframes(audio_data)
        #_ASR.process_audio(resampled_audio)
        return frame

    def stop(self):
        print("stop")
        self.audio_file.close()
        super().stop()

@socket.on("BE-send-offer")
async def send_offer(sid, offer):
    peerconnections[sid] = RTCPeerConnection()
    # relayers[sid] = MediaRelay()
    @(peerconnections[sid]).on("track")
    async def on_track(track):
        if track.kind == "audio":
            print("Received audio track")
            audio_track = AudioTrack(track)
            black_hole = MediaBlackhole()
            black_hole.addTrack(audio_track)
            for pc_sid, pc in peerconnections.items():
                if pc_sid != sid: 
                    pc.addTrack(audio_track)
                    # print(test)
                    # pc.addTrack(relayers[pc_sid].subscribe(track))
            await black_hole.start()

    # @(peerconnections[sid]).on("icecandidate")
    # async def on_icecandidate(event):
    #     candidate = event.candidate
    #     print(candidate)
    #     if candidate:
    #         await socket.emit("FE-new-ice-candidate", {
    #             "candidate": candidate
    #         })
                    

    session_description = RTCSessionDescription(sdp=offer["sdp"], type=offer["type"])
    await peerconnections[sid].setRemoteDescription(session_description)
    answer = await peerconnections[sid].createAnswer()
    await peerconnections[sid].setLocalDescription(answer)
    if peerconnections[sid].localDescription is not None:
        await socket.emit("FE-send-answer", {
            "sdp": peerconnections[sid].localDescription.sdp,
            "type": peerconnections[sid].localDescription.type,
        })

# @socket.on("BE-new-ice-candidate")
# async def new_ice_candidate(sid, candidate):
#     new_candidate = json.loads(candidate["candidate"])
#     ip = new_candidate['candidate'].split(' ')[4]
#     port = new_candidate['candidate'].split(' ')[5]
#     protocol = new_candidate['candidate'].split(' ')[2]
#     priority = new_candidate['candidate'].split(' ')[3]
#     foundation = new_candidate['candidate'].split(' ')[0].split(":")[1]
#     component = new_candidate['candidate'].split(' ')[1]
#     type = new_candidate['candidate'].split(' ')[7]
#     rtc_candidate = RTCIceCandidate(
#             ip=ip,
#             port=port,
#             protocol=protocol,
#             priority=priority,
#             foundation=foundation,
#             component=component,
#             type=type,
#             sdpMid=new_candidate['sdpMid'],
#             sdpMLineIndex=new_candidate['sdpMLineIndex']
#         )
#     await peerconnections[sid].addIceCandidate(rtc_candidate)
    # print("Added ICE candidate")
    # print(peerconnections[sid].iceGatheringState )

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


#transcription_thread = threading.Thread(target=_ASR.start)

whisper_model = WhisperModel("base", device="cpu", compute_type="float32")

if __name__ == "__main__":
    #transcription_thread.start()

    web.run_app(app, port=8080, host="localhost")

    #audio, _ = librosa.load("record.wav", sr=16000, dtype=np.float32)

    #print(audio)

    #_ASR.process_audio(audio)

    #transcription_thread.join()