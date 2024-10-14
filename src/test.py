from flask import Flask, request, jsonify
from flask_cors import CORS
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from av import AudioFrame
import numpy as np
import asyncio

app = Flask(__name__)
CORS(app)

class AudioProcessor(MediaStreamTrack):
    kind = "audio"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.chunk_size = 960  # 20ms at 48kHz

    async def recv(self):
        frame = await self.track.recv()
        
        # Convert the frame to numpy array
        audio_data = frame.to_ndarray()
        
        # Process audio in chunks
        for i in range(0, len(audio_data), self.chunk_size):
            chunk = audio_data[i:i+self.chunk_size]
            
            # Here you can process each chunk as needed
            # For example, let's just print the RMS of each chunk
            rms = np.sqrt(np.mean(chunk**2))
            print(f"Chunk RMS: {rms}")

        # Return the original frame
        return frame

pcs = set()

@app.route('/offer', methods=['POST'])
async def offer(request):
    params = await request.json  # This is synchronous
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("track")
    def on_track(track):
        print(f"Received {track.kind} track")
        if track.kind == "audio":
            audio_processor = AudioProcessor(track)
            pc.addTrack(audio_processor)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return jsonify({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })

@app.route('/shutdown', methods=['POST'])
async def shutdown():
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)
