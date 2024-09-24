import asyncio
import json
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRecorder

# Store peer connections
pcs = set()

async def webrtc_handler(request):
    params = await request.json()

    if 'offer' in params:
        # Create RTC peer connection
        pc = RTCPeerConnection()
        pcs.add(pc)

        # Create a media recorder to save audio
        recorder = MediaRecorder('output.wav')

        @pc.on('track')
        async def on_track(track):
            print(f'Received track: {track.kind}')
            if track.kind == 'audio':
                recorder.addTrack(track)

        # Set remote description
        offer = RTCSessionDescription(sdp=params['offer']['sdp'], type=params['offer']['type'])
        await pc.setRemoteDescription(offer)

        # Create and return answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.json_response({'answer': pc.localDescription.sdp, 'type': pc.localDescription.type})

    elif 'candidate' in params:
        candidate = params['candidate']
        await pc.addIceCandidate(candidate)
        return web.Response(text='Candidate added')

async def cleanup():
    # Clean up peer connections
    while True:
        await asyncio.sleep(10)
        for pc in pcs:
            if pc.iceConnectionState == 'closed':
                pcs.remove(pc)

app = web.Application()
app.add_routes([web.post('/webrtc', webrtc_handler)])

# Run cleanup in the background
asyncio.ensure_future(cleanup())

web.run_app(app, port=5000)