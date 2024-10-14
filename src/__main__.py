import asyncio
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
import aiohttp_cors
import json

peerConnections = set()

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    con = RTCPeerConnection()
    peerConnections.add(con)

    @con.on("connectionstatechange")
    async def on_connectionstatechange():
        if con.connectionState == "failed":
            await con.close()
            peerConnections.discard(con)

    @con.on('datachannel')
    def on_datachannel(channel):
        print(f"New data channel: {channel.label}")
        
        @channel.on("message")
        def on_message(message):
            print(f"Received message on channel {channel.label}: {message}")
            print(f"Type of message: {type(message)}")

    await con.setRemoteDescription(offer)
    answer = await con.createAnswer()
    await con.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": con.localDescription.sdp, "type": con.localDescription.type}
        ),
    )

async def on_shutdown(app):
    coros = [pc.close() for pc in peerConnections]
    await asyncio.gather(*coros)
    peerConnections.clear()

if __name__ == "__main__":
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_post("/offer", offer)
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })
    for route in list(app.router.routes()):
        cors.add(route)
    web.run_app(app, host="127.0.0.1", port=3000)
