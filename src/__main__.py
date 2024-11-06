import asyncio
from io import BytesIO

import websockets
from ASR.ASR import ASR

_ASR = ASR("small", "auto","int8", 1200)


async def handler(websocket):
    has_received_first_byte = False

    while True:
        try:
            # Receiving binary data directly from the client
            data = await websocket.recv()

            if has_received_first_byte:
            #Handle the audion data with Whisper
                
                _ASR.receive_audio_chunk(data)
            else: 
                has_received_first_byte = True
                _ASR.save_metadata(data)

            # Optionally, send an acknowledgment back to the client
            await websocket.send("Chunk received")
        except websockets.ConnectionClosed:
            print("Connection closed")
            break

# Start WebSocket server
start_server = websockets.serve(handler, "127.0.0.1", 3000)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
