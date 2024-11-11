import asyncio
from io import BytesIO
import json

import websockets
from ASR.ASR import ASR

_ASR = ASR("small", "auto","int8", 1200)

connected_clients = set()



async def handler(websocket):
    has_received_first_byte = False
    print("[BACKEND] New client connected!")
    connected_clients.add(websocket)
    try:
        async for message in websocket:
            transcribed_text = ''

            data = json.loads(message)

            client_id = data.get('clientId')
            audio_data = data.get('audioData')
            audio_data = bytes(audio_data)

            print("client_id: ", client_id)

            if has_received_first_byte:
                transcribed_text = _ASR.receive_audio_chunk(audio_data)
            else:
                _ASR.save_metadata(audio_data)
                has_received_first_byte = True

            if transcribed_text != '':
                print("sending to client: ", client_id)
                await websocket.send(transcribed_text)
    except websockets.ConnectionClosed:
        print(f"Client disconnected")
    finally:
        # Remove the client from the set when it disconnects
        connected_clients.remove(websocket)

async def main():
    async with websockets.serve(handler, "localhost", 3000):
        print("[BACKEND] READY!")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())