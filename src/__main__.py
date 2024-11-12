import asyncio
from io import BytesIO
import json
import numpy as np
import websockets
from ASR.ASR import ASR

# Initialize the ASR model
_ASR = ASR("small", "auto", "int8", 1200)

# Store connected clients
connected_clients = set()

# Queue for managing audio chunks from clients
audio_queue = asyncio.Queue()

async def process_audio_chunks():
    has_recieved_first_byte = False
    while True:
        # Process each chunk in the queue one at a time
        client_id, audio_data = await audio_queue.get()

        if not has_recieved_first_byte:
            _ASR.save_metadata(audio_data)
            has_recieved_first_byte = True
            continue

        transcribed_text = _ASR.receive_audio_chunk(audio_data)

        if transcribed_text:
            print(f"[BACKEND] Transcribed for client {client_id}: {transcribed_text}")
            
            # Send the transcribed text to all connected clients
            for client in connected_clients:
                await client.send(transcribed_text)
                
async def handler(websocket):
    print("[BACKEND] New client connected!")
    connected_clients.add(websocket)
    
    try:
        async for message in websocket:
            # Parse incoming message
            data = json.loads(message)
            client_id = data.get('clientId')
            audio_data = bytes(data.get('audioData'))
            
            # Add audio chunk to the queue for processing
            await audio_queue.put((client_id, audio_data))

    except websockets.ConnectionClosed:
        print(f"[BACKEND] Client disconnected.")
    finally:
        connected_clients.remove(websocket)

async def main():
    # Start the websocket server
    async with websockets.serve(handler, "localhost", 3000):
        print("[BACKEND] READY!")
        
        await process_audio_chunks()

if __name__ == "__main__":
    asyncio.run(main())
