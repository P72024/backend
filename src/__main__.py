import asyncio
import json
import os
import pickle
from io import BytesIO
import numpy as np
import websockets
from ASR.ASR import ASR

# Initialize the ASR model
_ASR = ASR("small", "auto", "int8", 1200)

# Store connected clients
connected_clients = set()
rooms = dict()

# Queue for managing audio chunks from clients
audio_queue = asyncio.Queue()

async def process_audio_chunks():
    has_recieved_first_audio_chunk = False
    while True:
        # Process each chunk in the queue one at a time
        client_id, room_id, audio_data = await audio_queue.get()

        if not has_recieved_first_audio_chunk:
            _ASR.save_metadata(audio_data)
            has_recieved_first_audio_chunk = True
            continue

        transcribed_text = _ASR.receive_audio_chunk(audio_data)

        if transcribed_text:
            print(rooms)
            print(f"[BACKEND] Transcribed for client {client_id} in room {room_id}: {transcribed_text}")
            
            # Send the transcribed text to all connected clients in that room
            for (client_id, websocket) in list(rooms[room_id]):
                try:
                    await websocket.send(transcribed_text)
                except websockets.ConnectionClosed:
                    print(f"[BACKEND] Client disconnect unexpected.")
                    connected_clients.remove(websocket)
                    rooms[room_id].remove((client_id, websocket))
                    print(rooms)

async def handler(websocket):
    print("[BACKEND] New client connected!")
    connected_clients.add(websocket)
    
    try:
        async for message in websocket:
            # Parse incoming message
            data = json.loads(message)
            client_id = data.get('clientId')
            room_id = data.get('roomId')
            audio_data = bytes(data.get('audioData'))

            if room_id not in rooms:
                print("creating room: " + room_id)
                rooms[room_id] = [(client_id, websocket)]
                print(rooms)

            elif (client_id, websocket) not in rooms[room_id]:
                print("adding client to room: " + room_id)
                rooms[room_id].append((client_id, websocket))
            
            # Add audio chunk to the queue for processing
            await audio_queue.put((client_id, room_id, audio_data))

    except websockets.ConnectionClosed:
        print(f"[BACKEND] Client disconnected.")
    finally:
        print(f"[BACKEND] Client disconnected.")
        connected_clients.remove(websocket)
        rooms[room_id].remove((client_id, websocket))
        print(rooms)

async def save_chunk(data):
    # Check if file exists, and initialize it if not
    if os.path.exists('../benchmarking/testfiles/frederik.pkl'):
        with open('../benchmarking/testfiles/frederik.pkl', 'rb') as f:
            array = pickle.load(f)
    else:
        array = []

    # Append the new data and save back
    array.append(data)
    with open('../benchmarking/testfiles/frederik.pkl', 'wb') as f:
        pickle.dump(array, f)

async def save_metadata(data):
    # Check if file exists, and initialize it if not
    if os.path.exists('../benchmarking/testfiles/frederik_meta.pkl'):
        with open('../benchmarking/testfiles/frederik_meta.pkl', 'rb') as f:
            array = pickle.load(f)
    else:
        array = []

    # Append the new data and save back
    array.append(data)
    with open('../benchmarking/testfiles/frederik_meta.pkl', 'wb') as f:
        pickle.dump(array, f)# Start WebSocket server
        
start_server = websockets.serve(handler, "127.0.0.1", 3000)
print("[BACKEND] READY!")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()

async def main():
    # Start the websocket server
    async with websockets.serve(handler, "localhost", 3000):
        print("[BACKEND] READY!")
        
        await process_audio_chunks()

if __name__ == "__main__":
    asyncio.run(main())
