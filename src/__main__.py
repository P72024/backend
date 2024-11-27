import asyncio
import json
import os
import pickle
import websockets
import logging
import numpy as np

from ASR.ASR import ASR

# Configure the logging settings
logging.basicConfig(
    level=logging.INFO,  # Set the minimum log level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    handlers=[
        logging.FileHandler("logs/app.log", mode='a'),  # Append mode ('a')
        logging.StreamHandler()  # Also output logs to the console
    ]
)

# Create a logger instance
logger = logging.getLogger(__name__)

# Initialize the ASR model
_ASR = ASR("tiny.en", device="auto", compute_type="int8", max_context_length=100)

connected_clients = set()
rooms = dict()

# Queue for managing audio chunks from clients
audio_queue = asyncio.Queue()

async def process_audio_chunks():
    while True:
        # Process each chunk in the queue one at a time
        client_id, room_id, np_audio = await audio_queue.get()

        logging.warning(f"Transcribing audio chunk for client {client_id} in room {room_id}")
        transcribed_text = _ASR.process_audio(np_audio)

        if transcribed_text:
            logging.info(f"Transcribed for client {client_id} in room {room_id}: {transcribed_text}")
            
            # Send the transcribed text to all connected clients in that room
            for (client_id, websocket) in list(rooms[room_id]):
                try:
                    logging.info(f"Sending transcribed text to client {client_id} in room {room_id}")
                    await websocket.send(transcribed_text)
                except websockets.ConnectionClosed:
                    logging.warning("Client disconnected.")
                    connected_clients.remove(websocket)
                    leave_room(room_id, client_id, websocket)
                    print(rooms)

async def handler(websocket):
    logging.info("New client connected!")
    connected_clients.add(websocket)

    try:
        async for message in websocket:
            data = json.loads(message)

            client_id = data.get('clientId')
            room_id = data.get('roomId')
            audio_data = data.get('audioData')
            np_audio = np.array(audio_data, dtype=np.float32)

            join_room(room_id, client_id, websocket)

            
            # Add audio chunk to the queue for processing
            await audio_queue.put((client_id, room_id, np_audio))
            logging.info(f"Added audio chunk to queue for client {client_id} in room {room_id}")

    except websockets.ConnectionClosed:
        logging.info("Client disconnected exception caught.")
    finally:
        logging.info("Client disconnected gracefully. (in finally block)")
        connected_clients.remove(websocket)
        leave_room(room_id, client_id, websocket)
        print(rooms)

def join_room(room_id, client_id, websocket):
    if room_id not in rooms:
        logging.info(f"Client {client_id} is joining room {room_id}")
        logging.log(logging.WARNING, f"Room {room_id} does not exist. Creating new room.")
        rooms[room_id] = [(client_id, websocket)]
    elif (client_id, websocket) not in rooms[room_id]:
        logging.info(f"Client {client_id} is joining an existing room {room_id}")
        rooms[room_id].append((client_id, websocket))

def leave_room(room_id, client_id, websocket):
    if room_id in rooms:
        rooms[room_id].remove((client_id, websocket))
        logging.info(f'Client {client_id} left room {room_id}')

        if (rooms[room_id] == []):
            del rooms[room_id]
            logging.log(logging.WARNING, f'room {room_id} has been deleted')
        
        logging.info(f"Rooms: {rooms}")

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

async def main():
    # Start the websocket server
    async with websockets.serve(handler, "0.0.0.0", 3000, ping_interval=20, ping_timeout=10):
        logging.info("Server is ready!")

        await process_audio_chunks()

if __name__ == "__main__":
    asyncio.run(main())
