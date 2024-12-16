import asyncio
import json
import logging
import os
import pickle
import re
import time
import uuid
from io import BytesIO

import numpy as np
import websockets

from ASR.ASR import ASR
from ASR.tweaked import ASR_tweaked
from Util import unix_seconds_to_ms

if not os.path.exists("logs"):
    os.makedirs("logs")

# make an empty app.log file in the logs folder if it doesn't exist
if not os.path.exists("logs/app.log"):
    with open("logs/app.log", "w") as f:
        f.write("")

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
_ASR = ASR_tweaked()

connected_clients = dict()
rooms = dict()
room_admins = dict()

# Queue for managing audio chunks from clients
audio_queue = asyncio.Queue()

async def process_audio_chunks_to_pkl():
    while True:
        client_id, _, room_id, np_audio, _, _ = await audio_queue.get()

        regex = r":(\d+(\.\d+)?)$"
        min_chunk_size = int(re.search(regex, client_id).group(1))
        speech_threshold = float(re.search(regex, room_id).group(1))

        try:
            await save_chunk(np_audio, min_chunk_size, speech_threshold)

        except Exception as e:
            logging.error(f"Error saving audio chunk: {e}")


async def process_audio_chunks():
    while True:
        # Process each chunk in the queue one at a time
        client_id, screen_name, room_id, np_audio, sentAt, receivedAt = await audio_queue.get()

        audio_queue_wait_time = unix_seconds_to_ms(time.time()) - receivedAt
        processing_start_time = unix_seconds_to_ms(time.time())

        logging.warning(f"Transcribing audio chunk for client {client_id} in room {room_id}")
        (transcribed_text, transcribe_time, update_context_time) = _ASR.process_audio(np_audio, room_id)

        if transcribed_text:
            logging.info(f"Transcribed for client {client_id} in room {room_id}: {transcribed_text}. Took {unix_seconds_to_ms(time.time()) - processing_start_time}ms")
            
            # Send the transcribed text to all connected clients in that room
            for (client_id, websocket) in list(rooms[room_id]):
                try:
                    websocket_send_start_time = unix_seconds_to_ms(time.time())
                    logging.info(f"Sending transcribed text to client {client_id} in room {room_id}")

                    await websocket.send(json.dumps({
                        "message": transcribed_text,
                        "screen_name": screen_name,
                        "type": "transcribed text",
                        "sentAt": sentAt,
                        "receivedAt": receivedAt,
                        "processingTime": {
                            "total": unix_seconds_to_ms(time.time()) - processing_start_time,
                            "transcription_time": transcribe_time,
                            "update_context_time": update_context_time,
                        },
                        "queueWaitTime": audio_queue_wait_time,
                    }))

                    logging.info(f"Sent transcribed text to client {client_id} in room {room_id}. Took {unix_seconds_to_ms(time.time()) - websocket_send_start_time}ms")
                except websockets.ConnectionClosed:
                    logging.warning("Client disconnected.")
                    if client_id in connected_clients:
                        connected_clients.pop(client_id)
                    await disconnecting(room_id, client_id, websocket)

async def handler(websocket):
    logging.info("New client connected!")

    try:
        async for message in websocket:
            data = json.loads(message)
            type = data.get("type")
            
            match type:
                case 'create id':
                    await create_id(websocket)
                case 'disconnecting':
                    room_id = data.get("roomId")
                    client_id = data.get("clientId")
                    await disconnecting(room_id, client_id, websocket)
                case 'leave room':
                    room_id = data.get("roomId")
                    client_id = data.get("clientId")
                    await leave_room(room_id, client_id, websocket)
                case 'kickout':
                    await kickout(data)
                case 'message':
                    await handle_message(data)
                case 'create or join':
                    room_id = data.get("roomId")
                    client_id = data.get("clientId")
                    logging.info(f"Create or join room: {room_id}")
                    await join_room(room_id, client_id, websocket)
                case 'audio':
                    await get_audio(data)
                case 'ping':
                    await websocket.send(json.dumps(data))
                case 'benchmarking':
                    data["receiveTime"] = unix_seconds_to_ms(time.time())
                    del data["audioData"]
                    await websocket.send(json.dumps(data))
                case _:
                    logging.warning(f"Incorrect type on message: {data}")
    except websockets.ConnectionClosed:
        logging.info("Client connection closed")
        client_id = next((key for key, value in connected_clients.items() if value == websocket), None)
        room_id = next((key for key, value in rooms.items() if client_id in value), None)
        if client_id in connected_clients:
            connected_clients.pop(client_id)
        await disconnecting(room_id, client_id, websocket)

async def create_id(websocket):
    id = None
    while id is None or id in connected_clients:
        id = str(uuid.uuid4())
    connected_clients.update({ id: websocket })
    await websocket.send(json.dumps({'message': id, 'type': 'get id'}))

async def get_audio(data):
    client_id = data.get('clientId')
    screen_name = data.get('screenName', client_id)
    room_id = data.get('roomId')
    audio_data = data.get('audioData')
    sentAt = data.get('sentAt')
    receivedAt = unix_seconds_to_ms(time.time())
    np_audio = np.array(audio_data, dtype=np.float32)
    
    # Add audio chunk to the queue for processing
    await audio_queue.put((client_id, screen_name, room_id, np_audio, sentAt, receivedAt))
    logging.info(f"Added audio chunk to queue for client {client_id} in room {room_id}")

async def disconnecting(room_id, client_id, websocket):
    if room_id not in rooms:
        return
    if (client_id, websocket) in rooms[room_id]:
        rooms[room_id].remove((client_id, websocket))
    if client_id in connected_clients:
        connected_clients.pop(client_id)
    logging.info(f'Client {client_id} left room {room_id}')

    if (len(rooms[room_id]) == 0):
        del rooms[room_id]
        logging.log(logging.WARNING, f'room {room_id} has been deleted')
    else:
        await broadcast({
            "message": client_id,
            "type": "leave"
        }, client_id, room_id) 
    logging.info(f"Connected Clients: {connected_clients}")
    logging.info(f"Rooms: {rooms}")


async def kickout(data):
    room_id = data.get("roomId")
    client_to_kick_id = data.get("clientToKickId")
    client_id = data.get("clientId")
    
    if room_id in room_admins and room_admins[room_id] == client_id:
        await broadcast({
            "message": client_to_kick_id,
            "type": "kickout",
        }, client_id, room_id)
        client_to_kick = next(((key, value) for key, value in connected_clients.items() if client_to_kick_id == key))
        rooms[room_id].remove(client_to_kick)
    else:
        logging.info(f"{client_id} is not an admin")

async def handle_message(data):
    to_id = data.get("toId")
    client_id = data.get("clientId")
    room_id = data.get("roomId")
    message = data.get("message")
    message_type = data.get("type")

    if to_id:
        logging.info(f"From {client_id} to {to_id} {message_type}")
        await connected_clients.get(to_id).send(json.dumps({
            "message": { 
                "message": message, 
                "client_id": client_id,
            },
            "type": "message"
        }))
    elif room_id:
        logging.info(f"From {client_id} to room {room_id} {message_type}")
        await broadcast({
           "message": {
                "message": message,
                "client_id": client_id
            },
            "type": "message",
        }, client_id, room_id)
    else:
        logging.info(f"From {client_id} to everyone {message_type}")
        await broadcast({
            "message": {
                "message": message,
                "client_id": client_id
            },
            "type": "message",
        }, client_id)

async def broadcast(message, from_client_id, room_id=None):
    if room_id is not None:
        for client_id, websocket in rooms[room_id]:
            try:
                if from_client_id != client_id:
                    await websocket.send(json.dumps(message))
            except websockets.ConnectionClosed:
                logging.warning("Client disconnected.")
                if client_id in connected_clients:
                    connected_clients.pop(client_id)
                await disconnecting(room_id, client_id, websocket)
    else:
        for client_id, websocket in connected_clients.items():
            try:
                if from_client_id != client_id:
                    await websocket.send(json.dumps(message))
            except websockets.ConnectionClosed:
                logging.warning("Client disconnected.")
                if client_id in connected_clients:
                    connected_clients.pop(client_id)
                room_id = next((key for key, value in rooms.items() if client_id in value), None)
                if room_id is not None:
                    await disconnecting(room_id, client_id, websocket)

async def join_room(room_id, client_id, websocket):
    if room_id not in rooms:
        logging.info(f"Client {client_id} is joining room {room_id}")
        logging.log(logging.WARNING, f"Room {room_id} does not exist. Creating new room.")
        rooms[room_id] = [(client_id, websocket)]
        room_admins[room_id] = client_id
        await websocket.send(json.dumps({
            "message": {
                "room_id": room_id, 
                "client_id": client_id
            },
            "type": "created"
        }))

    elif (client_id, websocket) not in rooms[room_id]:
        logging.info(f"Client {client_id} is joining an existing room {room_id}")
        await broadcast({
            "message": room_id,
            "type": "join",
        }, client_id, room_id)
        rooms[room_id].append((client_id, websocket))
        await websocket.send(json.dumps({
            "message": {
                "room_id": room_id,
                "client_id": client_id
            },
            "type": "joined",
        }))
        await broadcast({
            "message": client_id,
            "type": "ready",
        }, client_id, room_id)

async def leave_room(room_id, client_id, websocket):
    if room_id in rooms:
        if (client_id, websocket) in rooms[room_id]:
            rooms[room_id].remove((client_id, websocket))
        logging.info(f'Client {client_id} left room {room_id}')
        await websocket.send(json.dumps({
            "message": room_id,
            "type": "left room",
        }))

        if (rooms[room_id] == []):
            del rooms[room_id]
            logging.log(logging.WARNING, f'room {room_id} has been deleted')
        else:
            await broadcast({
                "message": client_id,
                "type": "leave"
            }, client_id, room_id)
        
        logging.info(f"Rooms: {rooms}")

    
def get_absolute_path(relative_path):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), relative_path)


async def save_chunk(data, min_chunk_size, speech_threshold):
    # Check if file exists, and initialize it if not

    file_path = get_absolute_path(f'../benchmarking/testfiles/eval_files2/{min_chunk_size}-{speech_threshold}.pkl')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    

    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            array = pickle.load(f)
    else:
        array = [] 
    array.append(data)

    with open(file_path, 'wb') as f:
        pickle.dump(array, f)


async def main():
    # Start the websocket server
    async with websockets.serve(handler, "0.0.0.0", 3000):
        logging.info("Server is ready!")

        # run this to save audio chunks for benchmarking
        # await process_audio_chunks_to_pkl()

        # run this for standard transcription functionality
        await process_audio_chunks()
if __name__ == "__main__":
    asyncio.run(main())
