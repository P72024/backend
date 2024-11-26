import asyncio
import json
import os
import pickle
from io import BytesIO

import websockets

from ASR.ASR import ASR
import logging

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
_ASR = ASR("tiny", device="auto", compute_type="int8", max_context_length=100, chunk_limit=3)

connected_clients = dict()
rooms = dict()
room_admins = dict()

# Queue for managing audio chunks from clients
audio_queue = asyncio.Queue()

async def process_audio_chunks():
    has_recieved_first_audio_chunk = False

    while True:
        # Process each chunk in the queue one at a time
        client_id, room_id, audio_data = await audio_queue.get()

        if not has_recieved_first_audio_chunk:
            logging.info("Saving metadata for the first audio chunk.")
            _ASR.save_metadata(audio_data)
            has_recieved_first_audio_chunk = True
            continue

        logging.warning(f"Transcribing audio chunk for client {client_id} in room {room_id}")
        print(f"Transcribing audio chunk for client {client_id} in room {room_id}")
        transcribed_text = _ASR.receive_audio_chunk(audio_data)

        if transcribed_text:
            logging.info(f"Transcribed for client {client_id} in room {room_id}: {transcribed_text}")
            
            # Send the transcribed text to all connected clients in that room
            for (client_id, websocket) in list(rooms[room_id]):
                try:
                    logging.info(f"Sending transcribed text to client {client_id} in room {room_id}")
                    await websocket.send(transcribed_text)
                except websockets.ConnectionClosed:
                    logging.warning("Client disconnected.")
                    connected_clients.pop(client_id)
                    await leave_room(room_id, client_id, websocket)
                    print(rooms)

async def handler(websocket):
    logging.info("New client connected!")

    try:
        async for message in websocket:
            data = json.loads(message)
            
            match data["type"]:
                case 'disconnecting':
                    await disconnecting(data)
                case 'leave room':
                    room_id = data.get(room_id)
                    client_id = data.get("clientId")
                    await leave_room(room_id, client_id, websocket)
                case 'kickout':
                    await kickout(data)
                case 'message':
                    await handle_message(data)
                case 'create or join':
                    room_id = data.get(room_id)
                    client_id = data.get("clientId")
                    logging.log(f"Create or join room: {room_id}")
                    await join_room(room_id, client_id, websocket)
                case _:
                    break

            
            transcribed_text = ''


            client_id = data.get('clientId')
            room_id = data.get('roomId')
            audio_data = bytes(data.get('audioData'))

            await join_room(room_id, client_id, websocket)
            
            # Add audio chunk to the queue for processing
            await audio_queue.put((client_id, room_id, audio_data))


    except websockets.ConnectionClosed:
        logging.info("Client disconnected exception caught.")
    finally:
        logging.info("Client disconnected gracefully. (in finally block)")
        connected_clients.pop(client_id)
        await leave_room(room_id, client_id, websocket)
        print(rooms)

async def disconnecting(data):
    room_id = data.get("roomId")
    client_id = data.get("clientId")
    if room_id not in rooms:
        return
    await broadcast({
        "message": client_id,
        "type": "leave"
    })


async def kickout(data):
    room_id = data.get("roomId")
    client_to_kick_id = data.get("clientToKickId")
    client_id = data.get("clientId")
    
    if client_id in room_admins:
        await broadcast({
            "message": client_to_kick_id,
            "type": "kickout",
        }, room_id)
        client_to_kick = next(((key, value) for key, value in connected_clients.items() if client_to_kick_id == key))
        rooms[room_id].remove(client_to_kick)
    else:
        logging.log(f"{client_id} is not an admin")

async def handle_message(data):
    to_id = data.get("toId")
    client_id = data.get("clientId")
    room_id = data.get("roomId")
    message_type = data.get("type")

    if to_id:
        logging.log(f"From {client_id} to {to_id} {message_type}")
        await connected_clients.get(to_id).send({
            "message": f"Message: {data}, {client_id}",
            "type": "message"
        })
    elif room_id:
        logging.log(f"From {client_id} to room {room_id} {message_type}")
        await broadcast({
            "message": f"Message {data} {client_id}",
            "type": "message",
        }, room_id)
    else:
        logging.log(f"From {client_id} to everyone {message_type}")
        await broadcast({
            "message": f"Message {data} {client_id}",
            "type": "message",
        })

async def broadcast(message, room_id=None):
    if room_id is not None:
        for client_id, websocket in rooms[room_id]:
            try:
                await websocket.send(message)
            except websockets.ConnectionClosed:
                logging.warning("Client disconnected.")
                connected_clients.pop(client_id)
                await leave_room(room_id, client_id, websocket)
                print(rooms)
    else:
        for client_id, websocket in connected_clients.items():
            try:
                await websocket.send(message)
            except websockets.ConnectionClosed:
                logging.warning("Client disconnected.")
                connected_clients.pop(client_id)
                room_id = next((key for key, value in rooms.items() if client_id in value), None)
                if room_id is not None:
                    await leave_room(room_id, client_id, websocket)
                    print(rooms)



async def join_room(room_id, client_id, websocket):
    connected_clients.update({ client_id: websocket })

    if room_id not in rooms:
        logging.info(f"Client {client_id} is joining room {room_id}")
        logging.log(logging.WARNING, f"Room {room_id} does not exist. Creating new room.")
        rooms[room_id] = [(client_id, websocket)]
        room_admins[room_id] = client_id
        websocket.send({
            "message": {
                "room_id": room_id, 
                "client_id": client_id
            },
            "type": "created"
        })

    elif (client_id, websocket) not in rooms[room_id]:
        logging.info(f"Client {client_id} is joining an existing room {room_id}")
        await broadcast({
            "message": room_id,
            "type": "join",
        }, room_id)
        rooms[room_id].append((client_id, websocket))
        websocket.send({
            "message": {
                "room_id": room_id,
                "client_id": client_id
            },
            "type": "joined",
        })
        await broadcast({
            "message": client_id,
            "type": "ready",
        }, room_id)

async def leave_room(room_id, client_id, websocket):
    if room_id in rooms:
        rooms[room_id].remove((client_id, websocket))
        logging.info(f'Client {client_id} left room {room_id}')
        websocket.send({
            "message": room_id,
            "type": "left room",
        })
        await broadcast({
            "message": client_id,
            "type": "leave"
        }, room_id)

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
