import asyncio
import json
import os
import pickle
from io import BytesIO

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

            # print("client_id: ", client_id)

            if has_received_first_byte:
            #Handle the audion data with Whisper
                _ASR.receive_audio_chunk(audio_data)
                await save_chunk(audio_data)
            else: 
                has_received_first_byte = True
                await save_metadata(audio_data)
                _ASR.save_metadata(audio_data)

            if transcribed_text != '':
                print("sending to client: ", client_id)
                await websocket.send(transcribed_text)
    except websockets.ConnectionClosed:
        print(f"Client disconnected")
    finally:
        # Remove the client from the set when it disconnects
        connected_clients.remove(websocket)

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
