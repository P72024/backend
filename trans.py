import queue
import threading

import speech_recognition as sr
import whisper

# Load Whisper model (you can also use 'tiny' or 'small' for faster performance)
model = whisper.load_model("base", device='cpu')

# Queue to hold audio chunks
audio_queue = queue.Queue()

# Function to transcribe audio chunks
def transcribe_audio():
    while True:
        # Get audio chunk from the queue
        audio_data = audio_queue.get()

        # Save audio chunk to a temporary file
        with open("temp_chunk.wav", "wb") as f:
            f.write(audio_data)

        # Transcribe the chunk
        result = model.transcribe("temp_chunk.wav", fp16=False, language='English')
        print(f"Transcription: {result['text']}")

        # Indicate task completion
        audio_queue.task_done()

# Start the transcription thread
transcription_thread = threading.Thread(target=transcribe_audio, daemon=True)
transcription_thread.start()

# Function to capture and process audio chunks
def real_time_transcription():
    recognizer = sr.Recognizer()

    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        print("Calibrating microphone for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Microphone ready! Start speaking...")

        try:
            while True:
                # print("Listening for a short chunk...")

                # Capture audio chunk
                audio = recognizer.listen(source, phrase_time_limit=1)  # Use short chunks of 2 seconds

                # Put audio data into the queue for background transcription
                audio_queue.put(audio.get_wav_data())

        except KeyboardInterrupt:
            print("Stopped listening.")

# Run the real-time transcription
real_time_transcription()

