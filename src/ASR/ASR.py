from asyncio import sleep
from io import BytesIO
import re
from typing import List
import wave
from ASR.LocalAgreement import LocalAgreement
from faster_whisper import WhisperModel 
import soundfile as sf
from pydub import AudioSegment

class ASR:
    max_context_length = 100
    metadata: BytesIO = BytesIO()
    audio_buffer: List[BytesIO] = []
    local_agreement = LocalAgreement()
    context:str = ""
    confirmed_sentences: List[str] = []
    min_chunk_size = 3

    def __init__ (self, model_size: str, device="auto", compute_type = "int8", max_context_length=100):
        self.whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.max_context_length = max_context_length
        
    def transcribe(self, audio_buffer: BytesIO, context: str):
        print(audio_buffer.getbuffer().nbytes)

        audio_buffer.seek(0)
        # a lil debug tang
        with open("temp.webm", 'wb') as f:
            f.write(audio_buffer.getvalue())

        audio_buffer.seek(0)
        
        transcribed_text = ""
        segments, info = self.whisper_model.transcribe(audio_buffer, beam_size=5, initial_prompt=context)
        
        for segment in segments:
            transcribed_text += " " + segment.text
            
        return transcribed_text
    
    def save_metadata(self, metadata):
        print("saving metadata")
        self.metadata.write(metadata)
        self.metadata.seek(0)  # Reset buffer's position to the beginning

    def recieve_audio_chunk(self, audio_chunk):
        print("recieving audio chunk")
        self.audio_buffer.append(BytesIO(audio_chunk))

        print("audio buffer length: ", len(self.audio_buffer))
        print("min chunk size: ", self.min_chunk_size)
        if len(self.audio_buffer) > self.min_chunk_size:
            self.process_audio()
    
    def process_audio(self) -> str:
        print("processing audio")
        # transcribed_text = self.transcribe(BytesIO(self.metadata.read() + self.audio_buffer.read()), self.context)
        combined_bytes = self.metadata.getvalue() + b''.join(bio.getvalue() for bio in self.audio_buffer)
        combined_bytes_io = BytesIO(combined_bytes)
        
        transcribed_text = self.transcribe(combined_bytes_io, self.context)
        print(transcribed_text)
        confirmed_text = self.local_agreement.confirm_tokens(transcribed_text)
        #print(confirmed_text)
        punctuation = r"[.!?]"  # Regular expression pattern for ., !, or ?
        # Detect punctuation

        # if len(self.context) > self.max_context_length:
        #     self.confirmed_sentences = self.confirmed_sentences[-2:]
        #     self.context = " ".join(self.confirmed_sentences)
        #     #print("context truncated: " + self.context)

        #print("check punctuation: ", re.search(punctuation,confirmed_text))
        if re.search(punctuation,confirmed_text):
            split_sentence = re.split(f"({punctuation})", confirmed_text)

            # # Join the punctuation back to the respective parts of the sentence
            sentence = [split_sentence[i] + split_sentence[i+1] for i in range(0, len(split_sentence)-1, 2)]

            #print("sentence", sentence)
            self.confirmed_sentences.append(sentence[-1])
            # self.context = " ".join(self.confirmed_sentences)
            # print("context added: " + self.context)
            
            self.local_agreement.clear_confirmed_text()
            
            self.audio_buffer = []
            
        return confirmed_text