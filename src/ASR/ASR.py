from io import BytesIO
import re
import time
from typing import List
import wave

import numpy as np
from ASR.LocalAgreement import LocalAgreement
from faster_whisper import WhisperModel 
import soundfile
import os
import librosa

class ASR:
    max_context_length = 100
    local_agreement = LocalAgreement()
    context:str = ""
    confirmed_sentences: List[str] = []
    SAMPLING_RATE = 48000
    audio_buffer = np.array([], dtype=np.float32)


    def __init__ (self, model_size: str, device="auto", compute_type = "int8", max_context_length=100):
        self.whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.max_context_length = max_context_length
        self.audio_file = wave.open("recorded_audio.wav", "wb")
        self.audio_file.setnchannels(1)  # mono
        self.audio_file.setsampwidth(4)  # 16-bit audio
        self.audio_file.setframerate(self.SAMPLING_RATE)  # 48kHz
        
    def transcribe(self, audio_buffer, context: str):
        transcribed_text = ""
        segments, info = self.whisper_model.transcribe(audio_buffer, beam_size=5, initial_prompt=context)
        
        for segment in segments:
            transcribed_text += " " + segment.text
            
        return transcribed_text
    
    def process_audio(self, audio_chunk) -> str: 

        #self.audio_buffer = np.concatenate((self.audio_buffer, audio_chunk))
        # self.buffer += audio_chunk
        # if self.buffer != b"":
        #     sf_buffer = soundfile.SoundFile(BytesIO(self.buffer), samplerate=self.SAMPLING_RATE, channels=1, format='RAW', 
        #                 subtype='PCM_32')

        #     audio, _ = librosa.load(sf_buffer, sr=self.SAMPLING_RATE, dtype=np.float32)
        #     self.out.append(audio)
        #     self.buffer = b""
    
        # audio_data = np.concatenate(self.out)
        # audio_buffer = np.array([], dtype=np.float32)
        # audio_buffer = np.append(audio_buffer, audio_data)

        # self.audio_file.writeframes(audio_buffer.tobytes())


        transcribed_text = self.transcribe(audio_chunk, self.context)
        print(transcribed_text)
        #print("transcribed_text: " + transcribed_text)
        # confirmed_text = self.local_agreement.confirm_tokens(transcribed_text)
        # print(confirmed_text)
        # punctuation = r"[.!?]"  # Regular expression pattern for ., !, or ?
        # # Detect punctuation

        # if len(self.context) > self.max_context_length:
        #     self.confirmed_sentences = self.confirmed_sentences[-2:]
        #     self.context = " ".join(self.confirmed_sentences)
        #     #print("context truncated: " + self.context)

        # #print("check punctuation: ", re.search(punctuation,confirmed_text))
        # if re.search(punctuation,confirmed_text):
        #     split_sentence = re.split(f"({punctuation})", confirmed_text)

        #     # # Join the punctuation back to the respective parts of the sentence
        #     sentence = [split_sentence[i] + split_sentence[i+1] for i in range(0, len(split_sentence)-1, 2)]

        #     #print("sentence", sentence)
        #     self.confirmed_sentences.append(sentence[-1])
        #     self.context = " ".join(self.confirmed_sentences)
        #     #print("context added: " + self.context)
            
        #     self.local_agreement.clear_confirmed_text()
            
        #     # clear the audio buffer


        # return confirmed_text
    
    def start(self):
        print("Transcription process started...")
        # wait two seconds then call process audio
        while True:
            time.sleep(2)
            if self.buffer != b"": self.process_audio()