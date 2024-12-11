import logging
<<<<<<< HEAD
import re
from math import ceil
=======
import os
import sys
import time
>>>>>>> origin/benchmarking-backend

import numpy as np
import scipy.io.wavfile as wavfile
from faster_whisper import WhisperModel

<<<<<<< HEAD

class ASR:
    max_context_length = 200
    context:str = ""
    prev_chunks = None
    beam_size = 5
    use_context = True
    num_workers = 1


    def __init__ (self,
                  model_size: str = 'tiny.en',
                  beam_size: int = 5,
                  use_context: bool = True,
                  num_workers: int = 1,
                  device="auto",
                  compute_type = "int8_float32",
                  max_context_length=200):
        self.max_context_length = max_context_length
        self.beam_size = beam_size
        self.use_context = use_context
        self.num_workers = num_workers

        self.whisper_model = WhisperModel(model_size,
                                          device=device,
                                          compute_type=compute_type,
                                          num_workers=self.num_workers)
        
    def transcribe(self, audio_chunk: np.float32, context: str) -> str:  
        transcribed_text = ""

        if self.use_context:
            segments, _ = self.whisper_model.transcribe(
                audio_chunk, 
                append_punctuations=".,?!",
                language='en',
                beam_size=self.beam_size,
                initial_prompt=context,
            )
        else:
            segments, _ = self.whisper_model.transcribe(
                audio_chunk, 
                append_punctuations=".,?!",
                language='en',
                beam_size=self.beam_size,
            )
=======
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from Util import unix_seconds_to_ms


class ASR:
    def __init__ (self,
                  model_size: str = 'tiny.en',
                  beam_size: int = 5,
                  num_workers: int = 4,
                  device="auto",
                  compute_type = "int8_float32",
                  max_context_length=200,
                  ):
        self.whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type, num_workers=num_workers)
        self.max_context_length = max_context_length
        self.beam_size = beam_size

    def transcribe(self, audio_chunk: np.float32, context: str) -> str:  
        transcribed_text = ""

        segments, _ = self.whisper_model.transcribe(
            audio_chunk, 
            language='en',
            beam_size=self.beam_size,
            initial_prompt=context,
        )
>>>>>>> origin/benchmarking-backend
        
        for segment in segments:
            transcribed_text += " " + segment.text
            
        return transcribed_text
    
    
    def process_audio(self, audio_chunk: np.float32, room_id) -> tuple[str, str, str]:
        logging.info("[ASR] Processing audio chunk")
<<<<<<< HEAD
        return self.transcribe(audio_chunk, self.context)
        if self.prev_chunks is not None:
            audio_chunk = np.concatenate((self.prev_chunks, audio_chunk), axis=0)
            self.prev_chunks = None

        transcribed_text = self.transcribe(audio_chunk, self.context)
        if ("..." in transcribed_text or transcribed_text.endswith('-') or transcribed_text.endswith('- ')):
            # print(f"Something is unfinished")
            self.prev_chunks = audio_chunk
            return ""
        
        # If there was an unfinished sentence, merge it with the new transcription
        transcribed_text = transcribed_text.lstrip()
        pattern = r'[^a-zA-Z0-9.,\-/?! ]'
        transcribed_text = re.sub(pattern, '', transcribed_text)

        # print(f"[TRANSCRIPTION] {transcribed_text}")

        if self.use_context:
            self.update_context(transcribed_text)
        logging.info(f"[ASR] Updated context: {self.context}")
    
        return transcribed_text

    def update_context(self, new_text: str):
        """Update context with a sliding window to maintain continuity up to max_context_length words."""
        
        # Add the new transcription to context, treating it as a moving shingle
        if(len((self.context + " " + new_text).split()) >= self.max_context_length):
            words_to_keep = ceil(self.max_context_length * 0.1)
            self.context = ' '.join(self.context.split()[-words_to_keep:]) + " " + new_text
        else:
            if self.context == '':
                self.context = new_text
            else:
                self.context += " " + new_text
=======
        transcribe_start_time = time.time()
        transcribed_text = self.transcribe(audio_chunk, self.context[room_id] if room_id in self.context else "")
        transcribe_time = unix_seconds_to_ms(time.time() - transcribe_start_time)
        return (transcribed_text, transcribe_time, 0)
    
    def save_audio_to_file(self, audio_array, sample_rate, filename):
        # Ensure the array is float32
        audio_array = audio_array.astype(np.float32)
        
        # Normalize to prevent clipping (optional)
        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            audio_array = audio_array / max_val
        
        # Scale to int16 range for WAV file writing
        scaled_audio = np.int16(audio_array * 32767)
        
        # Write to file
        wavfile.write(filename, sample_rate, scaled_audio)
>>>>>>> origin/benchmarking-backend
