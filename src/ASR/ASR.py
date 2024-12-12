import logging
import os
import sys
import time

import numpy as np
import scipy.io.wavfile as wavfile
from faster_whisper import WhisperModel

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
        
        for segment in segments:
            transcribed_text += " " + segment.text
            
        return transcribed_text
    
    
    def process_audio(self, audio_chunk: np.float32, room_id) -> tuple[str, str, str]:
        logging.info("[ASR] Processing audio chunk")
        transcribe_start_time = time.time()
        transcribed_text = self.transcribe(audio_chunk, self.context[room_id] if room_id in self.context else "")
        if transcribed_text is not None:
                while transcribed_text.endswith('…'):
                    transcribed_text = transcribed_text[:-1]
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
