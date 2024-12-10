import logging
import re
from math import ceil

import numpy as np
from faster_whisper import WhisperModel


class ASR:
    max_context_length = 200
    context:str = ""
    prev_chunks = None
    beam_size = 5
    use_context = True
    num_workers = 1


    def __init__ (self,
                  model_size: str,
                  beam_size: int,
                  use_context: bool,
                  num_workers: int,
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
        
        for segment in segments:
            transcribed_text += " " + segment.text
            
        return transcribed_text
    
    
    def process_audio(self, audio_chunk: np.float32) -> str:
        logging.info("[ASR] Processing audio chunk")
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
