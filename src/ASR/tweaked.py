import logging
import os
import re
import sys
import time
from math import ceil

import numpy as np
from faster_whisper import WhisperModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from Util import unix_seconds_to_ms


class ASR_tweaked:
    context:dict = {}
    prev_chunks : dict = {}

    def __init__ (self, model_size: str = 'tiny.en',
                  beam_size: int = 3,
                  use_context: bool = True,
                  confidence_limit: float = 0.8,
                  num_workers: int = 4,
                  device="auto",
                  compute_type = "int8_float32",
                  max_context_length=200,
                  ):
        self.max_context_length = max_context_length
        self.conf_limit = confidence_limit
        self.beam_size = beam_size
        self.num_workers = num_workers
        self.use_context = use_context

        self.whisper_model = WhisperModel(model_size,
                                          device=device,
                                          compute_type=compute_type,
                                          num_workers=num_workers
                                          )
        
    def transcribe(self, audio_chunk: np.float32, context: str) -> str:  

        if self.use_context:
            segments, _ = self.whisper_model.transcribe(audio_chunk, 
                language='en',
                beam_size=self.beam_size,
                append_punctuations=".,?!",
                initial_prompt=context,
                word_timestamps=True,
            )
        else:
            segments, _ = self.whisper_model.transcribe(audio_chunk, 
                language='en',
                beam_size=self.beam_size,
                append_punctuations=".,?!",
                word_timestamps=True,
            )
        
        transcribed_words = []
        for segment in segments:
            for word in segment.words:
            # Append each word with its text and timestamp
                transcribed_words.append((word.word, word.start, word.end, word.probability))

        return transcribed_words
    
    
    def process_audio(self, audio_chunk: np.float32, room_uuid : str):
        if room_uuid in self.prev_chunks and self.prev_chunks[room_uuid] is not None:
            audio_chunk = np.concatenate((self.prev_chunks[room_uuid], audio_chunk), axis=0)
            self.prev_chunks[room_uuid] = None

        if room_uuid not in self.context:
            self.context[room_uuid] = ''

        transcribe_start_time = time.time()
        transcribed_words = self.transcribe(audio_chunk, self.context[room_uuid])

        transcribed_text = ""
        total_prob = 0
        for idx, (text, start, end, prob) in enumerate(transcribed_words):
            total_prob += prob
            transcribed_text += " " + text.strip()
            transcribed_text.strip()
            
        if transcribed_text is not None:
                while transcribed_text.endswith('…'):
                    transcribed_text = transcribed_text[:-1]

        transcribe_time = unix_seconds_to_ms(time.time() - transcribe_start_time)
        
        if total_prob != 0 and len(transcribed_words) != 0:
            logging.info(total_prob / len(transcribed_words))
            if total_prob / len(transcribed_words) > self.conf_limit and transcribed_words[-1][3] > self.conf_limit:
                update_context_start_time = time.time()
                self.update_context(transcribed_text, room_uuid)
                update_context_time = unix_seconds_to_ms(time.time() - update_context_start_time)
                # logging.info("Text Transcribed!")
                # print(f"[TRANSCRIPTION] {transcribed_text}")
                return (transcribed_text, transcribe_time, update_context_time)
            else:
                self.prev_chunks[room_uuid] = audio_chunk
                return ("", transcribe_time, 0)
                

        return ("", transcribe_time, 0)

    def update_context(self, new_text: str, room_uuid : str):
        """Update context with a sliding window to maintain continuity up to max_context_length words."""
        
        text = re.sub(r'[^a-zA-Z0-9,.?!\s]', '', new_text)
        text = re.sub(r'[,.?!]+', lambda match: match.group(0)[0], text)
    
        # Normalize spaces by collapsing multiple spaces into a single one
        text = re.sub(r'\s+', ' ', text).strip()
        # Add the new transcription to context, treating it as a moving shingle

        if(len((self.context[room_uuid] + " " + new_text).split()) >= self.max_context_length):
            words_to_keep = ceil(self.max_context_length * 0.1)
            self.context[room_uuid] = ' '.join(self.context[room_uuid].split()[-words_to_keep:]) + " " + new_text
        else:
            if self.context[room_uuid] == '':
                self.context[room_uuid] = new_text
            else:
                self.context[room_uuid] += " " + new_text
