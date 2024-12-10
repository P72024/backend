import logging
import re
from math import ceil

import numpy as np
from faster_whisper import WhisperModel


class ASR_tweaked:
    max_context_length = 200
    context:str = ""
    conf_limit = 0.4
    beam_size = 5
    use_context = True
    prev_chunks = None

    def __init__ (self, model_size: str,
                  beam_size: int,
                  use_context: bool,
                  confidence_limit: float,
                  num_workers: int,
                  device="auto",
                  compute_type = "int8_float32",
                  max_context_length=200,
                  ):
        self.max_context_length = max_context_length
        self.conf_limt = confidence_limit
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
    
    
    def process_audio(self, audio_chunk: np.float32):
        if self.prev_chunks is not None:
            audio_chunk = np.concatenate((self.prev_chunks, audio_chunk), axis=0)
            self.prev_chunks = None

        transcribed_words = self.transcribe(audio_chunk, self.context)

        transcribed_text = ""
        total_prob = 0
        for idx, (text, start, end, prob) in enumerate(transcribed_words):
            total_prob += prob
            transcribed_text += " " + text.strip()
            transcribed_text.strip()

        if total_prob != 0 and len(transcribed_words) != 0:
            logging.info(total_prob / len(transcribed_words))
            if total_prob / len(transcribed_words) > self.conf_limit and transcribed_words[-1][3] > self.conf_limit:
                self.update_context(transcribed_text)
                # logging.info("Text Transcribed!")
                print(f"[TRANSCRIPTION] {transcribed_text}")
                return transcribed_text
            else:
                self.prev_chunks = audio_chunk
                return ""
                

        return ""

    def update_context(self, new_text: str):
        """Update context with a sliding window to maintain continuity up to max_context_length words."""
        
        text = re.sub(r'[^a-zA-Z0-9,.?!\s]', '', new_text)
        text = re.sub(r'[,.?!]+', lambda match: match.group(0)[0], text)
    
        # Normalize spaces by collapsing multiple spaces into a single one
        text = re.sub(r'\s+', ' ', text).strip()
        # Add the new transcription to context, treating it as a moving shingle
        if(len((self.context + " " + new_text).split()) >= self.max_context_length):
            words_to_keep = ceil(self.max_context_length * 0.1)
            self.context = ' '.join(self.context.split()[-words_to_keep:]) + " " + new_text
        else:
            if self.context == '':
                self.context = new_text
            else:
                self.context += " " + new_text
