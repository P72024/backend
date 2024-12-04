
import logging
from math import ceil
import time
import scipy.io.wavfile as wavfile
import numpy as np
from faster_whisper import WhisperModel

from Util import unix_seconds_to_ms

class ASR:
    max_context_length = 200
    context: str = dict()
    unfinished_sentence = dict()
    previous_transcription = dict()

    def __init__ (self, model_size: str, device="auto", compute_type = "float16", max_context_length=200, num_workers = 1):
        self.whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type, num_workers=num_workers)
        self.max_context_length = max_context_length
        
    def transcribe(self, audio_chunk: np.float32, context: str) -> str:  
        transcribed_text = ""

        segments, info = self.whisper_model.transcribe(
            audio_chunk, 
            language='en',
            beam_size=12,
            initial_prompt=context,
        )
        
        for segment in segments:
            transcribed_text += " " + segment.text
            
        return transcribed_text
    
    
    def process_audio(self, audio_chunk: np.float32, room_id) -> tuple[str, str, str]:
        logging.info("[ASR] Processing audio chunk")
        transcribe_start_time = time.time()
        transcribed_text = self.transcribe(audio_chunk, self.context[room_id] if room_id in self.context else "")
        transcribe_time = unix_seconds_to_ms(time.time() - transcribe_start_time)
    
        update_context_start_time = time.time()
        self.update_context(transcribed_text, room_id)
        update_context_time = unix_seconds_to_ms(time.time() - update_context_start_time)
        logging.info(f"[ASR] Updated context: {self.context[room_id] if room_id in self.context else ''}")
    
        logging.info(f"[ASR] Finished processing audio chunk, transcribe time: {transcribe_time} ms, update context time: {update_context_time} ms")
        return (transcribed_text, transcribe_time, update_context_time)

    def confirm_text(self, transcribed_text: str, room_id) -> str:
        # Split the current and previous transcription into words
        new_words = transcribed_text.split()
        prev_words = self.previous_transcription[room_id].split() if room_id in self.previous_transcription else ""
        if len(prev_words) == 0:
            self.previous_transcription[room_id] = ' '.join(new_words)
            return ' '.join(new_words)
        # Initialize a list to store matching words
        matching_words = []

        # Compare words until two consecutive words differ
        differences = 0
        for i, word in enumerate(new_words):
            if i < len(prev_words) and word == prev_words[i]:
                matching_words.append(word)
            else:
                differences += 1
                if differences >= 2:
                    break
                matching_words.append(word)

        # Update previous transcription for future comparisons
        if room_id not in self.previous_transcription or self.previous_transcription[room_id] == "":
            self.previous_transcription[room_id] = transcribed_text
        else:
            self.previous_transcription[room_id] += " " + transcribed_text

        # Join and return the matching prefix as a single string
        return ' '.join(matching_words)    

    def update_context(self, new_text: str, room_id):
        """Update context with a sliding window to maintain continuity up to max_context_length words."""
        
        # Add the new transcription to context, treating it as a moving shingle
        if(room_id in self.context and len((self.context[room_id] + " " + new_text).split()) >= self.max_context_length):
            words_to_keep = ceil(self.max_context_length * 0.1)
            self.context[room_id] = ' '.join(self.context[room_id].split()[-words_to_keep:]) + " " + new_text
        else:
            if room_id not in self.context or self.context[room_id] == '':
                self.context[room_id] = new_text
            else:
                self.context[room_id] += " " + new_text
        

    def merge_sentences(self, unfinished: str, completed: str) -> str:
        """Merge unfinished and completed transcriptions by removing overlapping words and handling capitalization."""
        
        # Normalize both sentences to lowercase for comparison
        unfinished_lower = unfinished.strip().lower()
        completed_lower = completed.strip().lower()

        # Check if both sentences are the same when case is ignored
        if unfinished_lower == completed_lower:
            return completed.strip()  # Return the completed sentence only

        # Find the longest overlap from the end of 'unfinished' to the start of 'completed'
        overlap = self.longest_common_suffix_prefix(unfinished, completed)
        
        # Combine the two sentences without duplicating the overlapping part
        merged_sentence = unfinished + " " + completed[len(overlap):].strip()
        return merged_sentence.strip()

        
    def longest_common_suffix_prefix(self, str1: str, str2: str) -> str:
        """Find the longest common suffix of str1 that matches the prefix of str2."""
        
        min_len = min(len(str1), len(str2))
        for i in range(min_len, 0, -1):
            if str1[-i:] == str2[:i]:
                return str1[-i:]
        return ""


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