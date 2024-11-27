
import logging
import re
from math import ceil
from typing import List
import scipy.io.wavfile as wavfile
import numpy as np
from faster_whisper import WhisperModel

class ASR:
    max_context_length = 200
    audio_buffer: List[np.float32] = []
    context:str = ""
    unfinished_sentence = None
    previous_transcription = ""

    def __init__ (self, model_size: str, device="auto", compute_type = "float16", max_context_length=200):
        self.whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.max_context_length = max_context_length
        
    def transcribe(self, audio_buffer: np.float32, context: str) -> str:  
        transcribed_text = ""

        segments, info = self.whisper_model.transcribe(
            audio_buffer, 
            language='en',
            beam_size=12,
            initial_prompt=context
        )
        
        for segment in segments:
            transcribed_text += " " + segment.text
            
        return transcribed_text
    
    
    def process_audio(self, audio_chunk: np.float32) -> str:
        logging.info("[ASR] Processing audio chunk")
        transcribed_text = self.transcribe(audio_chunk, self.context)
        if "..." in transcribed_text or '- ' in transcribed_text:
            print('something is unfinished')
            self.unfinished_sentence = transcribed_text.replace("...", "")  # Remove trailing ellipsis
            self.unfinished_sentence = transcribed_text.replace("- ", "")
            return ""
        
        # If there was an unfinished sentence, merge it with the new transcription
        if self.unfinished_sentence:
            transcribed_text = transcribed_text
            self.unfinished_sentence = None  # Reset unfinished sentence
        transcribed_text = transcribed_text.lstrip()
        pattern = r'[^a-zA-Z0-9.,\-/?! ]'
        transcribed_text = re.sub(pattern, '', transcribed_text)
        confirmed_text = self.confirm_text(transcribed_text)
        print(f"[CONFIRMED TRANSCRIPTION] {confirmed_text}")

        self.update_context(transcribed_text)
        logging.info(f"[ASR] Updated context: {self.context}")
    
        # Clear audio buffer after processing to avoid duplicating input
        logging.info(f"[ASR] Clearing audio buffer {len(self.audio_buffer)}")
        self.audio_buffer.clear()
        logging.info(f"[ASR] Audio buffer cleared {len(self.audio_buffer)}")
        return transcribed_text

    def confirm_text(self, transcribed_text: str) -> str:
        # Split the current and previous transcription into words
        new_words = transcribed_text.split()
        prev_words = self.previous_transcription.split()
        if len(prev_words) == 0:
            self.previous_transcription = ' '.join(new_words)
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
        if self.previous_transcription == "":
            self.previous_transcription = transcribed_text
        else:
            self.previous_transcription += " " + transcribed_text

        # Join and return the matching prefix as a single string
        return ' '.join(matching_words)    

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