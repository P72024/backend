import re
import wave
from asyncio import sleep
from io import BytesIO
from typing import List

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from pydub import AudioSegment

from ASR.LocalAgreement import LocalAgreement


class ASR:
    max_context_length = 200
    metadata: BytesIO = BytesIO()
    audio_buffer: List[BytesIO] = []
    silence_threshold: np.float64 = np.float64(0.040)
    local_agreement = LocalAgreement()
    context:str = ""
    confirmed_sentences: List[str] = []
    min_chunk_size = 6
    unfinished_sentence = None
    min_silence_duration_ms = 300  # Minimum duration of silence to consider it as non-speech
    previous_buffer = BytesIO()
    previous_transcription = ""

    def __init__ (self, model_size: str, device="auto", compute_type = "float16", max_context_length=200):
        self.whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.max_context_length = max_context_length
        
    def transcribe(self, audio_buffer: BytesIO, context: str):
        # print(audio_buffer.getbuffer().nbytes)

        audio_buffer.seek(0)
        # a lil debug tang
        with open("temp.webm", 'wb') as f:
            f.write(audio_buffer.getvalue())

        audio_buffer.seek(0)
        
        transcribed_text = ""
        segments, info = self.whisper_model.transcribe(audio_buffer, language='en', beam_size=12, initial_prompt=context, condition_on_previous_text=True)
        
        for segment in segments:
            transcribed_text += " " + segment.text
            
        return transcribed_text
    
    def save_metadata(self, metadata):
        # print("saving metadata")
        self.metadata.write(metadata)
        self.metadata.seek(0)  # Reset buffer's position to the beginning

    def receive_audio_chunk(self, audio_chunk):
        # print("recieving audio chunk")
        self.audio_buffer.append(BytesIO(audio_chunk))

        # print("audio buffer length: ", len(self.audio_buffer))
        # print("min chunk size: ", self.min_chunk_size)
        if len(self.audio_buffer) > self.min_chunk_size:
            self.process_audio()
    
    def process_audio(self) -> str:
        combined_bytes = self.metadata.getvalue() + b''.join(bio.getvalue() for bio in self.audio_buffer)
        combined_bytes_io = BytesIO(combined_bytes)
        if(self.is_silent(combined_bytes_io)):
            self.audio_buffer.clear()
            return ""
        combined_bytes_io.seek(0)  # Reset for reading
        transcribed_text = self.transcribe(combined_bytes_io, self.context)
        if "..." in transcribed_text or '- ' in transcribed_text:
            self.unfinished_sentence = transcribed_text.replace("...", "")  # Remove trailing ellipsis
            self.unfinished_sentence = transcribed_text.replace("- ", "")
            # print('missing end of sentence')
            return ""
        
        # If there was an unfinished sentence, merge it with the new transcription
        if self.unfinished_sentence:
            # print(f"Merging sentences:\n1. '{transcribed_text}'\n2. '{self.unfinished_sentence}'")
            # transcribed_text = self.merge_sentences(self.unfinished_sentence, transcribed_text)
            transcribed_text = transcribed_text
            self.unfinished_sentence = None  # Reset unfinished sentence
        transcribed_text = transcribed_text.lstrip()
        confirmed_text = self.confirm_text(transcribed_text)
        # print(f"[CONFIRMED TRANSCRIPTION] {confirmed_text}")
        print(f"[TRANSCRIPTION] {transcribed_text}")
        self.update_context(transcribed_text)
    
        # Clear audio buffer after processing to avoid duplicating input
        self.audio_buffer.clear() 
        return transcribed_text


    def confirm_text(self, transcribed_text: str) -> str:
        # Split the current and previous transcription into words
        new_words = transcribed_text.split()
        prev_words = self.previous_transcription.split()

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
        self.previous_transcription = transcribed_text

        # Join and return the matching prefix as a single string
        return ' '.join(matching_words)    

    def update_context(self, new_text: str):
        """Update context with a sliding window to maintain continuity up to max_context_length words."""
        
        # Add the new transcription to context, treating it as a moving shingle
        if(len((self.context + " " + new_text).split()) > self.max_context_length):
            words_to_keep = max(int(self.max_context_length * 0.2), 1)
            self.context = self.context[-words_to_keep:] + new_text
        else:
            self.context += " " + new_text
        
        # Debug statement to check current context
        # print(f"Updated Context (Shingle): {self.context}")
    
    
    def is_silent(self, audio_bytes: BytesIO) -> bool:
        """Check if the audio chunk is silent based on RMS energy.
        
        Args:
            audio_bytes (BytesIO): Audio data in WebM format
        
        Returns:
            bool: True if the audio chunk is considered silent
        """
        # Reset buffer position
        audio_bytes.seek(0)
        
        # Load WebM audio using pydub
        audio = AudioSegment.from_file(audio_bytes, format="webm")
        
        # Convert audio to raw numpy array
        # Get audio data as an array of samples
        samples = np.array(audio.get_array_of_samples())
        
        # If audio is stereo, convert to mono by averaging channels
        if audio.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)
        
        # Calculate RMS energy
        rms_energy = np.sqrt(np.mean(np.square(samples)))
        
        # Normalize RMS energy based on audio parameters
        # pydub uses different scaling than soundfile, so we adjust the threshold
        normalized_rms = rms_energy / (1 << (audio.sample_width * 8 - 1))
        
        # print(f"[NORMALIZED RMS ENERGY] {normalized_rms}")
        
        # Check if energy is below the silence threshold
        return normalized_rms < self.silence_threshold
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

