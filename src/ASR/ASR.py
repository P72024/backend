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
    silence_threshold: np.float64 = np.float64(0.042)
    local_agreement = LocalAgreement()
    context:str = ""
    confirmed_sentences: List[str] = []
    min_chunk_size = 3
    unfinished_sentence = None
    min_silence_duration_ms = 300  # Minimum duration of silence to consider it as non-speech
    previous_buffer = BytesIO()
    previous_transcription = ""

    def __init__ (self, model_size: str, device="auto", compute_type = "float16", max_context_length=200):
        self.whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.max_context_length = max_context_length
        
    def transcribe(self, audio_buffer: BytesIO, context: str):
        transcribed_text = ""
        segments, info = self.whisper_model.transcribe(audio_buffer, language='en', beam_size=5, vad_filter=True, vad_parameters={'threshold': 0.6, 'min_silence_duration_ms': 300})
        
        for segment in segments:
            transcribed_text += " " + segment.text
            
        return transcribed_text
    
    def save_metadata(self, metadata):
        # print("saving metadata")
        self.metadata.write(metadata)
        self.metadata.seek(0)  # Reset buffer's position to the beginning

    def receive_audio_chunk(self, audio_chunk):
        transcribed_text = ''
        # print("recieving audio chunk")
        self.audio_buffer.append(BytesIO(audio_chunk))

        # print("audio buffer length: ", len(self.audio_buffer))
        # print("min chunk size: ", self.min_chunk_size)
        if len(self.audio_buffer) > self.min_chunk_size:
            transcribed_text = self.process_audio()

        return transcribed_text
    
    def process_audio(self) -> str:
        combined_bytes = self.metadata.getvalue() + b''.join(bio.getvalue() for bio in self.audio_buffer)
        combined_bytes_io = BytesIO(combined_bytes)
        
        transcribed_text = self.transcribe(combined_bytes_io, self.context)
        print(transcribed_text)
    
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
        """Check if the audio chunk is silent based on RMS energy."""
        # Reset buffer and read audio data

        audio = AudioSegment.from_file(audio_bytes, format='webm', codec='opus')
        ogg_audio = BytesIO()
        audio.export(ogg_audio, format='ogg')
        audio_bytes = ogg_audio
        audio_bytes.seek(0)
        audio_data, sample_rate = sf.read(audio_bytes)

        # Calculate RMS energy
        rms_energy = np.sqrt(np.mean(np.square(audio_data)))
        
        print(f"[RMS ENERGY] {rms_energy}")
        
        # Check if energy is below the silence threshold
        return rms_energy < self.silence_threshold

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

