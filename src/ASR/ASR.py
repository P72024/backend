
import logging
import re
import wave
from asyncio import sleep
from io import BytesIO
from math import ceil
from typing import List

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from pydub import AudioSegment

from .LocalAgreement import LocalAgreement


class ASR:
    max_context_length = 200
    metadata: BytesIO = BytesIO()
    audio_buffer: List[BytesIO] = []
    silence_threshold: np.float64 = np.float64(0.042)
    local_agreement = LocalAgreement()
    context:str = ""
    conf_limit = 0.87
    confirmed_sentences: List[str] = []
    min_chunk_size = 3
    min_chunk_size_default = 3
    unfinished_sentence = None
    min_silence_duration_ms = 300  # Minimum duration of silence to consider it as non-speech
    previous_buffer = BytesIO()
    previous_transcription = ""
    idx_zero_fails = 0
    idx_zero_fail_limit = 3

    def __init__ (self, model_size: str, device="auto", compute_type = "float16", max_context_length=200, chunk_limit=48):
        self.whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.min_chunk_size = chunk_limit
        self.min_chunk_size_default = chunk_limit
        self.max_context_length = max_context_length
        
    def transcribe(self, audio_buffer: BytesIO, context: str) -> str:
        # print(audio_buffer.getbuffer().nbytes)

        # audio_buffer.seek(0)
        # # a lil debug tang
        # with open("temp.webm", 'wb') as f:
        #     f.write(audio_buffer.getvalue())

        audio_buffer.seek(0)
        
        segments, _ = self.whisper_model.transcribe(audio_buffer, 
                                                        language='en',
                                                        beam_size=12,
                                                        initial_prompt=context,
                                                        condition_on_previous_text=True,
                                                        vad_filter=True,
                                                        vad_parameters={"threshold": 0.6, "min_silence_duration_ms": 300}, word_timestamps=True)
        
        transcribed_words = []
        for segment in segments:
            for word in segment.words:
            # Append each word with its text and timestamp
                transcribed_words.append((word.word, word.start, word.end, word.probability))

        return transcribed_words
    
    def save_metadata(self, metadata):
        # print("saving metadata")
        # print(type(metadata))
        self.metadata.write(metadata)
        self.metadata.seek(0)  # Reset buffer's position to the beginning

    def receive_audio_chunk(self, audio_chunk):
        # print("recieving audio chunk")
        transcription = ""
        self.audio_buffer.append(BytesIO(audio_chunk))

        # print("audio buffer length: ", len(self.audio_buffer))
        # print("min chunk size: ", self.min_chunk_size)
        if len(self.audio_buffer) > self.min_chunk_size:
            transcription = self.process_audio()

        return transcription

    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess the text by:
        - Removing punctuation (.,-?!)
        - Splitting it into words
        """
        # Remove punctuation using regex
        text = text.lower()
        clean_text = re.sub(r"[.,!?-]", "", text)
        # Split into words and return as a list
        return clean_text.strip().split()

    def get_overlap(self, transcription: str, previous_transcription: str) -> int:
        """
        Calculate the exact overlap between the end of `previous_transcription`
        and the beginning of `transcription`, ignoring punctuation.
        """
        # Preprocess both texts
        words1 = self.preprocess_text(previous_transcription)
        words2 = self.preprocess_text(transcription)

        # Find the maximum exact overlap
        overlap = 0
        max_overlap = min(len(words1), len(words2))  # Limit by the shorter list

        for j in range(1, max_overlap + 1):
            if words1[-j:] == words2[:j]:  # Compare the last `j` words of `words1` with the first `j` words of `words2`
                overlap = j

        return overlap
    
    def process_audio(self) -> str:
        combined_bytes = self.metadata.getvalue() + b''.join(bio.getvalue() for bio in self.audio_buffer)
        combined_bytes_io = BytesIO(combined_bytes)
        (sampleRate, sampleWidth, numChannels) = self.analyze(combined_bytes_io)
        # print(f"Sample Rate: {sampleRate}, Sample Width: {sampleWidth}, Number of Channels: {numChannels}")

        if(self.is_silent(combined_bytes_io)):
            logging.info("Silence detected!")
            self.audio_buffer.clear()
            return ""
        combined_bytes_io.seek(0)  # Reset for reading
        transcribed_words = self.transcribe(combined_bytes_io, self.context)
        start_audio_time_split : int = 0
        end_audio_time_split : int = 0
        transcribed_text = ""
        total_prob = 0
        for idx, (text, start, end, prob) in enumerate(transcribed_words):
            total_prob += prob
            transcribed_text += " " + text.strip()
            transcribed_text.strip()
        # print(f"total_prob: {total_prob}")
        # print(f"transcribed_words: {len(transcribed_words)}")
        if total_prob != 0 and len(transcribed_words) != 0:
            if total_prob / len(transcribed_words) > self.conf_limit:
                self.update_context(transcribed_text)
                overlap = self.get_overlap(transcribed_text, self.previous_transcription)
                confidence = total_prob/len(transcribed_words)
                # print(f"Overlaps for previous: {overlap}")
                yield (transcribed_text, overlap, confidence)
                self.previous_transcription = transcribed_text
                # print(f"[{total_prob/len(transcribed_words)}]: {transcribed_text}")
                self.min_chunk_size = self.min_chunk_size_default
                self.audio_buffer = self.audio_buffer[-5:]
            else:
                self.min_chunk_size += self.min_chunk_size_default


            # print(f"(INDEX: {idx}, TEXT: {text.strip()}, PROB: {prob}, STARTTIME: {start}, ENDTIME: {end})")
            # if prob > 0.50:
            #     transcribed_text += " " + text.strip()
            #     transcribed_text.strip()
            #     if idx % 3 == 0:
            #         end_audio_time_split = end
            #     yield text
            #     if idx == len(transcribed_words) -1:
            #         self.audio_buffer.clear()
            #         break
            #
            # elif prob < 0.50:
            #     # if idx == 0:
            #     #     self.idx_zero_fails += 1
            #     #     if self.idx_zero_fails >= self.idx_zero_fail_limit:
            #     #         end_audio_time_split = end
            #     #         yield text
            #     #         transcribed_text += " " + text.strip()
            #     #         transcribed_text.strip()
            #     #         self.remove_time_frame(start_audio_time_split, end_audio_time_split, sampleRate, sampleWidth, numChannels)
            #     #         self.idx_zero_fails = 0
            #     #         break
            #     # elif end_audio_time_split != 0:
            #     #     # A word was found that was not confident. Remove the emitted audio segments that has been yielded from the self.audio_buffer.
            #     self.remove_time_frame(start_audio_time_split, end_audio_time_split, sampleRate, sampleWidth, numChannels)
            #     # print(f"not good enough on index: {idx} with confidence: {prob}")
            #     break

                



        

        # print(f"context is : {self.context}")
    
        # Clear audio buffer after processing to avoid duplicating input
        # self.audio_buffer.clear()
        # return transcribed_text

    def remove_time_frame(self, start_time: int, end_time: int, sampleRate, sampleWidth, numChannels):
        print(f"Now removing time frame from Start: {start_time} till End: {end_time}")
        # Parameters to calculate time per chunk (adjust according to your format)
        sample_rate = sampleRate  # e.g., 16 kHz
        bytes_per_sample = sampleWidth  # e.g., 16-bit audio = 2 bytes per sample
        channels = numChannels          # e.g., mono audio
        bytes_per_second = sample_rate * bytes_per_sample * channels

        # Time to byte range
        start_byte = int(start_time * bytes_per_second)
        end_byte = int((end_time) * bytes_per_second)

        current_byte_position = 0
        new_audio_buffer = []

        for bio in self.audio_buffer:
            bio_data = bio.getvalue()
            bio_length = len(bio_data)

            # Check if this chunk overlaps with the time range
            if current_byte_position + bio_length <= start_byte:
                # This chunk is completely before the range, keep it as is
                new_audio_buffer.append(bio)
            elif current_byte_position >= end_byte:
                # This chunk is completely after the range, keep it as is
                new_audio_buffer.append(bio)
            else:
                # This chunk overlaps with the range
                if current_byte_position < start_byte:
                    # Keep the part before the start of the range
                    overlap_start = start_byte - current_byte_position
                    new_audio_buffer.append(BytesIO(bio_data[:overlap_start]))
                if current_byte_position + bio_length > end_byte:
                    # Keep the part after the end of the range
                    overlap_end = end_byte - current_byte_position
                    new_audio_buffer.append(BytesIO(bio_data[overlap_end:]))

            # Update the current byte position
            current_byte_position += bio_length

        # Replace the audio buffer with the modified one
        self.audio_buffer = new_audio_buffer

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
        
        # Debug statement to check current context
        # print(f"Updated Context (Shingle): {self.context}")
    
    def analyze(self, audio_bytes: BytesIO) -> bool:
        """Check if the audio chunk is silent based on RMS energy."""
        # Reset buffer and read audio data

        audio : AudioSegment = AudioSegment.from_file(audio_bytes, format='webm', codec='opus')
        samplewidth = audio.sample_width
        numchannels = audio.channels
        samplerate = audio.frame_rate
        audio_bytes.seek(0)
        return (samplerate, samplewidth, numchannels)

    def is_silent(self, audio_bytes: BytesIO) -> bool:
        """Check if the audio chunk is silent based on RMS energy."""
        # Reset buffer and read audio data

        audio : AudioSegment = AudioSegment.from_file(audio_bytes, format='webm', codec='opus')
        ogg_audio = BytesIO()
        audio.export(ogg_audio, format='ogg')
        audio_bytes = ogg_audio
        audio_bytes.seek(0)
        audio_data, sample_rate = sf.read(audio_bytes)

        # Calculate RMS energy
        rms_energy = np.sqrt(np.mean(np.square(audio_data)))
        
        # print(f"[RMS ENERGY] {rms_energy}")
        # logging.info(f"[RMS ENERGY] {rms_energy}")
        
        # Check if energy is below the silence threshold
        return rms_energy < self.silence_threshold
