import time

import librosa
import numpy as np
import jiwer
from src.ASR.ASR import ASR
async def process_audio_benchmark(wav_filename, txt_filename, asr):
    try:
        print(f"Processing {wav_filename} and {txt_filename}")
        transcribed_text = ""
        actual_text = ""
        audio, _ = librosa.load(wav_filename, sr=16000, dtype=np.float32)
        # Chunk the audio into 2 second chunks
        chunk_size = 16000 * 2

        #Start timer
        start_total_time = time.time()
        rounds = 0.0
        for i in range(0, len(audio), chunk_size):
            transcribed_text += asr.process_audio(audio[i:i + chunk_size])
            rounds += 1
        #End timer
        end_total_time = time.time()
        #total time elapsed
        total_time = end_total_time - start_total_time
        average_time_per_chunk = total_time / rounds
        with open(txt_filename, "r") as f:
            actual_text = f.read()
        transforms = jiwer.Compose(
            [
                jiwer.ExpandCommonEnglishContractions(),
                jiwer.RemoveEmptyStrings(),
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.RemovePunctuation(),
                jiwer.ReduceToListOfListOfWords(),
            ]
        )
        wer = jiwer.wer(actual_text, transcribed_text, truth_transform=transforms, hypothesis_transform=transforms)
        print(f"    WER: {wer} between {txt_filename} and {wav_filename}")
        print(f"    Total time using process_audio on {wav_filename}: {total_time} seconds")
        print(f"    Average time pr. chunk: {average_time_per_chunk} seconds, chunk size: {chunk_size}")
    except Exception as e:
        print(f"Error processing {wav_filename} and {txt_filename}: {e}")