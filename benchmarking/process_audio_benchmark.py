import os
import pickle
import sys
import time
from io import BytesIO

import jiwer
import librosa
import numpy as np
from aiortc.contrib.media import MediaPlayer
from prettytable import PrettyTable
from pydub import AudioSegment

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from ASR.ASR import ASR

sr = 16000

async def process_audio_benchmark(chunks_pkl, meta_pkl, txt_filename, size):
    asr = ASR(model_size=size, device="auto", compute_type="float32")
    try:
        transcribed_text = ""
        actual_text = ""
        with open(meta_pkl, 'rb') as f:
            meta = pickle.load(f)
        with open(chunks_pkl, 'rb') as f:
            chunks = pickle.load(f)
        asr.save_metadata(meta[0])
        #Start timer
        times = []
        for chunk in chunks:
            start_total_time = time.time()
            newText = asr.receive_audio_chunk(chunk)
            while newText.endswith('…'):
                newText = newText[:-1]
            transcribed_text += " " +  newText
            times.append(time.time() - start_total_time)
        #End timer
        #total time elapsed
        total_time = sum(times)
        average_chunk_time = "{:.7f}".format(np.average(times))
        max_chunk_time = "{:.7f}".format(max(times))
        min_chunk_time = "{:.7f}".format(min(times))

        metric_table = PrettyTable(['Metric', 'value'])
        metric_table.add_row(['Max. chunk time', max_chunk_time])
        metric_table.add_row(['Min. chunk time', min_chunk_time])
        metric_table.add_row(['Avg. chunk time', average_chunk_time])
        metric_table.add_row(['Total Transcription time', total_time])
        print(metric_table)
        # average_time_per_chunk = total_time / rounds
        with open(txt_filename, "r") as f:
            actual_text = f.read()
        print(actual_text)
        transforms : jiwer.Compose = jiwer.Compose(
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
        print(transforms(transcribed_text))
        finaltext = " ".join(transforms(transcribed_text)[0])
        print(f"The actual text is:\n{actual_text}")
        print(f'\n\nThe Transcribed text was:\n{finaltext}')
        wer = jiwer.wer(actual_text, transcribed_text, truth_transform=transforms, hypothesis_transform=transforms)
        print(f"    WER: {wer} between {txt_filename} and test")
        print(f"    Total time using process_audio on test: {total_time} seconds")
        # print(f"    Average time pr. chunk: {average_time_per_chunk} seconds, chunk size: {chunk_size}")
    except Exception as e:
        print(f"Error creating metrics for file: {txt_filename}: {e}")
    
