import os
import pickle
import sys
import time
from io import BytesIO
import csv

import jiwer
import numpy as np
from prettytable import PrettyTable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from ASR.ASR import ASR



async def process_audio_benchmark(chunks_pkl, txt_filename, model):
    asr = ASR(model_size=model, device="auto", compute_type="float32")
    transcribed_text = ""
    actual_text = ""
    with open(chunks_pkl, 'rb') as f:
        chunks = pickle.load(f)
    print(f"Number of chunks: {len(chunks)}")
    # sf.write("benchmark.wav", chunks, samplerate=sr)
    #Start timer
    times = []
    for chunk in chunks:
        start_total_time = time.time()
        newText = asr.process_audio(chunk)
        if newText is not None:
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
    # print(f"    WER: {wer * 100:.1f}% between {txt_filename} and test")
    # print(f"    Total time using process_audio on test: {total_time} seconds")
    # metric_table = PrettyTable(['Metric', 'value'])
    # metric_table.add_row(['Max. chunk time', max_chunk_time])
    # metric_table.add_row(['Min. chunk time', min_chunk_time])
    # metric_table.add_row(['Avg. chunk time', average_chunk_time])
    # metric_table.add_row(['Word Error Rate (WER)', f"{wer * 100:.1f}% "])
    # metric_table.add_row(['Total Transcription time', total_time])
    # print(metric_table)
    # print(f"    Average time pr. chunk: {average_time_per_chunk} seconds, chunk size: {chunk_size}")
    results = {
        "Max. chunk time": max_chunk_time,
        "Min. chunk time": min_chunk_time,
        "Avg. chunk time": average_chunk_time,
        "Word Error Rate (WER)": f"{wer * 100:.1f}%",
        "Total Transcription time": total_time
    }

    return results
