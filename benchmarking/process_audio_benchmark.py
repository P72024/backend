import csv
import os
import pickle
import sys
import time
from io import BytesIO

import GPUtil
import jiwer
import numpy as np
import psutil
from prettytable import PrettyTable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from ASR.ASR import ASR
from ASR.tweaked import ASR_tweaked


def measure_usage():
    # Get RAM usage in GB
    ram_used_gb = psutil.virtual_memory().used / (1024 ** 3)  # Convert bytes to GB

    # Get GPU usage: VRAM used (in MB) and GPU MHz
    gpus = GPUtil.getGPUs()
    gpu_vram_used_mb = 0  # Default to 0 if no GPU is found
    gpu_mhz = 0           # Default to 0 if no GPU is found

    if gpus:
        # Assume single GPU for simplicity (you can loop through `gpus` if needed)
        gpu = gpus[0]
        gpu_vram_used_mb = gpu.memoryUsed  # VRAM used in MB
        gpu_mhz = gpu.clock  # GPU clock in MHz

    return ram_used_gb, gpu_vram_used_mb, gpu_mhz

async def process_audio_benchmark(chunks_pkl, txt_filename, params : dict, use_gpu : bool):
# Define required keys and their expected types
    required_keys_with_types = {
        "model_type": str,
        "beam_size": int,
        "use_context": bool,
        "confidence_limit": float,
        "confidence_based": bool,
        "num_workers": int,
        "compute_type": str,
    }

    # Validate keys and their types
    missing_keys = required_keys_with_types.keys() - params.keys()
    assert not missing_keys, f"Missing keys: {missing_keys}"

    for key, expected_type in required_keys_with_types.items():
        assert isinstance(params[key], expected_type), (
            f"Key '{key}' must be of type {expected_type.__name__}, "
            f"but got {type(params[key]).__name__}"
        )

    if params["confidence_based"] == True:
        asr = ASR_tweaked(model_size=params["model_type"],
                          beam_size=params["beam_size"],
                          use_context=params["use_context"],
                          confidence_limit=params["confidence_limit"],
                          num_workers=params["num_workers"],
                          device="cuda" if use_gpu else "auto",
                          compute_type=params["compute_type"])
    else:
        asr = ASR(model_size=params["model_type"],
                  beam_size=params["beam_size"],
                  use_context=params["use_context"],
                  num_workers=params["num_workers"],
                  device="cuda" if use_gpu else "auto",
                  compute_type=params["compute_type"])

    transcribed_text = ""
    actual_text = ""
    with open(chunks_pkl, 'rb') as f:
        chunks = pickle.load(f)
    print(f"Number of chunks: {len(chunks)}")
    # sf.write("benchmark.wav", chunks, samplerate=sr)
    #Start timer
    times = []
    
    ## GPU VALUES
    total_GPU_VRAM_usage = 0
    peak_GPU_VRAM_usage = 0
    avg_GPU_VRAM_usage = 0

    total_GPU_clock_usage = 0
    peak_GPU_clock_usage = 0
    avg_GPU_clock_usage = 0


    ## RAM VALUES
    total_RAM_usage = 0
    peak_RAM_usage = 0
    avg_RAM_usage = 0

    for (chunk, isLastOfSpeech) in chunks:
        start_total_time = time.time()
        newText = asr.process_audio(chunk)
        if newText is not None:
            while newText.endswith('…'):
                newText = newText[:-1]
            transcribed_text += " " +  newText
        times.append(time.time() - start_total_time)
        ram_usage_chunk, gpu_usage_vram_chunk, gpu_usage_clock_chunk = measure_usage()
        #VRAM
        total_GPU_VRAM_usage += gpu_usage_vram_chunk
        if gpu_usage_vram_chunk > peak_GPU_VRAM_usage:
            peak_GPU_VRAM_usage = gpu_usage_vram_chunk

        #CLOCKSPEED
        total_GPU_clock_usage += gpu_usage_clock_chunk
        if gpu_usage_clock_chunk > peak_GPU_clock_usage:
            peak_GPU_clock_usage = gpu_usage_clock_chunk

        #RAM
        total_RAM_usage += ram_usage_chunk
        if ram_usage_chunk > peak_RAM_usage:
            peak_RAM_usage = ram_usage_chunk

    avg_GPU_VRAM_usage= total_GPU_VRAM_usage / len(chunks)
    avg_GPU_clock_usage = total_GPU_clock_usage / len(chunks)
    avg_RAM_usage = total_RAM_usage / len(chunks)
    #End timer
    #total time elapsed
    total_time = sum(times)
    average_chunk_time = "{:.7f}".format(np.average(times))
    max_chunk_time = "{:.7f}".format(max(times))
    min_chunk_time = "{:.7f}".format(min(times))

    # average_time_per_chunk = total_time / rounds
    with open(txt_filename, "r") as f:
        actual_text = f.read()
    # print(actual_text)

            # jiwer.ExpandCommonEnglishContractions(),
            # jiwer.RemoveEmptyStrings(),
            # jiwer.ToLowerCase(),
            # jiwer.RemoveMultipleSpaces(),
            # jiwer.Strip(),
            # jiwer.RemovePunctuation(),
            # jiwer.ReduceToListOfListOfWords(),
    transforms : jiwer.Compose = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemoveEmptyStrings(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )

    # print(transforms(transcribed_text))
    finaltext = " ".join(transforms(transcribed_text)[0])
    transformed_actual_text = " ".join(transforms(actual_text)[0])
    print(f"The actual text is:\n{transformed_actual_text}")
    print(f'\n\nThe Transcribed text was:\n{finaltext}')
    # wer = jiwer.wer(transformed_actual_text, transcribed_text, truth_transform=transforms, hypothesis_transform=transforms, )

    measures = jiwer.compute_measures(transformed_actual_text, finaltext, truth_transform=transforms, hypothesis_transform=transforms)
    # print(f"Measurements from Jiwer: \n{measures}")
    # print(f"    WER: {wer * 100:.1f}% between {txt_filename} and test")
    # print(f"    Total time using process_audio on test: {total_time} seconds")
    metric_table = PrettyTable(['Metric', 'value'])
    metric_table.add_row(['Max. chunk time', max_chunk_time])
    metric_table.add_row(['Min. chunk time', min_chunk_time])
    metric_table.add_row(['Avg. chunk time', average_chunk_time])
    metric_table.add_row(['Word Error Rate (WER)', f"{measures['wer'] * 100:.1f}% "])
    metric_table.add_row(['Word Information Loss (WIL)', f"{measures['wil'] * 100:.1f}%"])
    metric_table.add_row(['Total Transcription time', total_time])
    print(metric_table)
    # print(f"    Average time pr. chunk: {average_time_per_chunk} seconds, chunk size: {chunk_size}")
    results = {
        "Max. chunk time": max_chunk_time,
        "Min. chunk time": min_chunk_time,
        "Avg. chunk time": average_chunk_time,
        "Word Error Rate (WER)": f"{measures['wer'] * 100:.1f}%",
        "Word Information Loss (WIL)": f"{measures['wil'] * 100:.1f}%",
        "Total Transcription time": total_time,
        "Total GPU VRAM Usage": total_GPU_VRAM_usage,
        "Total GPU Clock Speed": total_GPU_clock_usage,
        "Peak GPU VRAM Usage": peak_GPU_VRAM_usage,
        "Peak GPU Clock Speed": peak_GPU_clock_usage,
        "Avg. GPU VRAM Usage": avg_GPU_VRAM_usage,
        "Avg. GPU Clock Speed": avg_GPU_VRAM_usage,
        "Total RAM Usage": total_RAM_usage,
        "Peak RAM Usage": peak_RAM_usage,
        "Avg. RAM Usage": avg_RAM_usage
    }

    return results
