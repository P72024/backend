import asyncio
import csv
import os
import pickle
import queue
import random
import subprocess
import sys
import threading
import time
from io import BytesIO
from multiprocessing import process
from typing import List

import GPUtil
import jiwer
import numpy as np
import psutil
from prettytable import PrettyTable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from ASR.ASR import ASR
from ASR.tweaked import ASR_tweaked
from Util import unix_seconds_to_ms


## GPU VALUES
total_GPU_VRAM_usage = "total_GPU_VRAM_usage"
peak_GPU_VRAM_usage = "peak_GPU_VRAM_usage"
avg_GPU_VRAM_usage = "avg_GPU_VRAM_usage"

total_GPU_clock_usage = "total_GPU_clock_usage"
peak_GPU_clock_usage = "peak_GPU_clock_usage"
avg_GPU_clock_usage = "avg_GPU_clock_usage"


## RAM VALUES
total_RAM_usage = "total_RAM_usage"
peak_RAM_usage = "peak_RAM_usage"
avg_RAM_usage = "avg_RAM_usage"


def measure_usage():
    # Get RAM usage in GB
    get_clockspeed_command = ["nvidia-smi", "--query-gpu=clocks.gr", "--format=csv,noheader,nounits"]
    ram_used_gb = psutil.virtual_memory().used / (1024 ** 3)  # Convert bytes to GB

    # Get GPU usage: VRAM used (in MB) and GPU MHz
    gpus = GPUtil.getGPUs()
    gpu_vram_used_mb = 0  # Default to 0 if no GPU is found
    gpu_mhz = 0           # Default to 0 if no GPU is found

    if gpus:
        # Assume single GPU for simplicity (you can loop through `gpus` if needed)
        gpu = gpus[0]
        gpu_vram_used_mb = gpu.memoryUsed  # VRAM used in MB
        process = subprocess.Popen(get_clockspeed_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output, error = process.communicate()
        # print(f"Clock speed output: {output}")
        if process.returncode == 0:
            gpu_mhz = int(output.strip())
        else: 
            gpu_mhz = 0  # GPU clock in MHz

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
                        compute_type="auto")
    else:
        asr = ASR(model_size=params["model_type"],
                beam_size=params["beam_size"],
                num_workers=params["num_workers"],
                device="cuda" if use_gpu else "auto",
                compute_type="auto")

    transcribed_text = ""
    actual_text = ""
    with open(chunks_pkl, 'rb') as f:
        chunks = pickle.load(f)
    # print(f"Number of chunks: {len(chunks)}")
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

    for chunk in chunks:
        (new_text, transcribe_time, _) = asr.process_audio(chunk, '1')
        transcribed_text += " " +  new_text
        times.append(transcribe_time)
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
    # print(f"The actual text is:\n{transformed_actual_text}")
    # print(f'\n\nThe Transcribed text was:\n{finaltext}')
    # wer = jiwer.wer(transformed_actual_text, transcribed_text, truth_transform=transforms, hypothesis_transform=transforms, )

    measures = jiwer.compute_measures(transformed_actual_text, finaltext, truth_transform=transforms, hypothesis_transform=transforms)
    # print(f"Measurements from Jiwer: \n{measures}")
    # print(f"    WER: {wer * 100:.1f}% between {txt_filename} and test")
    # print(f"    Total time using process_audio on test: {total_time} seconds")
    # metric_table = PrettyTable(['Metric', 'value'])
    # metric_table.add_row(['Max. chunk time', max_chunk_time])
    # metric_table.add_row(['Min. chunk time', min_chunk_time])
    # metric_table.add_row(['Avg. chunk time', average_chunk_time])
    # metric_table.add_row(['Word Error Rate (WER)', f"{measures['wer'] * 100:.1f}% "])
    # metric_table.add_row(['Word Information Loss (WIL)', f"{measures['wil'] * 100:.1f}%"])
    # metric_table.add_row(['Total Transcription time', total_time])
    # print(metric_table)
    # print(f"    Average time pr. chunk: {average_time_per_chunk} seconds, chunk size: {chunk_size}")
    results = {
        "Max. chunk time": max_chunk_time,
        "Min. chunk time": min_chunk_time,
        "Avg. chunk time": average_chunk_time,
        "Word Error Rate (WER)": round(measures['wer'] * 100, 1),
        "Word Information Loss (WIL)": round(measures['wil'] * 100, 1),
        "Total Transcription time": total_time,
        "Total GPU VRAM Usage": total_GPU_VRAM_usage,
        "Total GPU Clock Speed": total_GPU_clock_usage,
        "Peak GPU VRAM Usage": peak_GPU_VRAM_usage,
        "Peak GPU Clock Speed": peak_GPU_clock_usage,
        "Avg. GPU VRAM Usage": avg_GPU_VRAM_usage,
        "Avg. GPU Clock Speed": avg_GPU_clock_usage,
        "Total RAM Usage": total_RAM_usage,
        "Peak RAM Usage": peak_RAM_usage,
        "Avg. RAM Usage": avg_RAM_usage
    }

    return results

async def simulate_process_audio(result_dict, usages_dict, times_dict, asr_model, shared_queue: asyncio.Queue, stop_event: asyncio.Event, queue_times_dict: dict):
    while not stop_event.is_set() or not shared_queue.empty():
        try:
            (client_id, chunk, chunk_number) = await asyncio.wait_for(shared_queue.get(), timeout=5)
            queue_times_dict[client_id][chunk_number]["taken_out_of_queue"] = time.time()
            (new_text, transcribe_time, _) = await asyncio.to_thread(asr_model.process_audio, chunk, client_id) # So transcribing doesn't block adding audio to queue. Just like in our application
            times_dict[client_id].append(transcribe_time)
            result_dict[client_id]["transcription"] += " " +  new_text
            result_dict[client_id]["transcribe_time"] = transcribe_time

            ram_usage_chunk, gpu_usage_vram_chunk, gpu_usage_clock_chunk = measure_usage()
            #VRAM
            usages_dict[client_id][total_GPU_VRAM_usage] += gpu_usage_vram_chunk
            if gpu_usage_vram_chunk > usages_dict[client_id][peak_GPU_VRAM_usage]:
                usages_dict[client_id][peak_GPU_VRAM_usage] = gpu_usage_vram_chunk

            #CLOCKSPEED
            usages_dict[client_id][total_GPU_clock_usage] += gpu_usage_clock_chunk
            if gpu_usage_clock_chunk > usages_dict[client_id][peak_GPU_clock_usage]:
                usages_dict[client_id][peak_GPU_clock_usage] = gpu_usage_clock_chunk

            #RAM
            usages_dict[client_id][total_RAM_usage] += ram_usage_chunk
            if ram_usage_chunk > usages_dict[client_id][peak_RAM_usage]:
                usages_dict[client_id][peak_RAM_usage] = ram_usage_chunk
        except:
            pass
            # import traceback
            # traceback.print_exc()

async def put_audio_into_shared_queue(chunks, client_id, shared_queue: asyncio.Queue, queue_times_dict: dict):
    queue_times_dict[client_id] = dict()
    for idx, (chunk, _) in enumerate(chunks):
        queue_times_dict[client_id][idx] = { "inserted": time.time() }
        await shared_queue.put((client_id, chunk, idx))
        await asyncio.sleep(2.4) # Give control back to asyncio loop

async def process_audio_stress_test(chunks_pkls, txt_filenames, params: dict, use_gpu: bool):
    # Define required keys and their expected types
    required_keys_with_types = {
        "num_workers": int,
        "num_concurrent_rooms": int,
        "model_type": str,
        "beam_size": int,
        "use_context": bool,
        "confidence_limit": float,
        "confidence_based": bool,
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
                        compute_type="auto")
    else:
        asr = ASR(model_size=params["model_type"],
                beam_size=params["beam_size"],
                num_workers=params["num_workers"],
                device="cuda" if use_gpu else "auto",
                compute_type="auto")

    result_dict = dict()
    chunks_dict = dict()
    times_dict = dict()
    usages_dict = dict()
    queue_times_dict = dict()

    for client_id in range(0, params["num_concurrent_rooms"]):
        result_dict[client_id] = { 
            "transcription": "",
            "transcribe_time": 0,
        }

        with open(chunks_pkls[client_id], 'rb') as f:
            chunks_dict[client_id] = pickle.load(f)

        times_dict[client_id] = []

        usages_dict[client_id] = {
            total_GPU_VRAM_usage: "",
            peak_GPU_VRAM_usage: "",
            avg_GPU_VRAM_usage: "",
            total_GPU_clock_usage: "",
            peak_GPU_clock_usage: "",
            avg_GPU_clock_usage: "",
            total_RAM_usage: "",
            peak_RAM_usage: "",
            avg_RAM_usage: "",
        }
        ## GPU VALUES
        usages_dict[client_id][total_GPU_VRAM_usage] = 0
        usages_dict[client_id][peak_GPU_VRAM_usage] = 0
        usages_dict[client_id][avg_GPU_VRAM_usage] = 0

        usages_dict[client_id][total_GPU_clock_usage] = 0
        usages_dict[client_id][peak_GPU_clock_usage] = 0
        usages_dict[client_id][avg_GPU_clock_usage] = 0

        ## RAM VALUES
        usages_dict[client_id][total_RAM_usage] = 0
        usages_dict[client_id][peak_RAM_usage] = 0
        usages_dict[client_id][avg_RAM_usage] = 0

    shared_queue = asyncio.Queue()
    stop_event = asyncio.Event()

    put_audio_tasks = [
        asyncio.create_task(
            put_audio_into_shared_queue(chunks_dict[client_id], client_id, shared_queue, queue_times_dict)
        )
        for client_id in range(0, params["num_concurrent_rooms"])
    ]

    simulate_process_audio_tasks = [
        asyncio.create_task(
            simulate_process_audio(result_dict, usages_dict, times_dict, asr, shared_queue, stop_event, queue_times_dict)
        )
        # for _ in range(0, params["num_workers"])
    ]

    # Wait for all put audio tasks to finish
    await asyncio.gather(*put_audio_tasks)

    # Signal the stop event to tell clients to stop processing
    stop_event.set()

    await asyncio.gather(*simulate_process_audio_tasks)

    for client_id in range(0, params["num_concurrent_rooms"]):
        usages_dict[client_id][avg_GPU_VRAM_usage] = usages_dict[client_id][total_GPU_VRAM_usage] / len(chunks_pkls[client_id])
        usages_dict[client_id][avg_GPU_clock_usage] = usages_dict[client_id][total_GPU_clock_usage] / len(chunks_pkls[client_id])
        usages_dict[client_id][avg_RAM_usage] = usages_dict[client_id][total_RAM_usage] / len(chunks_pkls[client_id])

    total_times_dict = dict()
    average_chunk_times_dict = dict()
    max_chunk_times_dict = dict()
    min_chunk_times_dict = dict()

    queue_times_list_dict = dict()
    average_queue_times_dict = dict()
    max_queue_times_dict = dict()
    min_queue_times_dict = dict()

    actual_texts = dict()
    
    for client_id in range(0, params["num_concurrent_rooms"]):
        total_times_dict[client_id] = sum(times_dict[client_id])
        average_chunk_times_dict[client_id] = "{:.7f}".format(np.average(times_dict[client_id]))
        max_chunk_times_dict[client_id] = "{:.7f}".format(max(times_dict[client_id]))
        min_chunk_times_dict[client_id] = "{:.7f}".format(min(times_dict[client_id]))

        for (_, timers) in queue_times_dict[client_id].items():
            if client_id not in queue_times_list_dict:
                queue_times_list_dict[client_id] = [unix_seconds_to_ms(timers["taken_out_of_queue"] - timers["inserted"])]
            else:
                queue_times_list_dict[client_id].append(unix_seconds_to_ms(timers["taken_out_of_queue"] - timers["inserted"]))

        average_queue_times_dict[client_id] = "{:.7f}".format(np.average(queue_times_list_dict[client_id]))
        max_queue_times_dict[client_id] = "{:.7f}".format(max(queue_times_list_dict[client_id]))
        min_queue_times_dict[client_id] = "{:.7f}".format(min(queue_times_list_dict[client_id]))

        # average_time_per_chunk = total_time / rounds
        with open(txt_filenames[client_id], "r") as f:
            actual_texts[client_id] = f.read()
    # print(actual_texts)

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
    measures = dict()
    results = dict()
    for client_id in range(0, params["num_concurrent_rooms"]):
        finaltext = " ".join(transforms(result_dict[client_id]["transcription"])[0])
        transformed_actual_text = " ".join(transforms(actual_texts[client_id])[0])
        # print(f"The actual text is:\n{transformed_actual_text}")
        # print(f'\n\nThe Transcribed text was:\n{finaltext}')
        # wer = jiwer.wer(transformed_actual_text, transcribed_text, truth_transform=transforms, hypothesis_transform=transforms, )

        measures[client_id] = jiwer.compute_measures(transformed_actual_text, finaltext, truth_transform=transforms, hypothesis_transform=transforms)
        # print(f"Measurements from Jiwer: \n{measures}")
        # print(f"    WER: {wer * 100:.1f}% between {txt_filename} and test")
        # print(f"    Total time using process_audio on test: {total_time} seconds")
        # metric_table = PrettyTable(['Metric', 'value'])
        # metric_table.add_row(['Max. chunk time', max_chunk_time])
        # metric_table.add_row(['Min. chunk time', min_chunk_time])
        # metric_table.add_row(['Avg. chunk time', average_chunk_time])
        # metric_table.add_row(['Word Error Rate (WER)', f"{measures['wer'] * 100:.1f}% "])
        # metric_table.add_row(['Word Information Loss (WIL)', f"{measures['wil'] * 100:.1f}%"])
        # metric_table.add_row(['Total Transcription time', total_time])
        # print(metric_table)
        # print(f"    Average time pr. chunk: {average_time_per_chunk} seconds, chunk size: {chunk_size}")
        results[client_id] = {
            "Max. chunk time": max_chunk_times_dict[client_id],
            "Min. chunk time": min_chunk_times_dict[client_id],
            "Avg. chunk time": average_chunk_times_dict[client_id],
            "Max. queue time": max_queue_times_dict[client_id],
            "Min. queue time": min_queue_times_dict[client_id],
            "Avg. queue time": average_queue_times_dict[client_id],
            "Word Error Rate (WER)": round(measures[client_id]['wer'] * 100, 1),
            "Word Information Loss (WIL)": round(measures[client_id]['wil'] * 100, 1),
            "Total Transcription time": total_times_dict[client_id],
            "Total GPU VRAM Usage": usages_dict[client_id][total_GPU_VRAM_usage],
            "Total GPU Clock Speed": usages_dict[client_id][total_GPU_clock_usage],
            "Peak GPU VRAM Usage": usages_dict[client_id][peak_GPU_VRAM_usage],
            "Peak GPU Clock Speed": usages_dict[client_id][peak_GPU_clock_usage],
            "Avg. GPU VRAM Usage": usages_dict[client_id][avg_GPU_VRAM_usage],
            "Avg. GPU Clock Speed": usages_dict[client_id][avg_GPU_clock_usage],
            "Total RAM Usage": usages_dict[client_id][total_RAM_usage],
            "Peak RAM Usage": usages_dict[client_id][peak_RAM_usage],
            "Avg. RAM Usage": usages_dict[client_id][avg_RAM_usage],
        }

    return results
