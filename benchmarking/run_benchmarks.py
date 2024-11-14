import asyncio
import os
import sys

from memory_profiler import profile

from process_audio_benchmark import process_audio_benchmark

# Add the src folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from ASR.ASR import ASR

size="medium"
chunk_limit=48
context_length_limit=80

def get_absolute_path(relative_path):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), relative_path)

@profile
async def run_benchmarks():
    await process_audio_benchmark(f"{get_absolute_path('testfiles/parl1.pkl')}", f"{get_absolute_path('testfiles/parl1_meta.pkl')}", f"{get_absolute_path('testfiles/parl1.txt')}", size, context_length_limit, chunk_limit)
    # await process_audio_benchmark("./testfiles/test1.pkl", "./testfiles/test1_meta.pkl", "testfiles/test1_transcription.txt", asr)
    # await process_audio_benchmark("./testfiles/test1.pkl", "./testfiles/test1_meta.pkl", "./testfiles/test1.txt", asr)
    # await process_audio_benchmark("./testfiles/test2.pkl", "./testfiles/test2_meta.pkl", "./testfiles/test2.txt", asr)
    # await process_audio_benchmark("./testfiles/test2.pkl", "./testfiles/test2_meta.pkl", "testfiles/test2_transcription.txt", asr)
    # await process_audio_benchmark(f"{get_absolute_path('testfiles/ozzy.pkl')}", f"{get_absolute_path('testfiles/ozzy_meta.pkl')}", f"{get_absolute_path('testfiles/ozzy.txt')}", size)
    # await process_audio_benchmark(f"{get_absolute_path('testfiles/peter2.pkl')}", f"{get_absolute_path('testfiles/peter2_meta.pkl')}", f"{get_absolute_path('testfiles/peter2.txt')}", size)
    # await process_audio_benchmark("./testfiles/peter.pkl", "./testfiles/peter_meta.pkl", "./testfiles/peter.txt", size)
    # await process_audio_benchmark("./testfiles/casper.pkl", "./testfiles/casper_meta.pkl", "./testfiles/casper.txt", size)
    # await process_audio_benchmark("./testfiles/frederik.pkl", "./testfiles/frederik_meta.pkl", "./testfiles/frederik.txt", size)

asyncio.run(run_benchmarks())
