import asyncio

from benchmarking.process_audio_benchmark import process_audio_benchmark
from src.ASR.ASR import ASR


async def run_benchmarks():
    asr = ASR(model_size="base", device="auto", compute_type="float32")
    await process_audio_benchmark("testfiles/MaleList57.wav", "testfiles/Male.txt", asr)
    await process_audio_benchmark("testfiles/femaleList3.wav", "testfiles/Female.txt", asr)


asyncio.run(run_benchmarks())