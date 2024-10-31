import asyncio

from benchmarking.process_audio_benchmark import process_audio_benchmark

async def run_benchmarks():
    await process_audio_benchmark("testfiles/MaleList57.wav", "testfiles/Male.txt")
    await process_audio_benchmark("testfiles/femaleList3.wav", "testfiles/Female.txt")


asyncio.run(run_benchmarks())