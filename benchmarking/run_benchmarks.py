import argparse
import asyncio
import csv
import os
import sys
from datetime import datetime
from itertools import product

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import yaml
from ASR.ASR import ASR
from memory_profiler import profile

from process_audio_benchmark import process_audio_benchmark


def get_absolute_path(relative_path):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), relative_path)

folder_path = 'testfiles/eval_files/'
results_file_path = get_absolute_path("results/results.csv")


# Add the src folder to the Python path
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Benchmarks with optional GPU usage")
    
    # Step 2: Add the --gpu argument with store_true, which defaults to False
    parser.add_argument('--gpu', action='store_true', help="Use GPU for benchmarking")
    
    # Parse arguments
    return parser.parse_args()

config_file_path = get_absolute_path("config.yaml")

# Final updated combinations
        






async def run_benchmarks(use_gpu : bool, combinations, files):
    total_combinations = len(combinations)
    for file_idx, (filename, file) in enumerate(files, 1):
        for i, params in enumerate(combinations, 1):
            print(f"Combination {i} of {total_combinations} with file {file_idx} of {len(files)}: {params}")
            #TODO: EVT. kør mere end een test og så tag et gennemsnit af alle resultaterne.
            transcription_results = await process_audio_benchmark(f"{get_absolute_path(file)}", f"{get_absolute_path('testfiles/benchmark.txt')}", params, use_gpu)

            csv_row = [
                datetime.today().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                filename]

            for (key, val) in params.items():
                if key == 'confidence_limit' and params["confidence_based"] == False:
                    csv_row.append("N/A")
                else:
                    csv_row.append(val)

            for (_, val) in transcription_results.items():
                csv_row.append(val)


            with open(results_file_path, "a", newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(csv_row)



async def generate_combinations(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    parameters = config["grid_search"]["parameters"]["global"]
    confidence_limits = config["grid_search"]["parameters"]["confidence"]["confidence_limit"]
    keys, values = zip(*parameters.items())
    combinations = [dict(zip(keys, combo)) for combo in product(*values)]
    updated_combinations = []
    for combination in combinations:
        if combination["confidence_based"]:
            # Update the original combination with the first confidence limit
            combination["confidence_limit"] = confidence_limits[0]
            updated_combinations.append(combination)

            # Create additional combinations for the other confidence limits
            for limit in confidence_limits[1:]:
                new_combination = combination.copy()
                new_combination["confidence_limit"] = limit
                updated_combinations.append(new_combination)
        else:
            # Keep the combination as is if confidence_based is False
            combination["confidence_limit"] = 0.0
            updated_combinations.append(combination)
    return updated_combinations


async def main():
    args = parse_arguments()
    titles = [
            "Date",
            "Filename",
                "compute_type",
            "model_type",
            "beam_size",
            "use_context",
            "confidence_based",
            "num_workers",
            "confidence_limit",
            "Max. chunk time",
            "Min. chunk time",
            "Avg. chunk time",
            "Word Error Rate (WER)",
            "Word Information Loss (WIL)",
            "Total Transcription Time",
            "Total GPU VRAM Usage",
            "Total GPU Clock Speed",
            "Peak GPU VRAM Usage",
            "Peak GPU Clock Speed",
            "Avg. GPU VRAM Usage",
            "Avg. GPU Clock Speed",
            "Total RAM Usage",
            "Peak RAM Usage",
            "Avg. RAM Usage",
            ]

    with open(results_file_path, "w") as f:
        csv.writer(f).writerow(titles)
    use_gpu = args.gpu  # True if --gpu is passed, otherwise False
    combinations = await generate_combinations(config_file_path)
    print(f"Using GPU: {use_gpu}")
    files = []
    for file in os.listdir(folder_path):
        if file.endswith('.pkl'):
            files.append((file, folder_path + file))
    # Example of conditional usage based on use_gpu
    if use_gpu:
        print("Running benchmarks on GPU...")
    else:
        print("Running benchmarks on CPU...")
    await run_benchmarks(use_gpu, combinations, files)

asyncio.run(main())

