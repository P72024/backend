import argparse
import asyncio
import csv
import os
import sys
from datetime import datetime
from itertools import product

from rich.progress import (BarColumn, Progress, TaskProgressColumn, TextColumn,
                           TimeRemainingColumn)

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
        
num_iterations = 3






async def run_benchmarks(use_gpu : bool, combinations, files):
    total_combinations = len(combinations)
    total_files = len(files)
    with Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TimeRemainingColumn(),
) as progress:
        combinationsProgress = progress.add_task("[green]Benchmarking Combinations...", total=total_combinations)
        fileProgress = progress.add_task("[red]Benchmarking files...", total=total_files)
        iterationProgress = progress.add_task("[yellow]Running Iterations...", total=num_iterations)
        
        for file_idx, (filename, file) in enumerate(files, 1):
            progress.reset(combinationsProgress)
            for i, params in enumerate(combinations, 1):
                progress.reset(iterationProgress)
                progress.console.print(f"Combination {i} of {total_combinations} with file {file_idx} of {len(files)}: {params}")
                #TODO: EVT. kør mere end een test og så tag et gennemsnit af alle resultaterne.

                results_array = []
                for i in range(num_iterations):
                    transcription_results = await process_audio_benchmark(f"{get_absolute_path(file)}", f"{get_absolute_path('testfiles/benchmark.txt')}", params, use_gpu)
                    results_array.append(transcription_results)

                # calculate the average values from the iterations
                # print(results_array)
                results = {}

                for key in results_array[0]:
                    # Initialize a list to store the converted values for each iteration
                    values = []

                    # Handle numeric values or percentages
                    for result in results_array:
                        value = result[key]
                        
                        # Convert values to float if they are strings (ignore percentage symbols)
                        if isinstance(value, str):
                            if "%" in value:
                                # Remove the '%' symbol and convert to float
                                value = float(value.strip('%'))
                            else:
                                # Convert to float directly
                                value = float(value)

                        values.append(value)

                    # Compute average for numeric values
                    if isinstance(values[0], (int, float)):  # Numeric values
                        # Average for numeric values
                        avg_value = sum(values) / num_iterations
                        # For percentage values, format them back to a percentage string
                        if isinstance(result[key], str) and "%" in result[key]:
                            results[key] = f"{avg_value:.1f}%"
                        else:
                            results[key] = avg_value
                    else:
                        # For non-numeric or unknown types, set as "N/A" or handle accordingly
                        results[key] = "N/A"
                # Print or use the final averaged results
                # print(f"The average result is: {results}")


                csv_row = [
                    datetime.today().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                    filename]

                for (key, val) in params.items():
                    if key == 'confidence_limit' and params["confidence_based"] == False:
                        csv_row.append("N/A")
                    else:
                        csv_row.append(val)

                for (_, val) in results.items():
                    csv_row.append(val)


                with open(results_file_path, "a", newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(csv_row)
                progress.update(combinationsProgress, advance=1)
            progress.update(fileProgress, advance=1)



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
    folder = os.path.dirname(results_file_path)
    if folder:  # Only create folder if there's a directory specified
        os.makedirs(folder, exist_ok=True)
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

