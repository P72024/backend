import asyncio
import os
import sys
import yaml
import csv
from itertools import product

from memory_profiler import profile
from process_audio_benchmark import process_audio_benchmark
from ASR.ASR import ASR


def get_absolute_path(relative_path):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), relative_path)

# Add the src folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

config_file_path = get_absolute_path("config.yaml")
with open(config_file_path, "r") as f:
    config = yaml.safe_load(f)
parameters = config["grid_search"]["parameters"]
keys, values = zip(*parameters.items())
combinations = [dict(zip(keys, combo)) for combo in product(*values)]
total_combinations = len(combinations)



async def run_benchmarks():
    for i, params in enumerate(combinations, 1):
        print(f"Combination {i} of {total_combinations}: {params}")
        #TODO: kør tests på alle parametre fra config og evt flere (lige nu er det kun model_type/size). Test for alle testfilerne. Måske bare smid alle testfiler i samme mappe og så kør testen for alle filer i den mappe.
        #TODO: EVT. kør mere end een test og så tag et gennemsnit af alle resultaterne.

                                                                    # filnavn tilsvarer paramtetrene fra vadfilteret.
        results = await process_audio_benchmark(f"{get_absolute_path('testfiles/8-0.8.pkl')}", f"{get_absolute_path('testfiles/benchmark.txt')}", params["model_type"])

        params = [
            ['Param', 'Value'],
            ['Model', params['model_type']],
        ]
        results_file_path = get_absolute_path("results/results.csv")
        #TODO: resultaterne er ret dårlige pt. Tror måske der er noget galt med pkl filerne eller den måde vi læser dem på. Højst sandysnliggt sample rate eller lignende som ikke er blevet opdateret til det nye format fra VADfilteret på fronten.

        #TODO: skriv tidspunkt(dato), filnavn og parametre plus evalueringskriterier ud på een linje i csv. Og bare tilføj til csv'en i stedet for at overskrive den.
        with open(results_file_path, "w") as f:
            csv.writer(f).writerows(params)
            csv.writer(f).writerows(results)


asyncio.run(run_benchmarks())
