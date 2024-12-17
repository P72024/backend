import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast

def plot_scatter(df, x_col, y_col, xlabel, ylabel, title):
    plt.figure(figsize=(10, 5))
    plt.scatter(df[x_col], df[y_col])
    plt.plot(df[x_col], df[y_col])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()

path_stress_test = './results/Stress_test/raw_data/stress_test_results_avg.csv'
stress_test_results = pd.read_csv(path_stress_test)

def run_process_stress_results(results):
    df = pd.DataFrame(results, columns=["num_workers", "num_concurrent_rooms", "client_results"])
    df["client_results"] = df["client_results"].apply(ast.literal_eval)
    df_specific_num_of_rooms = df[df["num_concurrent_rooms"] == 2]
    max_queue_time_dict, min_queue_time_dict, avg_queue_time_dict, avg_queue_time_with_rooms_dict = {}, {}, {}, {}

    for index, row in df.iterrows():
        max_queue_time_dict[index] = []
        min_queue_time_dict[index] = []
        avg_queue_time_dict[index] = []
        for client_id, value in row["client_results"].items():
            max_queue_time_dict[index].append(float(value["Max. queue time"]) / 1000)
            min_queue_time_dict[index].append(float(value["Min. queue time"]) / 1000)
            avg_queue_time_dict[index].append(float(value["Avg. queue time"]) / 1000)
    
    for index, row in df_specific_num_of_rooms.iterrows():
        avg_queue_time_with_rooms_dict[index] = []
        for client_id, value in row["client_results"].items():
            avg_queue_time_with_rooms_dict[index].append(float(value["Max. queue time"]) / 1000)

    max_queue_time = []
    min_queue_time = []
    avg_queue_time = []
    avg_queue_time_with_rooms = []

    for index, val in max_queue_time_dict.items():
        max_queue_time.append(sum(val)/ len(val))
    for index, val in min_queue_time_dict.items():
        min_queue_time.append(sum(val)/ len(val))
    for index, val in avg_queue_time_dict.items():
        avg_queue_time.append(sum(val)/ len(val))
    for index, val in avg_queue_time_with_rooms_dict.items():
        avg_queue_time_with_rooms.append(sum(val)/ len(val))

    plot_scatter({ 
            "Avg. queue time": avg_queue_time,
            "Number of concurrent rooms": df["num_concurrent_rooms"]
        }, 
        "Number of concurrent rooms", 
        "Avg. queue time", 
        "Number of concurrent rooms",
        "Avg. queue time", 
        "Avg. queue time vs. Number of concurrent rooms"
    )

    # Latency for number of workers with specific number of rooms, so that we can conclude what effect workers have on queue time
    plot_scatter({ 
            "Avg. queue time": avg_queue_time_with_rooms,
            "Number of workers": df_specific_num_of_rooms["num_workers"]
        }, 
        "Number of workers", 
        "Avg. queue time", 
        "Number of workers",
        "Avg. queue time", 
        "Avg. queue time vs. Number of workers"
    )

run_process_stress_results(stress_test_results)
