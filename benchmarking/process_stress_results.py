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

path_stress_test = './results/Stress_test/raw_data/stress_test_results.csv'
stress_test_results = pd.read_csv(path_stress_test)

def run_process_stress_results(results):
    df = pd.DataFrame(results, columns=["num_workers", "num_concurrent_rooms", "client_results"])
    df["client_results"] = df["client_results"].apply(ast.literal_eval)
    df1 = df[df["num_concurrent_rooms"] == 2]
    max_queue_time, min_queue_time, avg_queue_time, avg_queue_time_with_rooms = [], [], [], []

    for client_id, value in df.iloc[0]["client_results"].items():
        max_queue_time.append(float(value["Max. queue time"]) / 1000)
        min_queue_time.append(float(value["Min. queue time"]) / 1000)
        avg_queue_time.append(float(value["Avg. queue time"]) / 1000)
    
    for client_id, value in df1.iloc[0]["client_results"].items():
        avg_queue_time_with_rooms.append(float(value["Avg. queue time"]) / 1000)

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
            "Avg. queue time": avg_queue_time,
            "Number of workers": df1["num_workers"]
        }, 
        "Number of workers", 
        "Avg. queue time", 
        "Number of workers",
        "Avg. queue time", 
        "Avg. queue time vs. Number of workers"
    )

run_process_stress_results(stress_test_results)
