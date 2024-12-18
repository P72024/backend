import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast

def plot_concurrent_rooms(df, x_col, y_col, xlabel, ylabel, title, file_path):
    coefficients = np.polyfit(df[x_col], df[y_col], 2)
    a, b, c = coefficients
    regression_line = a*(df[x_col]**2) + b*(df[x_col]) + c

    y=f'{a:.2f}x^2 + {b:.2f}x + {c:.2f}'

    plot(df[x_col], df[y_col], xlabel, ylabel, title, file_path, reg_line=regression_line, reg_y=y, label='Regression Line (y={a:.2f}x^2 + {b:.2f}x + {c:.2f}', with_regression=True)


def plot_workers(df, x_col, y_col, xlabel, ylabel, title, file_path):
    plot(df[x_col], df[y_col], xlabel, ylabel, title, file_path)

def plot(x, y, xlabel, ylabel, title, file_path, reg_line=None, reg_y=None, label=None, with_regression=None):
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y)
    
    if with_regression:
        plt.plot(x, reg_line, label=label)
        plt.text(10, 40.5, reg_y, fontsize=12)
    else:
        plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(file_path, format='svg')


path_stress_test = './results/Stress_test/raw_data/stress_test_results_avg.csv'
stress_test_results = pd.read_csv(path_stress_test)

def run_process_stress_results(results):
    df = pd.DataFrame(results, columns=["num_workers", "num_concurrent_rooms", "client_results"])
    df["client_results"] = df["client_results"].apply(ast.literal_eval)
    worker_range = [1, 2, 4, 8, 16]
    concurrent_range = [2, 4, 8, 16, 32]
    max_queue_time_dict, min_queue_time_dict, avg_queue_time_dict, avg_queue_time_with_rooms_dict, avg_queue_time_with_workers_dict = {}, {}, {}, {}, {}

    for index, row in df.iterrows():
        max_queue_time_dict[index] = []
        min_queue_time_dict[index] = []
        avg_queue_time_dict[index] = []
        for client_id, value in row["client_results"].items():
            max_queue_time_dict[index].append(float(value["Max. queue time"]) / 1000)
            min_queue_time_dict[index].append(float(value["Min. queue time"]) / 1000)
            avg_queue_time_dict[index].append(float(value["Avg. queue time"]) / 1000)
    
    for workers in worker_range:
        df_specific_num_of_workers = df[df["num_workers"] == workers]
        avg_queue_time_with_workers = []
        avg_queue_time_with_workers_dict = dict()

        for index, row in df_specific_num_of_workers.iterrows():
            avg_queue_time_with_workers_dict[index] = []
            for client_id, value in row["client_results"].items():
                avg_queue_time_with_workers_dict[index].append(float(value["Avg. queue time"]) / 1000)
        
        for index, val in avg_queue_time_with_workers_dict.items():
            avg_queue_time_with_workers.append(sum(val)/ len(val))

        plot_concurrent_rooms({ 
                "Avg. queue time": avg_queue_time_with_workers,
                "Number of concurrent rooms": df_specific_num_of_workers["num_concurrent_rooms"]
            }, 
            "Number of concurrent rooms", 
            "Avg. queue time", 
            "Number of concurrent rooms",
            "Avg. queue time", 
            "Avg. queue time vs. Number of concurrent rooms",
            f"./results/Stress_test/processed_results/number_of_workers/{workers}_workers.svg"

        )


    max_queue_time = []
    min_queue_time = []
    avg_queue_time = []

    for index, val in max_queue_time_dict.items():
        max_queue_time.append(sum(val)/ len(val))
    for index, val in min_queue_time_dict.items():
        min_queue_time.append(sum(val)/ len(val))
    for index, val in avg_queue_time_dict.items():
        avg_queue_time.append(sum(val)/ len(val))

    

    # Latency for number of workers with specific number of rooms, so that we can conclude what effect workers have on queue time
    for con in concurrent_range:
        df_specific_num_of_rooms = df[df["num_concurrent_rooms"] == con]
        avg_queue_time_with_rooms = []
        avg_queue_time_with_rooms_dict = dict()

        for index, row in df_specific_num_of_rooms.iterrows():
            avg_queue_time_with_rooms_dict[index] = []
            for client_id, value in row["client_results"].items():
                avg_queue_time_with_rooms_dict[index].append(float(value["Avg. queue time"]) / 1000)
        
        for index, val in avg_queue_time_with_rooms_dict.items():
            avg_queue_time_with_rooms.append(sum(val)/ len(val))

        plot_workers({ 
                "Avg. queue time": avg_queue_time_with_rooms,
                "Number of workers": df_specific_num_of_rooms["num_workers"]
            }, 
            "Number of workers", 
            "Avg. queue time", 
            "Number of workers",
            "Avg. queue time", 
            "Avg. queue time vs. Number of workers",
            f"./results/Stress_test/processed_results/concurrent_rooms/{con}_concurrent_rooms.svg"
        )

run_process_stress_results(stress_test_results)
