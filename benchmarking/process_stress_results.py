import json
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import numpy as np
import ast

rcParams.update({
    'font.size': 20,              
    'axes.titlesize': 22,
    'axes.labelsize': 21,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'font.family': 'serif',
})

def plot_concurrent_rooms(avg_df, raw_df, x_col, y_col, xlabel, ylabel, title, file_path):
    # coefficients = np.polyfit(avg_df[x_col], avg_df[y_col], 2)
    # a, b, c = coefficients
    # regression_line = a*(avg_df[x_col]**2) + b*(avg_df[x_col]) + c

    # y=f'{a:.2f}x^2 + {b:.2f}x + {c:.2f}'

    plot(avg_df[x_col], avg_df[y_col], raw_df[x_col], raw_df[y_col], xlabel, ylabel, title, file_path, type="rooms")


def plot_workers(avg_df, raw_df, x_col, y_col, xlabel, ylabel, title, file_path):
    plot(avg_df[x_col], avg_df[y_col], raw_df[x_col], raw_df[y_col], xlabel, ylabel, title, file_path, type="workers")

def plot(avg_x, avg_y, raw_x, raw_y, xlabel, ylabel, title, file_path, reg_line=None, reg_y=None, label=None, with_regression=None, type=None):
    plt.figure(figsize=(10, 5))

    if type == "rooms":
        # Create groups 
        index_groups = []
        for j in range(0, 3):
            index_group = []
            for i in range(raw_x.index[0] + j, raw_x.index[-1] + 1, raw_x.index[3] - raw_x.index[0]):
                index_group.append(int(i/(raw_x.index[0]) - 1 + i - raw_x.index[0])) if raw_x.index[0] != 0 else index_group.append(i)
            index_groups.append(index_group)
        for i, group in enumerate(index_groups):
            x = raw_x.iloc[group]
            y = [raw_y[index] for index in group]
            plt.plot(x, y, label=f"Iteration {i + 1}", linestyle='-', marker='o')
    elif type == "workers":
        # Create groups 
        index_groups = []
        for j in range(0, 3):
            x = 0
            index_group = []
            for i in range(raw_x.index[0] + j, raw_x.index[-1] + 1, raw_x.index[3] - raw_x.index[0]):
                p = 2 if j == 1 else 2
                index_group.append(int(i/i) - 1 + ((x + j) * 3) - (p * j)) if i != 0 else index_group.append(i)
                x += 1
                index_group.sort()
            index_groups.append(index_group)
        for i, group in enumerate(index_groups):
            x = raw_x.iloc[group]
            y = [raw_y[index] for index in group]
            plt.plot(x, y, label=f"Iteration {i + 1}", linestyle='-', marker='o')

    # Plot average data

    plt.scatter(avg_x, avg_y, color='red')
    plt.plot(avg_x, avg_y, color='red', label="Average", marker='o')
    
    # if with_regression:
    #     plt.plot(x, reg_line, label=label)
    #     plt.text(10, 40.5, reg_y, fontsize=12)
    # else:
    #     plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.savefig(file_path, format='svg')


path_stress_test = './results/Stress_test/raw_data/stress_test_results_avg.csv'
path_stress_test_raw = './results/Stress_test/raw_data/stress_test_results.csv'
stress_test_results = pd.read_csv(path_stress_test)
stress_test_results_raw = pd.read_csv(path_stress_test_raw)

def run_process_stress_results(results, raw_results):
    df = pd.DataFrame(results, columns=["num_workers", "num_concurrent_rooms", "client_results"])
    df["client_results"] = df["client_results"].apply(ast.literal_eval)
    df_raw = pd.DataFrame(raw_results, columns=["num_workers", "num_concurrent_rooms", "client_results"])
    df_raw["client_results"] = df_raw["client_results"].apply(ast.literal_eval)

    worker_range = [1, 2, 4, 8, 16]
    concurrent_range = [2, 4, 8, 16, 32]
    max_queue_time_dict, min_queue_time_dict, avg_queue_time_dict = {}, {}, {}

    avg_queue_time_with_rooms_dict, avg_queue_time_with_workers_dict, avg_queue_time_with_rooms_dict_raw, avg_queue_time_with_workers_dict_raw =  {}, {}, {}, {}

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
        df_specific_num_of_workers_raw = df_raw[df_raw["num_workers"] == workers]
        avg_queue_time_with_workers_raw = []
        avg_queue_time_with_workers_dict_raw = dict()

        for index, row in df_specific_num_of_workers.iterrows():
            avg_queue_time_with_workers_dict[index] = []
            for client_id, value in row["client_results"].items():
                avg_queue_time_with_workers_dict[index].append(float(value["Avg. queue time"]) / 1000)
        
        for index, val in avg_queue_time_with_workers_dict.items():
            avg_queue_time_with_workers.append(sum(val)/ len(val))
        
        # Raw data
        for index, row in df_specific_num_of_workers_raw.iterrows():
            avg_queue_time_with_workers_dict_raw[index] = []
            for client_id, value in row["client_results"].items():
                avg_queue_time_with_workers_dict_raw[index].append(float(value["Avg. queue time"]) / 1000)
        
        for index, val in avg_queue_time_with_workers_dict_raw.items():
            avg_queue_time_with_workers_raw.append(sum(val)/ len(val))

        plot_concurrent_rooms(
            { 
                "Avg. queue time": avg_queue_time_with_workers,
                "Number of concurrent rooms": df_specific_num_of_workers["num_concurrent_rooms"]
            }, 
            {
                "Avg. queue time": avg_queue_time_with_workers_raw,
                "Number of concurrent rooms": df_specific_num_of_workers_raw["num_concurrent_rooms"]
            },
            "Number of concurrent rooms", 
            "Avg. queue time", 
            "Number of concurrent rooms",
            "Avg. queue time (seconds)", 
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
        df_specific_num_of_rooms_raw = df_raw[df_raw["num_concurrent_rooms"] == con]
        avg_queue_time_with_rooms_raw = []
        avg_queue_time_with_rooms_dict_raw = dict()

        for index, row in df_specific_num_of_rooms.iterrows():
            avg_queue_time_with_rooms_dict[index] = []
            for client_id, value in row["client_results"].items():
                avg_queue_time_with_rooms_dict[index].append(float(value["Avg. queue time"]) / 1000)
        
        for index, val in avg_queue_time_with_rooms_dict.items():
            avg_queue_time_with_rooms.append(sum(val)/ len(val))

        # Raw
        for index, row in df_specific_num_of_rooms_raw.iterrows():
            avg_queue_time_with_rooms_dict_raw[index] = []
            for client_id, value in row["client_results"].items():
                avg_queue_time_with_rooms_dict_raw[index].append(float(value["Avg. queue time"]) / 1000)
        
        for index, val in avg_queue_time_with_rooms_dict_raw.items():
            avg_queue_time_with_rooms_raw.append(sum(val)/ len(val))

        plot_workers({ 
                "Avg. queue time": avg_queue_time_with_rooms,
                "Number of workers": df_specific_num_of_rooms["num_workers"]
            }, 
            { 
                "Avg. queue time": avg_queue_time_with_rooms_raw,
                "Number of workers": df_specific_num_of_rooms_raw["num_workers"]
            }, 
            "Number of workers", 
            "Avg. queue time", 
            "Number of workers",
            "Avg. queue time (seconds)", 
            "Avg. queue time vs. Number of workers",
            f"./results/Stress_test/processed_results/concurrent_rooms/{con}_concurrent_rooms.svg"
        )

run_process_stress_results(stress_test_results, stress_test_results_raw)
