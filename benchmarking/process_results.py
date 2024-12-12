import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def calculate_distance(x, y, weight):
    if pd.isna(x) or pd.isna(y):
        print(f"Debug: Invalid values encountered. Avg. chunk time: {x}, WIL/WER: {y}")
        return np.nan
    return np.sqrt(x**2 + y**2 + weight)

def plot_and_find_top_points(results, results_client, top_x_num_points, weight, werWilThreshold, take_latency_into_account):
    filtered_results = results[(results['Word Error Rate (WER)'] <= werWilThreshold) & (results['Word Information Loss (WIL)'] <= werWilThreshold)]
    
    results_client["Filename"] =  "" + results_client["min_chunk_size"].astype(str) + "-" + results_client["speech_threshold"].astype(str) + ".pkl"
    
    filtered_results = pd.merge(filtered_results, results_client, on='Filename')
    if take_latency_into_account:
        filtered_results["total_chunk_time"] = filtered_results["Avg. chunk time"] + (filtered_results["avg_VADFilterTime"] + filtered_results["avg_chunkProcessTime"] + filtered_results["avg_chunkRoundTripTime"])/1000
    else:
        filtered_results["total_chunk_time"] = filtered_results["Avg. chunk time"]
    if take_latency_into_account:
        # Plot the scatter plot for WER vs. Total Chunk Time
        plot_scatter(filtered_results, 'total_chunk_time', 'Word Error Rate (WER)', 'Total chunk time', 'Word Error Rate (WER)', 'WER vs. Total Chunk Time')
        # Plot the scatter plot for WIL vs. Total Chunk Time
        plot_scatter(filtered_results, 'total_chunk_time', 'Word Information Loss (WIL)', 'Total chunk time', 'Word Information Loss (WIL)', 'WIL vs. Total Chunk Time')
    else:
        # Plot the scatter plot for WER vs. Avg. Chunk Time
        plot_scatter(filtered_results, 'Avg. chunk time', 'Word Error Rate (WER)', 'Avg. Chunk Time', 'Word Error Rate (WER)', 'WER vs. Avg. Chunk Time')
        # Plot the scatter plot for WIL vs. Avg. Chunk Time
        plot_scatter(filtered_results, 'Avg. chunk time', 'Word Information Loss (WIL)', 'Avg. Chunk Time', 'Word Information Loss (WIL)', 'WIL vs. Avg. Chunk Time')

    filtered_results['distance_wer'] = filtered_results.apply(lambda row: calculate_distance(row['total_chunk_time'], row['Word Error Rate (WER)'], weight), axis=1)
    top_wer_points = filtered_results.nsmallest(top_x_num_points, 'distance_wer')

    filtered_results['distance_wil'] = filtered_results.apply(lambda row: calculate_distance(row['total_chunk_time'], row['Word Information Loss (WIL)'], weight), axis=1)
    top_wil_points = filtered_results.nsmallest(top_x_num_points, 'distance_wil')

    combined_points = pd.merge(top_wer_points, top_wil_points)
    if take_latency_into_account:
        plot_scatter(combined_points, 'Word Information Loss (WIL)', 'Word Error Rate (WER)', 'Word Information Loss (WIL)', 'Word Error Rate (WER)', 'WER vs. WIL for Combined Points (Latency Considered)')
    else:
        plot_scatter(combined_points, 'Word Information Loss (WIL)', 'Word Error Rate (WER)', 'Word Information Loss (WIL)', 'Word Error Rate (WER)', 'WER vs. WIL for Combined Points (Latency Not Considered)')
    
    return combined_points

def process_and_save_lowest_distance_points(top_combined_points, intervals, output_path):
    # Save the row with the lowest distance combined for each interval of total_chunk_time
    lowest_distance_per_interval = pd.DataFrame()
    for interval in intervals:
        interval_points = top_combined_points[(top_combined_points['total_chunk_time'] >= interval - 1) & (top_combined_points['total_chunk_time'] < interval)]
        if not interval_points.empty:
            lowest_distance_row = interval_points.nsmallest(1, 'distance_combined')
            lowest_distance_per_interval = pd.concat([lowest_distance_per_interval, lowest_distance_row])

    # Save the results to a CSV file
    lowest_distance_per_interval.to_csv(output_path, index=False)

def distance_total_time_table(top_combined_points, take_latency_into_account):
    if take_latency_into_account:
        plot_scatter(top_combined_points, 'total_chunk_time', 'distance_combined', 'Total Chunk Time', 'WER+WIL Distance', 'Total Chunk Time vs. WER+WIL Distance (Latency Considered)')
    else:
        plot_scatter(top_combined_points, 'total_chunk_time', 'distance_combined', 'Avg. chunk time', 'WER+WIL Distance', 'Avg. chunk time vs. WER+WIL Distance (Latency Not Considered)')

def plot_scatter(df, x_col, y_col, xlabel, ylabel, title):
    plt.figure(figsize=(10, 5))
    plt.scatter(df[x_col], df[y_col])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()




results = pd.read_csv('./results/results.csv')
results_client = pd.read_csv('./results/results_client_avg.csv')

results['Word Error Rate (WER)'] = results['Word Error Rate (WER)'].str.rstrip('%').astype(float)
results['Word Information Loss (WIL)'] = results['Word Information Loss (WIL)'].str.rstrip('%').astype(float)


# Parameters
top_x_num_points = 10000
weight = 1
max_werWilThreshold = 30
take_latency_into_account = False

# Call the function
combined_points = plot_and_find_top_points(results, results_client, top_x_num_points, weight, max_werWilThreshold, take_latency_into_account)

combined_points['distance_combined'] = combined_points.apply(lambda row: calculate_distance(row['Word Information Loss (WIL)'], row['Word Error Rate (WER)'], weight), axis=1)
top_combined_points = combined_points.nsmallest(top_x_num_points, 'distance_combined')

top_combined_points.to_csv('./results/top_combined_points.csv', index=False)

distance_total_time_table(top_combined_points, take_latency_into_account)

process_and_save_lowest_distance_points(top_combined_points, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], './results/lowest_distance_per_interval.csv')