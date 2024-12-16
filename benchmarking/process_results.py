import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def calculate_distance(x, y, weight):
    if pd.isna(x) or pd.isna(y):
        print(f"Debug: Invalid values encountered. Avg. chunk time: {x}, WIL/WER: {y}")
        return np.nan
    return np.sqrt(x**2 + y**2 + weight)

def plot_and_find_top_points(results, results_client, top_x_num_points, weight, werWilThreshold, take_latency_into_account, current_path):
    filtered_results = results[(results['Word Error Rate (WER)'] <= werWilThreshold) & (results['Word Information Loss (WIL)'] <= werWilThreshold)]
    
    results_client["Filename"] =  "" + results_client["min_chunk_size"].astype(str) + "-" + results_client["speech_threshold"].astype(str) + ".pkl"
    
    filtered_results = pd.merge(filtered_results, results_client, on='Filename')
    results = pd.merge(results, results_client, on='Filename')
    if current_path == "woman":
        results["total_chunk_time"] = results["Avg. chunk time"]/1000 + (results["avg_VADFilterTime"] + results["avg_chunkProcessTime"] + results["avg_chunkRoundTripTime"])/1000
    else:
        results["total_chunk_time"] = results["Avg. chunk time"] + (results["avg_VADFilterTime"] + results["avg_chunkProcessTime"] + results["avg_chunkRoundTripTime"])/1000
    
    if take_latency_into_account:
        if (current_path == "woman"):
            filtered_results["total_chunk_time"] = filtered_results["Avg. chunk time"]/1000 + (filtered_results["avg_VADFilterTime"] + filtered_results["avg_chunkProcessTime"] + filtered_results["avg_chunkRoundTripTime"])/1000
        else:
            filtered_results["total_chunk_time"] = filtered_results["Avg. chunk time"] + (filtered_results["avg_VADFilterTime"] + filtered_results["avg_chunkProcessTime"] + filtered_results["avg_chunkRoundTripTime"])/1000
            
    else:
        if (current_path == "woman"):
            filtered_results["Avg. chunk time"] = filtered_results["Avg. chunk time"]/1000
        else:
            filtered_results["Avg. chunk time"] = filtered_results["Avg. chunk time"]
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

    if take_latency_into_account:
        filtered_results['distance_wer'] = filtered_results.apply(lambda row: calculate_distance(row['total_chunk_time'], row['Word Error Rate (WER)'], weight), axis=1)
        top_wer_points = filtered_results.nsmallest(top_x_num_points, 'distance_wer')

        filtered_results['distance_wil'] = filtered_results.apply(lambda row: calculate_distance(row['total_chunk_time'], row['Word Information Loss (WIL)'], weight), axis=1)
        top_wil_points = filtered_results.nsmallest(top_x_num_points, 'distance_wil')
    else:
        filtered_results['distance_wer'] = filtered_results.apply(lambda row: calculate_distance(row['Avg. chunk time'], row['Word Error Rate (WER)'], weight), axis=1)
        top_wer_points = filtered_results.nsmallest(top_x_num_points, 'distance_wer')

        filtered_results['distance_wil'] = filtered_results.apply(lambda row: calculate_distance(row['Avg. chunk time'], row['Word Information Loss (WIL)'], weight), axis=1)
        top_wil_points = filtered_results.nsmallest(top_x_num_points, 'distance_wil')

    combined_points = pd.merge(top_wer_points, top_wil_points)
    if take_latency_into_account:
        plot_scatter(combined_points, 'Word Information Loss (WIL)', 'Word Error Rate (WER)', 'Word Information Loss (WIL)', 'Word Error Rate (WER)', 'WER vs. WIL for Combined Points (Latency Considered)')
    else:
        plot_scatter(combined_points, 'Word Information Loss (WIL)', 'Word Error Rate (WER)', 'Word Information Loss (WIL)', 'Word Error Rate (WER)', 'WER vs. WIL for Combined Points (Latency Not Considered)')
    
    return combined_points

def process_and_save_lowest_distance_points(top_combined_points, intervals, output_path, take_latency_into_account):
    # Save the row with the lowest distance combined for each interval of total_chunk_time
    lowest_distance_per_interval = pd.DataFrame()
    for interval in intervals:
        if take_latency_into_account:
            interval_points = top_combined_points[(top_combined_points['total_chunk_time'] >= interval - 1) & (top_combined_points['total_chunk_time'] < interval)]
        else:
            interval_points = top_combined_points[(top_combined_points['Avg. chunk time'] >= interval - 1) & (top_combined_points['Avg. chunk time'] < interval)]
        if not interval_points.empty:
            lowest_distance_row = interval_points.nsmallest(1, 'distance_combined')
            lowest_distance_per_interval = pd.concat([lowest_distance_per_interval, lowest_distance_row])

    # Save the results to a CSV file
    lowest_distance_per_interval.to_csv(output_path, index=False)

def distance_total_time_table(top_combined_points, take_latency_into_account):
    if take_latency_into_account:
        plot_scatter(top_combined_points, 'total_chunk_time', 'distance_combined', 'Total Chunk Time', 'WER+WIL Distance', 'Total Chunk Time vs. WER+WIL Distance (Latency Considered)')
    else:
        plot_scatter(top_combined_points, 'Avg. chunk time', 'distance_combined', 'Avg. chunk time', 'WER+WIL Distance', 'Avg. chunk time vs. WER+WIL Distance (Latency Not Considered)')

def extract_chunk_size(filename):
    return int(filename.split('-')[0])


def plot_error_bars(df, x_col, y_col, model_col, xlabel, ylabel, title):
    # Extract chunk size from Filename
    if x_col == 'min_chunk_size':
        df[x_col] = df['Filename'].apply(extract_chunk_size)
    
    # Group by chunk size and model, then calculate statistics
    stats = df.groupby([x_col, model_col])[y_col].agg(['mean', 'std', 'min', 'max']).reset_index()
    
    # Prepare data for categorical plotting
    chunk_sizes = sorted(df[x_col].unique())
    models = stats[model_col].unique()
    model_colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    x_positions = np.arange(len(chunk_sizes))
    width = 0.2  # Space between model bars
    
    plt.figure(figsize=(12, 6))
    
    # Loop through models and plot bars side-by-side for each chunk size
    for i, model in enumerate(models):
        model_stats = stats[stats[model_col] == model]
        
        # Compute adjusted positions for the model
        adjusted_positions = x_positions + (i - len(models) / 2) * width
        
        # Plot mean ± std dev
        plt.errorbar(
            adjusted_positions, model_stats['mean'], yerr=model_stats['std'],
            fmt='o', label=f'{model} Mean ± Std Dev', color=model_colors[i], capsize=5
        )
        
        # Add scatter points for min and max without legend for each model
        plt.scatter(adjusted_positions, model_stats['min'], color=model_colors[i], marker='v')
        plt.scatter(adjusted_positions, model_stats['max'], color=model_colors[i], marker='^')
    
    # Add a single legend for Min and Max
    plt.scatter([], [], color='black', marker='v', label='Min')  # Empty scatter for legend
    plt.scatter([], [], color='black', marker='^', label='Max')
    
    # Update axis ticks and labels
    plt.xticks(x_positions, chunk_sizes, rotation=45)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def plot_scatter(df, x_col, y_col, xlabel, ylabel, title):
    plt.figure(figsize=(10, 5))
    plt.scatter(df[x_col], df[y_col])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()


path_backend_woman = './results/Results_woman/raw_data/backend/results_avg_woman.csv'
path_client_woman = './results/Results_woman/raw_data/client/results_client_avg_woman.csv'
path_backend_man = './results/Results_man/raw_data/backend/results_avg.csv'
path_client_man = './results/Results_man/raw_data/client/results_client_avg.csv'
processed_results_woman = './results/Results_woman/processed_results/'
processed_results_man = './results/Results_man/processed_results/'
current_path = "woman"

results = pd.read_csv(path_backend_woman)
results_client = pd.read_csv(path_client_woman)

if current_path != 'woman':
    results['Word Error Rate (WER)'] = results['Word Error Rate (WER)'].str.rstrip('%').astype(float)
    results['Word Information Loss (WIL)'] = results['Word Information Loss (WIL)'].str.rstrip('%').astype(float)



def run_process_results (with_latency, current_path):
    # Parameters
    top_x_num_points = 10000
    weight = 1
    max_werWilThreshold = 200
    take_latency_into_account = with_latency

    # Call the function
    combined_points = plot_and_find_top_points(results, results_client, top_x_num_points, weight, max_werWilThreshold, take_latency_into_account, current_path)

    combined_points['distance_combined'] = combined_points.apply(lambda row: calculate_distance(row['Word Information Loss (WIL)'], row['Word Error Rate (WER)'], weight), axis=1)
    top_combined_points = combined_points.nsmallest(top_x_num_points, 'distance_combined')
    
    plot_error_bars(top_combined_points, 'min_chunk_size', 'distance_combined', 'model_type', 'Min chunk Size', 'WER + WIL Distance', 'Combined distance vs. Chunk Size')
    plot_error_bars(top_combined_points, 'min_chunk_size', 'total_chunk_time','model_type', 'Min chunk Size', 'Total chunk time', 'Latency vs. Chunk Size')
    plot_error_bars(top_combined_points, 'confidence_based', 'distance_combined', 'model_type', 'Confidence based', 'WER + WIL Distance', 'Combined distance vs. Confidence based')
    plot_error_bars(top_combined_points, 'confidence_based', 'total_chunk_time', 'model_type', 'Confidence based', 'Total chunk time', 'Latency vs. Confidence based')

    if take_latency_into_account:
        top_combined_points.to_csv(processed_results_woman + 'top_combined_points_with_latency.csv', index=False)
    else: 
        top_combined_points.to_csv(processed_results_woman + 'top_combined_points_without_latency.csv', index=False)

    distance_total_time_table(top_combined_points, take_latency_into_account)

    if take_latency_into_account:
        process_and_save_lowest_distance_points(top_combined_points, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], processed_results_woman + 'lowest_distance_per_interval_with_latency.csv', take_latency_into_account)
    else:
        process_and_save_lowest_distance_points(top_combined_points, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], processed_results_woman + 'lowest_distance_per_interval_without_latency.csv', take_latency_into_account)

run_process_results(True, current_path)