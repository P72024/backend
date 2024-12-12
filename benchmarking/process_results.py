import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Fetch results from the server using scp


# Load the data
results = pd.read_csv('./results/results_avg.csv')
results["Chunk Size"] = results["Filename"].str.extract(r"^(\d+)-")[0].astype(int)  # Extract Chunk Size
results["VAD Threshold"] = results["Filename"].str.extract(r"-(\d+\.\d+).pkl")[0].astype(float)  # Extract VAD Threshold
results = results.drop(columns=["Filename"])

# results = results.where(results["model_type"] == "distil-large-v3")
results['Word Error Rate (WER)'] = results['Word Error Rate (WER)'].str.rstrip('%').astype(float)
results['Word Information Loss (WIL)'] = results['Word Information Loss (WIL)'].str.rstrip('%').astype(float)

config_params = ['Chunk Size', 'VAD Threshold', 'model_type', 'beam_size', 'use_context', 'confidence_based', 'num_workers', 'confidence_limit', 'Word Error Rate (WER)', 'Word Information Loss (WIL)', 'Avg. chunk time', 'Max. chunk time']
# Sort data by Word Error Rate (WER) for ordered plotting
results_avg_sorted: pd.DataFrame = results.sort_values(by=['Word Information Loss (WIL)', 'Word Error Rate (WER)', 'Avg. chunk time','Max. chunk time'])
print(f"Lowest WER config:\n{results_avg_sorted.head(15)[config_params].to_string()}")

# Convert boolean parameters to strings
results_avg_sorted['use_context'] = results_avg_sorted['use_context'].astype(str)
results_avg_sorted['confidence_based'] = results_avg_sorted['confidence_based'].astype(str)

# Define the parameters to plot
params = ['model_type', 'beam_size', 'use_context', 'confidence_based', 'num_workers', 'confidence_limit']
metrics = ['Word Error Rate (WER)', 'Word Information Loss (WIL)', 'Avg. chunk time']

# Create line plots with vertical error bars and connecting lines for each parameter and metric
for param in params:
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        means = results_avg_sorted.groupby(param)[metric].mean()
        stds = results_avg_sorted.groupby(param)[metric].std()
        plt.errorbar(x=means.index, y=means, yerr=stds, fmt='o', capsize=5, capthick=2, elinewidth=2)
        plt.plot(means.index, means, marker='o')
        plt.title(f'{metric} by {param}')
        plt.xlabel(param)
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(True)
        plt.show()
