
import pandas as pd
import os

def get_absolute_path(relative_path):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), relative_path)

# Load your data from CSV or string
df = pd.read_csv(get_absolute_path('results/results_client.csv'))
print(df.head())


# # Compute the averages for each combination of min_chunk_size and speech_threshold
# # across all iterations (ignoring 'i' in this case)
df_grouped = df.pivot_table(
    values="timer_value",
    index=["min_chunk_size", "speech_threshold"],
    columns="timer_type",
    aggfunc="mean"
).reset_index()

# Renaming the columns for better readability
df_grouped.columns.name = None
df_grouped.rename(columns={
    "chunkProcessTime": "avg_chunkProcessTime",
    "chunkRoundTripTime": "avg_chunkRoundTripTime",
    "frontendToBackendSendTime": "avg_frontendToBackendSendTime",
    "backendToFrontendSendTime": "avg_backendToFrontendSendTime",
    "VADFilterTime": "avg_VADFilterTime"
}, inplace=True)

column_order = ["min_chunk_size", "speech_threshold", "avg_VADFilterTime", "avg_chunkProcessTime", "avg_chunkRoundTripTime", "avg_frontendToBackendSendTime", "avg_backendToFrontendSendTime"]

df_grouped = df_grouped[column_order]

# Save or display the result
print(df_grouped.head())

print(get_absolute_path('results/results_client_avg.csv'))

df_grouped.to_csv(get_absolute_path('results/results_client_avg.csv'), index=False)
