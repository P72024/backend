import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def calculate_distance(x, y, weight):
    if pd.isna(x) or pd.isna(y):
        print(f"Debug: Invalid values encountered. Avg. chunk time: {x}, WIL/WER: {y}")
        return np.nan
    return np.sqrt(x**2 + y**2 + weight)

def plot_and_find_top_points(results, x, weight, werWilThreshold):
    filtered_results = results[(results['Word Error Rate (WER)'] <= werWilThreshold) & (results['Word Information Loss (WIL)'] <= werWilThreshold)]
    
    plt.figure(figsize=(10, 5))
    plt.scatter(filtered_results['Avg. chunk time'], filtered_results['Word Error Rate (WER)'])
    plt.xlabel('Avg. Chunk Time')
    plt.ylabel('Word Error Rate (WER)')
    plt.title('WER vs. Avg. Chunk Time')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.scatter(filtered_results['Avg. chunk time'], filtered_results['Word Information Loss (WIL)'])
    plt.xlabel('Avg. Chunk Time')
    plt.ylabel('Word Information Loss (WIL)')
    plt.title('WIL vs. Avg. Chunk Time')
    plt.grid(True)
    plt.show()
    

    filtered_results['distance_wer'] = filtered_results.apply(lambda row: calculate_distance(row['Avg. chunk time'], row['Word Error Rate (WER)'], weight), axis=1)
    top_wer_points = filtered_results.nsmallest(x, 'distance_wer')

    filtered_results['distance_wil'] = filtered_results.apply(lambda row: calculate_distance(row['Avg. chunk time'], row['Word Information Loss (WIL)'], weight), axis=1)
    top_wil_points = filtered_results.nsmallest(x, 'distance_wil')

    print("filtered_results Head values of 'distance_wil':\n", filtered_results['distance_wil'].sort_values(ascending=False).head())
    print("filtered_results Tail values of 'distance_wil':\n", filtered_results['distance_wil'].sort_values(ascending=False).tail())
    print("Top wil: Head values of 'word information loss':\n", top_wil_points['distance_wil'].sort_values(ascending=False).head())
    print("Top wil: Tail values of 'word information loss':\n", top_wil_points['distance_wil'].sort_values(ascending=False).tail())
    
    combined_points = pd.merge(top_wer_points, top_wil_points)

    print("combined points: Head values of 'distance_wil':\n", combined_points['distance_wil'].sort_values(ascending=False).head())
    print("combined points: Tail values of 'distance_wil':\n", combined_points['distance_wil'].sort_values(ascending=False).tail())

    plt.figure(figsize=(10, 5))
    plt.scatter(combined_points['Word Information Loss (WIL)'], combined_points['Word Error Rate (WER)'])
    plt.xlabel('Word Information Loss (WIL)')
    plt.ylabel('Word Error Rate (WER)')
    plt.title('WER vs. WIL for Combined Points')
    plt.grid(True)
    plt.show()

    return combined_points

def display_table(df):
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Top Combined Points")

    frame = ttk.Frame(root)
    frame.pack(fill='both', expand=True)

    tree = ttk.Treeview(frame, columns=list(df.columns), show='headings')
    tree.pack(fill='both', expand=True)
    
    # Add a horizontal scrollbar
    hsb = ttk.Scrollbar(frame, orient='horizontal', command=tree.xview)
    hsb.pack(side='bottom', fill='x')
    tree.configure(xscrollcommand=hsb.set)

    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, anchor='center')

    for index, row in df.iterrows():
        tree.insert('', 'end', values=list(row))

    root.mainloop()


results = pd.read_csv('./results/results.csv')
results_client = pd.read_csv('./results/results_client.csv')

results['Word Error Rate (WER)'] = results['Word Error Rate (WER)'].str.rstrip('%').astype(float)
results['Word Information Loss (WIL)'] = results['Word Information Loss (WIL)'].str.rstrip('%').astype(float)


# Parameters
x = 400
weight = 1  


# Call the function
combined_points = plot_and_find_top_points(results, x, weight, 20.0)

combined_points['distance_combined'] = combined_points.apply(lambda row: calculate_distance(row['Word Information Loss (WIL)'], row['Word Error Rate (WER)'], weight), axis=1)
top_combined_points = combined_points.nsmallest(x, 'distance_combined')


top_combined_points.to_csv('./results/top_combined_points.csv', index=False)
# Display the top x data points in a Tkinter window
display_table(top_combined_points)