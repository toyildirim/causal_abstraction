import pickle
import os
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog


class PklVisualizer:
    # def __init__(self, file_path):
    #     self.file_path = file_path
    #     self.data = self._load_file()
    def __init__(self, file_path=None):
        # If no path is provided, open the file browser immediately
        if not file_path:
            self.file_path = self.select_file_via_dialog()
        else:
            self.file_path = file_path

        if self.file_path:
            self.data = self._load_file()
        else:
            print("No file selected. Exiting.")
            self.data = None
    @staticmethod
    def select_file_via_dialog():
        """Opens a system file dialog to choose a .pkl file."""
        root = tk.Tk()
        root.withdraw()  # Hide the main tiny tkinter window
        root.attributes("-topmost", True)  # Bring the dialog to the front

        file_path = filedialog.askopenfilename(
            title="Select a Pickle (.pkl) file",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            initialdir=os.getcwd()
        )
        root.destroy()  # Cleanup the tkinter instance
        return file_path

    def _load_file(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"No file found at {self.file_path}")
        with open(self.file_path, 'rb') as f:
            return pickle.load(f)

    def summary(self):
        """Displays metadata based on the object type."""
        print(f"\n--- File: {os.path.basename(self.file_path)} ---")
        if isinstance(self.data, (nx.Graph, nx.DiGraph)):
            print(f"Type: NetworkX Graph | Nodes: {len(self.data)} | Edges: {len(self.data.edges())}")
        elif isinstance(self.data, dict):
            print(f"Type: Dictionary | Keys: {list(self.data.keys())}")
            # CASE 3: It's a Pandas DataFrame
        elif isinstance(self.data, pd.DataFrame):
            self._display_dataframe()
        else:
            print(f"Type: {type(self.data)}")

    # --- GRAPH METHODS ---
    def plot_graph(self):
        """Visualizes Causal Graphs."""
        if not isinstance(self.data, nx.Graph):
            print("Skip: Data is not a Graph object.")
            return

        plt.figure(figsize=(10, 7))
        pos = nx.spring_layout(self.data, seed=42)
        nx.draw(self.data, pos, with_labels=True, node_color='skyblue',
                node_size=1500, arrowsize=20, font_weight='bold', font_size=8)
        plt.title(f"Causal Structure: {os.path.basename(self.file_path)}")

    # --- DICTIONARY / SENSOR METHODS ---
    def plot_heatmap(self, key='obs', rows=100):
        if not isinstance(self.data, dict) or key not in self.data:
            return
        plt.figure(f"Heatmap: {key}")
        sns.heatmap(self.data[key][:rows, :], cmap='viridis')
        plt.title(f"Heatmap: {key}")

    def plot_distributions(self, keys=['red', 'green', 'blue']):
        if not isinstance(self.data, dict): return
        plt.figure("Distributions")
        for key in keys:
            if key in self.data:
                sns.kdeplot(self.data[key].flatten(), label=key, fill=True)
        plt.legend()
        plt.title("Value Distributions")

    def plot_correlations(self, key='obs'):
        if not isinstance(self.data, dict) or key not in self.data:
            return
        plt.figure(f"Correlation: {key}")
        corr = np.corrcoef(self.data[key].T)
        sns.heatmap(corr, cmap='coolwarm', center=0)
        plt.title(f"Feature Correlation: {key}")

    def _display_dataframe(self):
        print(f"--- Detected Pandas DataFrame ---")
        print(self.data.head())
        plt.figure()
        self.data.iloc[:100, :10].plot(kind='line')  # Plot first 10 columns
        plt.title("DataFrame Overview")

    # --- AUTOMATIC ROUTER ---
    def auto_visualize(self):
        """Smart function to just 'show me what's in here'."""
        self.summary()
        if isinstance(self.data, nx.Graph):
            self.plot_graph()
        elif isinstance(self.data, dict):
            self.plot_heatmap('obs')
            self.plot_correlations('obs')

if __name__ == "__main__":
    # Display the current working directory
    print(f"Current Directory: {os.getcwd()}")
    # 1. Specify the path to your .pkl file
    # FILE_PATH = '/home/taylanozgur/BackUp/taylanozgur/METU/CogS/Thesis/rePare/repare/src/expt/results/causalchamber/preprocessed/blocks.pkl'
    #FILE_PATH = '/home/taylanozgur/BackUp/taylanozgur/METU/CogS/Thesis/rePare/repare/src/expt/results/causalchamber/preprocessed/grouptargets.pkl'
    # FILE_PATH = '/home/taylanozgur/BackUp/taylanozgur/METU/CogS/Thesis/rePare/repare/src/expt/results/causalchamber/preprocessed/partition_parts.pkl'
    # FILE_PATH = '/home/taylanozgur/BackUp/taylanozgur/METU/CogS/Thesis/rePare/repare/src/expt/results/causalchamber/preprocessed/singleenvdata.pkl'
    # FILE_PATH = '/home/taylanozgur/BackUp/taylanozgur/METU/CogS/Thesis/rePare/repare/src/expt/results/causalchamber/preprocessed/singleenvlabels.pkl'
    # FILE_PATH = '/home/taylanozgur/BackUp/taylanozgur/METU/CogS/Thesis/rePare/repare/src/expt/results/causalchamber/preprocessed/singleenvtargets.pkl'
    # FILE_PATH = '/home/taylanozgur/BackUp/taylanozgur/METU/CogS/Thesis/rePare/repare/src/expt/results/causalchamber/preprocessed/truedagfull.pkl'
    # FILE_PATH = '/home/taylanozgur/BackUp/taylanozgur/METU/CogS/Thesis/rePare/repare/src/expt/results/causalchamber/preprocessed/truegraph.pkl'
    # FILE_PATH = '/home/taylanozgur/BackUp/taylanozgur/METU/CogS/Thesis/rePare/repare/src/expt/results/causalchamber/preprocessed/truelabels.pkl'
    try:
        # 2. Initialize the utility class
        # print(f"--- Loading: {FILE_PATH} ---")
        # viz = PklVisualizer(FILE_PATH)
        viz = PklVisualizer()
        # 3. Display data structure summary
        # This tells you exactly what keys and array sizes you are working with
        print("\n[STEP 1] Data Summary:")
        viz.summary()
        # 1. Trigger the plot creation
        viz.plot_graph()
        # OR viz.plot_heatmap('obs')

        #4. Generate Visualizations
        print("\n[STEP 2] Generating Heatmap for 'obs'...")
        # Showing the first 50 rows to see the pattern of the 20 features
        viz.plot_heatmap(key='obs', rows=50)

        print("\n[STEP 3] Comparing distributions for RGB keys...")
        # Helps identify if red, green, or blue data ranges differ
        viz.plot_distributions(keys=['red', 'green', 'blue'])

        print("\n[STEP 4] Calculating Correlation Matrix for 'obs'...")
        # Identifies which of the 20 features are redundant
        viz.plot_correlations(key='obs')

        # 2. Check if any figures were actually created
        if plt.get_fignums():
            print(f"Windows created. Attempting to display {len(plt.get_fignums())} plot(s)...")
            plt.show(block=True)
        else:
            print("Error: No figures were generated. Check if the data type was correct.")
        # # 4. Generate Visualizations
        # print("\n[STEP 2] Generating Heatmap for 'obs'...")
        # # Showing the first 50 rows to see the pattern of the 20 features
        # viz.plot_heatmap(key='obs', rows=50)
        #
        # print("\n[STEP 3] Comparing distributions for RGB keys...")
        # # Helps identify if red, green, or blue data ranges differ
        # viz.plot_distributions(keys=['red', 'green', 'blue'])
        #
        # print("\n[STEP 4] Calculating Correlation Matrix for 'obs'...")
        # # Identifies which of the 20 features are redundant
        # viz.plot_correlations(key='obs')

        # print("\n--- Visualization Complete ---")
        # print("--- Windows opened. Close the windows to end the script. ---")
        #
        # # This is the "Anchor" that keeps everything alive
        # plt.show()

    # except FileNotFoundError:
    #     print(f"Error: Could not find the file at '{FILE_PATH}'. "
    #           "Please check the path and try again.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")