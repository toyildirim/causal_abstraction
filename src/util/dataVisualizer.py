import pickle
import os
import networkx as nx
import numpy as np
import matplotlib
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox

# Set the backend for Linux/PyCharm compatibility
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


class DataVisualizer:
    """Logic class updated to handle both .pkl and .dot files."""
    @staticmethod
    def plot_data(data, filename):
        # 1. Handle Causal Graphs
        if isinstance(data, (nx.Graph, nx.DiGraph)):
            plt.figure(f"Causal Graph: {filename}", figsize=(10, 7))
            pos = nx.spring_layout(data, seed=42)
            nx.draw(data, pos, with_labels=True, node_color='skyblue',
                    node_size=1500, arrowsize=20, font_weight='bold')
            plt.title(f"Structure: {filename}")
            plt.show(block=False)  # block=False allows opening multiple windows

        # 2. Handle Sensor Data (Dictionaries)
        elif isinstance(data, dict):
            key = 'obs' if 'obs' in data else list(data.keys())[0]
            plt.figure(f"Heatmap: {key} ({filename})", figsize=(8, 5))
            sns.heatmap(data[key][:100, :], cmap='viridis')
            plt.title(f"Data: {key} in {filename}")
            plt.show(block=False)
        # 3. It's a Pandas DataFrame
        elif isinstance(data, pd.DataFrame):
            print(f"--- Detected Pandas DataFrame ---")
            print(data.head())
            plt.figure()
            data.iloc[:100, :10].plot(kind='line')  # Plot first 10 columns
            plt.title("DataFrame Overview")
        else:
            messagebox.showinfo("Data Info", f"Unknown format: {type(data)}")


class LauncherApp:
    """The Main GUI Frame."""

    def __init__(self, root):
        self.root = root
        self.root.title("Thesis Data Visualizer")
        self.root.geometry("400x200")

        # UI Layout
        self.label = tk.Label(root, text="Data Visualizer", font=("Arial", 14, "bold"))
        self.label.pack(pady=20)

        self.btn_browse = tk.Button(
            root,
            text="Choose Data File",
            command=self.load_and_visualize,
            width=20,
            height=2,
            bg="#4CAF50",
            fg="white"
        )
        self.btn_browse.pack(pady=10)

        self.status = tk.Label(root, text="Waiting for file...", fg="gray")
        self.status.pack(side="bottom", pady=10)

    def load_and_visualize(self):
        file_path = filedialog.askopenfilename(
            title="Select File",
            filetypes=[
                ("Supported Files", "*.pkl *.dot"),
                ("Pickle files", "*.pkl"),
                ("Graphviz files", "*.dot"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        filename = os.path.basename(file_path)
        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext == '.pkl':
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)

            elif ext == '.dot':
                # Convert .dot file to a NetworkX Graph
                # pydot is used as the parser
                data = nx.drawing.nx_pydot.read_dot(file_path)
                # NetworkX sometimes loads pydot graphs with extra nested info;
                # we convert to DiGraph for consistent plotting
                data = nx.DiGraph(data)

            else:
                raise ValueError("Unsupported file extension.")

            self.status.config(text=f"Displaying: {filename}", fg="green")
            DataVisualizer.plot_data(data, filename)

        except Exception as e:
            messagebox.showerror("Error", f"Could not process file:\n{str(e)}")
            self.status.config(text="Error loading file", fg="red")

if __name__ == "__main__":
    root = tk.Tk()
    app = LauncherApp(root)
    root.mainloop()