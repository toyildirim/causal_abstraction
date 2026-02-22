import pickle
import os
import networkx as nx
import numpy as np
import matplotlib
import pandas as pd
import pydot
import tkinter as tk
from tkinter import filedialog, messagebox

from numpy.ma.core import size

from rdaConverter import RdaConverter as rda
from file_utils import FileUtil as fu
from abstraction_methods.cagres.cagres import CaGreS as cg
from abstraction_methods.reducedag.reduceDag import DAGReducer as dr

# Set the backend for Linux/PyCharm compatibility
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from pyvis.network import Network
import webbrowser

# Constants
DATA_PATH = '/home/taylanozgur/BackUp/taylanozgur/METU/CogS/Thesis/MyProjects/DataVisualizer/data/'
HTML_DIR = os.path.join(DATA_PATH, "html")

class DataVisualizer:
    """Handles visualization for Graphs and Sensor Data."""

    @staticmethod
    def plot_data(data, filename):
        """Routes data to the appropriate visualizer based on type."""
        base_name = os.path.basename(filename)

        # 1. Direct Graph Objects
        if isinstance(data, (nx.Graph, nx.DiGraph)):
            DataVisualizer._launch_graph_windows(data, base_name)

        # 2. Dictionaries (RDA outpu
        # t or sensor groups)
        elif isinstance(data, dict):
            # Detect if it is a spatial collection
            if all(k in data for k in ['coordinates', 'observations']):
                DataVisualizer._plot_spatial_dataset(data, base_name)
            else:
                isCollection = False
                for key, val in data.items():
                    title = f"{base_name}_{key}"
                    if isinstance(val, (nx.Graph, nx.DiGraph)):
                        exposures, outcomes = DataVisualizer.find_causal_targets(val)
                        DataVisualizer._launch_graph_windows(val, title)
                    elif isinstance(val, (pd.DataFrame, np.ndarray)):
                        isCollection = True
                        break
                if isCollection:
                    DataVisualizer._plot_heatmap(data)
                    DataVisualizer._plot_distributions(data)
                    DataVisualizer._plot_correlations(data)
                        # DataVisualizer._plot_heatmap(val, title)
                # DataVisualizer._plot_distributions(data)
                # DataVisualizer._plot_correlations(data)
            # elif isinstance(val, (pd.DataFrame, np.ndarray)):
            #     isDict = True
            #     break
            # if isDict:
            #     DataVisualizer._plot_heatmap(data)
            #     DataVisualizer._plot_distributions(data)
            #     DataVisualizer._plot_correlations(data)

        # 3. Direct DataFrames
        elif isinstance(data, pd.DataFrame):
            DataVisualizer._plot_dataframe_overview(data, base_name)

        else:
            messagebox.showinfo("Data Info", f"Unknown format: {type(data)}")

    # @staticmethod
    # def _plot_heatmap(val, title):
    #     """Internal helper for sensor data heatmaps."""
    #     plt.figure(f"Heatmap: {title}", figsize=(8, 5))
    #     # Use iloc for safe slicing of DataFrames
    #     target = val.iloc[:100, :15] if isinstance(val, pd.DataFrame) else val[:100, :15]
    #     sns.heatmap(target, cmap='viridis')
    #     plt.title(title)
    #     plt.show(block=False)
    @staticmethod
    def _plot_heatmap(data, key='obs', rows=100):
        if not isinstance(data, dict) or key not in data:
            return
        plt.figure(f"Heatmap: {key}")
        sns.heatmap(data[key][:rows, :], cmap='viridis')
        plt.title(f"Heatmap: {key}")

    @staticmethod
    def _plot_dataframe_overview(df, title):
        """Internal helper for standard DataFrame line plots."""
        print(f"--- Detected Pandas DataFrame: {title} ---")
        plt.figure(f"Overview: {title}")
        df.iloc[:100, :10].plot(kind='line')
        plt.title(f"Overview: {title}")
        plt.show(block=False)

    @staticmethod
    def _plot_distributions(data, keys=['red', 'green', 'blue']):
        if not isinstance(data, dict): return
        plt.figure("Distributions")
        for key in keys:
            if key in data:
                sns.kdeplot(data[key].flatten(), label=key, fill=True)
        plt.legend()
        plt.title("Value Distributions")

    @staticmethod
    def _plot_correlations(data, key='obs'):
        if not isinstance(data, dict) or key not in data:
            return
        plt.figure(f"Correlation: {key}")
        corr = np.corrcoef(data[key].T)
        sns.heatmap(corr, cmap='coolwarm', center=0)
        plt.title(f"Feature Correlation: {key}")

    @staticmethod
    def _plot_spatial_dataset(data_dict, title):
        """
        Specialized plotter for dictionaries with 'coordinates',
        'observations', and 'targets'.
        """
        # 1. Prepare Coordinates
        coords = data_dict.get('coordinates')
        # Flatten from [[val], [val]] to [val, val]
        x_ticks = np.array(coords['x']).flatten()
        y_ticks = np.array(coords['y']).flatten()

        # 2. Setup Figure: Top row for Spatial Maps, Bottom for Temporal Heatmaps
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
        fig.suptitle(f"Spatial-Temporal Analysis: {title}", fontsize=16)

        for i, key in enumerate(['observations', 'targets']):
            data = np.array(data_dict[key])

            # --- SPATIAL SNAPSHOT (Top Row) ---
            # Reshape first sample (row 0) to grid dimensions (len(y), len(x))
            try:
                spatial_grid = data[0].reshape(len(y_ticks), len(x_ticks))
                im_spatial = axes[0, i].pcolormesh(x_ticks, y_ticks, spatial_grid,
                                                   shading='auto', cmap='RdBu_r')
                plt.colorbar(im_spatial, ax=axes[0, i])
                axes[0, i].set_title(f"{key}: Spatial Map (Sample 0)")
                axes[0, i].set_xlabel("Longitude")
                axes[0, i].set_ylabel("Latitude")
            except ValueError:
                axes[0, i].text(0.5, 0.5, "Reshape Error: Dimension Mismatch", ha='center')

            # --- TEMPORAL HEATMAP (Bottom Row) ---
            # Time (Rows) vs Features (Columns)
            # We slice to first 100 samples and first 50 features for clarity
            sns.heatmap(data[:100, :50], ax=axes[1, i], cmap='viridis', cbar=True)
            axes[1, i].set_title(f"{key}: Time-Series Heatmap (Overview)")
            axes[1, i].set_xlabel("Feature Index")
            axes[1, i].set_ylabel("Time Sample")

        plt.show(block=False)


        # Check the relationship between the first observation feature and the first target
        # correlation, p_value = stats.pearsonr(data['observations'][:, 0], data['targets'][:, 0])
        #
        # print(f"Correlation: {correlation:.2f}")
        # print(f"Statistical Significance (p-value): {p_value:.4f}")

    @staticmethod
    def find_causal_targets(G):
        # Find nodes with no parents (Potential Exposures)
        exposures = [n for n, deg in G.in_degree() if deg == 0]

        # Find nodes with no children (Potential Outcomes)
        outcomes = [n for n, deg in G.out_degree() if deg == 0]
        # 3. Log the results
        print("-" * 30)
        print("AUTOMATIC CAUSAL LOG")
        print("-" * 30)
        print(f"Potential Exposures (Sources): {', '.join(exposures) if exposures else 'None'}")
        print(f"Potential Outcomes (Sinks):    {', '.join(outcomes) if outcomes else 'None'}")
        print("-" * 30)
        return exposures, outcomes

    @staticmethod
    def _launch_graph_windows(graph_obj, title):
        """Standardizes dual-view (Matplotlib + Pyvis) for DAGs."""
        # --- DESKTOP VIEW ---
        plt.figure(title, figsize=(10, 7))
        pos = nx.spring_layout(graph_obj, seed=42)
        nx.draw(graph_obj, pos, with_labels=True, node_color='skyblue',
                node_size=1500, arrowsize=20, font_weight='bold')
        plt.title(title)
        plt.show(block=False)

        # --- BROWSER VIEW (Pyvis) ---
        net = Network(height='750px', width='100%', notebook=False, directed=True)

        # Color coding for Causal Roles
        for node, attr in graph_obj.nodes(data=True):
            color = 'red' if 'exposure' in str(attr) else 'green' if 'outcome' in str(attr) else 'skyblue'
            label = f"{'EXPOSURE' if color == 'red' else 'OUTCOME' if color == 'green' else ''}: {node}"
            net.add_node(node, label=label, color=color)

        for u, v in graph_obj.edges():
            net.add_edge(u, v)

        net.toggle_physics(True)  #

        # Ensure HTML directory exists
        os.makedirs(HTML_DIR, exist_ok=True)
        safe_name = "".join([c if c.isalnum() else "_" for c in title])
        output_path = os.path.join(HTML_DIR, f"graph_{safe_name}.html")

        net.write_html(output_path)
        webbrowser.open(f"file://{os.path.abspath(output_path)}")


class LauncherApp:
    """The Main GUI Frame."""

    def __init__(self, root):
        self.root = root
        self.root.title("Thesis Data Visualizer")
        self.root.geometry("600x400")
        self.current_dag = None  # This is your "storage" for the selected DAG

        tk.Label(root, text="Data Visualizer", font=("Arial", 14, "bold")).pack(pady=20)

        self.btn_browse = tk.Button(root, text="Choose Data File", command=self.load_and_visualize,
                                    width=20, height=2, bg="#4CAF50", fg="white")
        self.btn_browse.pack(pady=10)

        self.btn_abstract = tk.Button(root, text="Abstract_CAGRES", command=self.abstract_cagres,
                                    width=20, height=2, bg="#4CAF50", fg="white")
        self.btn_abstract.pack(pady=10)

        self.btn_abstract = tk.Button(root, text="Abstract_REDUCEDAG", command=self.abstract_reducedag,
                                      width=20, height=2, bg="#4CAF50", fg="white")
        self.btn_abstract.pack(pady=10)

        self.status = tk.Label(root, text="Waiting for file...", fg="gray")
        self.status.pack(side="bottom", pady=10)

    def load_and_visualize(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Supported", "*.pkl *.dot *.rda *.cg"), ("All files", "*.*")]
        )
        if not file_path: return

        ext = os.path.splitext(file_path)[1].lower()
        try:
            import joblib  # Ensure you have 'pip install joblib'

            if ext == '.pkl':
                try:
                    # Try joblib first since elnino and oracle models often use it
                    loaded_content = joblib.load(file_path)
                    # El Nino specific structure: [Xraw, Yraw, coords]
                    if isinstance(loaded_content, tuple) and len(loaded_content) == 3:
                        Xraw, Yraw, coords = loaded_content
                        data = {
                            'observations': Xraw,
                            'targets': Yraw,
                            'coordinates': coords
                        }
                    else:
                        data = loaded_content
                except Exception:
                    # Fallback to standard pickle for smaller dicts/graphs
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
            # ...
            elif ext == '.dot':
                data = self._parse_dot(file_path)
            elif ext == '.rda':
                data = rda.convert_to_dict(file_path)  #
                # Use FileUtil for organized backup naming
                fu.save_as_pkl(data, file_path, DATA_PATH)
                # New logic for .cg text files
            elif ext == '.cg':
                 data = self._parse_cg(file_path)
            self.current_dag = data

            self.status.config(text=f"Loaded: {os.path.basename(file_path)}", fg="green")
            DataVisualizer.plot_data(data, file_path)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _parse_dot(self, file_path):
        """
        Reads a .dot file from a path and returns a NetworkX DiGraph.
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()

            # Standardize non-standard headers found in some causal tools
            if content.startswith('dag'):
                content = content.replace('dag', 'digraph', 1)

            # Parse the string content into a pydot object
            dot_graphs = pydot.graph_from_dot_data(content)
            if dot_graphs:
                # Convert to NetworkX for your DataVisualizer
                return nx.DiGraph(nx.nx_pydot.from_pydot(dot_graphs[0]))
        except Exception as e:
            print(f"Error reading or parsing .dot file: {e}")
        return None

    def _parse_cg(self, file_path):
        G = nx.DiGraph()
        with open(file_path, 'r') as f:
            lines = f.readlines()

        is_edge_section = False
        for line in lines:
            line = line.strip()
            if not line or line.startswith('<NODES>'):
                continue
            if line.startswith('<EDGES>'):
                is_edge_section = True
                continue

            if is_edge_section:
                # Parses "Z -> X" into ('Z', 'X')
                parent, child = [n.strip() for n in line.split('->')]
                G.add_edge(parent, child)
            else:
                # Adds nodes before edges are defined
                G.add_node(line)
        return G

    def abstract_cagres(self):
        import traceback
        try:
            # Check if the "storage" is empty
            if self.current_dag is None:
                messagebox.showwarning("Logic Error", "No DAG loaded! Please load a file first.")
                return
            abstracted_dag = cg(self.current_dag, size(self.current_dag.nodes)/2, None, 0)
            abstract_path = DATA_PATH = '/home/taylanozgur/BackUp/taylanozgur/METU/CogS/Thesis/MyProjects/DataVisualizer/data/' + '/cagres/abstracted'
            DataVisualizer.plot_data(abstracted_dag, abstract_path)
            # self.display_comparison(self.current_dag, abstracted_dag)

        except Exception as e:
            # This will print the FULL error path to your console
            print("\n--- DEBUG ERROR ---")
            traceback.print_exc()

            # This will show a popup so you know exactly what failed
            messagebox.showerror("Abstraction Error", f"Failed to run CaGreS:\n{e}")
    def abstract_reducedag(self):
        import traceback
        try:
            # Check if the "storage" is empty
            if self.current_dag is None:
                messagebox.showwarning("Logic Error", "No DAG loaded! Please load a file first.")
                return
            if isinstance(self.current_dag, dict):
                self.current_dag = next(iter(self.current_dag.values()))
            abstracted_dag = dr(self.current_dag, exposure="A", outcome="Y").reduce_dag()
            abstract_path = DATA_PATH = '/home/taylanozgur/BackUp/taylanozgur/METU/CogS/Thesis/MyProjects/DataVisualizer/data/' + '/reducedag/abstracted'
            DataVisualizer.plot_data(abstracted_dag, abstract_path)
            self.display_comparison(self.current_dag, abstracted_dag)

        except Exception as e:
            # This will print the FULL error path to your console
            print("\n--- DEBUG ERROR ---")
            traceback.print_exc()

            # This will show a popup so you know exactly what failed
            messagebox.showerror("Abstraction Error", f"Failed to run CaGreS:\n{e}")
    def display_comparison(self, original_dag, abstracted_dag):
        # Create a figure with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # 1. Plot Original on the left
        pos1 = nx.spring_layout(original_dag, seed=42)
        nx.draw(original_dag, pos1, ax=ax1, with_labels=True,
                node_color='lightblue', node_size=800, arrowsize=20)
        ax1.set_title("Original Causal Graph")

        # 2. Plot Abstracted on the right
        pos2 = nx.spring_layout(abstracted_dag, seed=42)
        nx.draw(abstracted_dag, pos2, ax=ax2, with_labels=True,
                node_color='lightgreen', node_size=800, arrowsize=20)
        ax2.set_title("Abstracted Model (CaGreS)")

        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    root = tk.Tk()
    app = LauncherApp(root)
    root.mainloop()