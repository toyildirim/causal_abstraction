import os
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import pydot
from PIL import report
from numpy.ma.core import size

from abstraction_methods.cagres.cagres import CaGreS as cg
from abstraction_methods.reducedag.reduce_dag import DAGReducer as dr
from abstraction_methods.transitcluster.transit_cluster import TransitCluster as tc
from util.file_utils import FileUtil as fu
from util.rda_converter import RdaConverter as rda
from util.graph_utils import GraphUtils, GraphMapper
from util.report import shd_reporter
from util.report.dag_reporter import GraphReport
import util.report.ri_reporter as ri_ari_reporter
import util.report.shd_reporter
import util.report.comparison_reporter as cr
# from abstraction_methods.repare import repare_known_dag_model as repare_dag
from abstraction_methods.repare import repare as repare_dag
# Set the backend for Linux/PyCharm compatibility
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from pyvis.network import Network
import webbrowser

import ml_3level_dag_creator as ml_dg_creator
from util.evaluation_utils import EvaluationUtils as eu

import random
import pandas as pd
from datetime import datetime

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt

# Constants
DATA_PATH = f"/home/taylanozgur/BackUp/taylanozgur/METU/CogS/Thesis/MyProjects/DataVisualizer/abstraction_results/dags"
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
            # DataVisualizer.plot_dag_left_to_right(data, base_name)
            # DataVisualizer.plot_dag_static(data, base_name)

        # 2. Dictionaries (RDA output or sensor groups)
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
        # plt.figure(title, figsize=(10, 7))
        # pos = nx.spring_layout(graph_obj, seed=42)
        # nx.draw(graph_obj, pos, with_labels=True, node_color='skyblue',
        #         node_size=1500, arrowsize=20, font_weight='bold')
        # plt.title(title)
        # plt.show(block=False)

        # --- BROWSER VIEW (Pyvis) ---
        net = Network(height='750px', width='100%', notebook=False, directed=True)

        # Helper function to convert node to string ID
        def get_node_id(node):
            """Convert a node (tuple, frozenset, etc.) to a unique string ID."""
            if isinstance(node, (set, frozenset)):
                return "|".join(map(str, sorted(node)))
            else:
                return str(node)
        # Color coding for Causal Roles
        for node, attr in graph_obj.nodes(data=True):
            # Color coding for Causal Roles
            # 1. Convert the frozenset/node to a unique string ID
            # This formats frozenset({1, 2}) into "1|2" for a cleaner look
            # node_id = "|".join(map(str, sorted(node))) if isinstance(node, (set, frozenset)) else str(node)
            node_id = get_node_id(node)
            is_exposure = 'exposure' in str(attr) or node_id == 'exposure' or node =='exposure'
            is_outcome =  'outcome' in str(attr)  or node_id == 'outcome' or node == 'outcome'
            if is_outcome:
                color = 'red'
            elif is_exposure:
                color = 'green'
            else:
                color = 'skyblue'

            label = f"{'EXPOSURE' if is_exposure else 'OUTCOME' if is_outcome else ''}: {node}"
            net.add_node(node_id, label=label, color=color)

        for u, v in graph_obj.edges():
            # net.add_edge(u, v)
            u_id = get_node_id(u)
            v_id = get_node_id(v)
            net.add_edge(u_id, v_id)
        net.toggle_physics(True)

        # Ensure HTML directory exists
        os.makedirs(HTML_DIR, exist_ok=True)
        safe_name = "".join([c if c.isalnum() else "_" for c in title])
        output_path = os.path.join(HTML_DIR, f"graph_{safe_name}.html")
        net.write_html(output_path)
        # webbrowser.open(f"file://{os.path.abspath(output_path)}")

    @staticmethod
    def plot_dag_static(G, path="dag.png"):
        path = path + ".png"
        plt.figure(figsize=(18, 8))

        pos = graphviz_layout(G, prog="dot")  # hierarchical DAG layout

        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=2500,
            font_size=8,
            arrows=True,
            arrowsize=15
        )

        plt.tight_layout()
        plt.margins(0.2)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_dag_left_to_right(G, path="dag.png"):
        A = nx.nx_agraph.to_agraph(G)
        A.graph_attr.update(rankdir="LR")  # left to right
        A.graph_attr.update(splines="true")
        A.node_attr.update(shape="ellipse", fontsize="10")

        A.layout(prog="dot")
        A.draw(f"{path}LR.png")



class LauncherApp:
    """The Main GUI Frame."""

    def __init__(self, root):
        self.root = root
        self.root.title("Thesis Data Visualfrom_r_formulaizer")
        self.root.geometry("600x800")
        self.current_dag = None  # This is your "storage" for the selected DAG
        self.current_abs_dag = None
        self.nodes = None  # To store master nodes if needed for abstraction methods
        tk.Label(root, text="Data Visualizer", font=("Arial", 14, "bold")).pack(pady=20)

        self.btn_browse = tk.Button(root, text="Choose Data File", command=self.load_and_visualize,
                                    width=20, height=2, bg="#4CAF50", fg="white")
        self.btn_browse.pack(pady=10)

        self.btn_abstract_cagres = tk.Button(root, text="Abstract_CAGRES", command=self.abstract_cagres,
                                    width=20, height=2, bg="#4CAF50", fg="white")
        self.btn_abstract_cagres.pack(pady=10)

        self.btn_abstract_reducedag = tk.Button(root, text="Abstract_REDUCEDAG", command=self.abstract_reducedag,
                                      width=20, height=2, bg="#4CAF50", fg="white")
        self.btn_abstract_reducedag.pack(pady=10)

        self.btn_abstract_transitcluster = tk.Button(root, text="Abstract_TRANSITCLUSTER", command=self.abstract_transitcluster,
                                      width=20, height=2, bg="#4CAF50", fg="white")
        self.btn_abstract_transitcluster.pack(pady=10)

        self.btn_abstract_repare = tk.Button(root, text="Abstract_REPARE",
                                                     command=self.abstract_repare,
                                                     width=20, height=2, bg="#4CAF50", fg="white")
        self.btn_abstract_repare.pack(pady=10)

        self.btn_abstract_all_L2 = tk.Button(root, text="DAG Abstraction OverAll For L2-Teen",
                                             command=self.configure_abstraction_for_L2,
                                             width=20, height=2, bg="#4CAF50", fg="white")
        self.btn_abstract_all_L2.pack(pady=10)

        self.btn_abstract_all_L1 = tk.Button(root, text="DAG Abstraction OverAll For L1-Child",
                                             command=self.configure_abstraction_for_L1,
                                             width=20, height=2, bg="#4CAF50", fg="white")
        self.btn_abstract_all_L1.pack(pady=10)

        self.status = tk.Label(root, text="Waiting for file...", fg="gray")
        self.status.pack(side="bottom", pady=10)

        self.oracle_abstracted_L1 = ml_dg_creator.create_child_level_dag_GPT54()
        self.oracle_abstracted_L2 = ml_dg_creator.create_teen_level_dag_GPT54()
        self.oracle_dag = ml_dg_creator.create_grad_level_dag_GPT54()
        self.time = datetime.now()
        self.tolerance = 1
        self.target_level = "L2"
        self.report_dir = f"/home/taylanozgur/BackUp/taylanozgur/METU/CogS/Thesis/MyProjects/DataVisualizer/abstraction_results"
        os.makedirs(self.report_dir,exist_ok=True)

    def load_and_visualize(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Supported", "*.pkl *.dot *.rda *.cg *.graphml"), ("All files", "*.*")]
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
                data = GraphUtils.load_from_dot(file_path)
            elif ext == '.graphml':
                data = GraphUtils.load_from_graphml(file_path)
            elif ext == '.rda':
                data = rda.convert_to_dict(file_path)  #
                # Use FileUtil for organized backup naming
                fu.save_as_pkl(data, file_path, DATA_PATH)
                # New logic for .cg text files
            elif ext == '.cg':
                 data = GraphUtils.load_from_cg(file_path)
            self.current_dag = data
            self.nodes= data.nodes() if isinstance(data, (nx.Graph, nx.DiGraph)) else None

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

    def abstract_cagres_ex(self):
        import traceback
        try:
            # Check if the "storage" is empty
            if self.current_dag is None:
                messagebox.showwarning("Logic Error", "No DAG loaded! Please load a file first.")
                return
            # abstracted_dag = cg(self.current_dag, size(self.current_dag.nodes)/2, None, 0)
            abstracted_dag_teen = cg(self.current_dag, size(self.oracle_abstracted_L2.nodes), None, 0)
            # abstracted_dag_child = cg(self.current_dag, size(self.ml_child_dag.nodes), None, 0)


            abstract_path = DATA_PATH + '/CaGreS/abstracted_dags'
            DataVisualizer.plot_data(abstracted_dag_teen, f"{abstract_path}_teen")
            # DataVisualizer.plot_data(abstracted_dag_child, f"{abstract_path}_child")
            # self.display_comparison(self.current_dag, abstracted_dag)
            abstracted_dags = []
            # abstracted_dags.append(abstracted_dag)
            abstracted_dags.append(abstracted_dag_teen)
            self.report_dag(abstracted_dags,"cagres")
            self.compare_dags(abstracted_dags, self.oracle_abstracted_L2)

            # abstracted_dags.remove(abstracted_dag_teen)
            # abstracted_dags.append(abstracted_dag_child)
            # self.report_dag(abstracted_dags,"cagres")
            # self.compare_dags(abstracted_dags, self.ml_child_dag)

        except Exception as e:
            # This will print the FULL error path to your console
            print("\n--- DEBUG ERROR ---")
            traceback.print_exc()

            # This will show a popup so you know exactly what failed
            messagebox.showerror("Abstraction Error", f"Failed to run CaGreS:\n{e}")

    def abstract_cagres(self):
        n_runs = 10
        # for target_level, target_k, oracle_dag in [
        #     ("L2", size(self.ml_teen_dag.nodes), self.ml_teen_dag),
        #     ("L1", size(self.ml_child_dag), self.ml_child_dag),
        # ]:

        import traceback
        try:
            # Check if the "storage" is empty
            if self.current_dag is None:
                messagebox.showwarning("Logic Error", "No DAG loaded! Please load a file first.")
                return
            # abstracted_dag = cg(self.current_dag, size(self.current_dag.nodes)/2, None, 0)
            # abstracted_dag_teen = cg(self.current_dag, size(self.ml_teen_dag.nodes), None, 0)
            # abstracted_dag_child = cg(self.current_dag, size(self.ml_child_dag.nodes), None, 0)

            abstract_path = DATA_PATH + '/CaGreS/abstracted_dags'

            # DataVisualizer.plot_data(abstracted_dag_child, f"{abstract_path}_child")
            # self.display_comparison(self.current_dag, abstracted_dag)
            abstracted_dags = []
            # abstracted_dags.append(abstracted_dag)
            for seed in range(n_runs):
                random.seed(seed)

                candidate_dag = cg(self.current_dag, size(self.current_abs_dag.nodes), None, 0)
                abstracted_dags.append(candidate_dag)
                DataVisualizer.plot_data(candidate_dag, f"{abstract_path}/cagres_{self.target_level}_{seed+1}_{self.time}")
            self.report_dag(abstracted_dags, "CaGreS",target_level = self.target_level)
            self.compare_dags(dags=abstracted_dags, oracle_abstracted_dag=self.current_abs_dag, abstraction_method="CaGreS", target_level =self.target_level)

            # abstracted_dags.remove(abstracted_dag_teen)
            # abstracted_dags.append(abstracted_dag_child)
            # self.report_dag(abstracted_dags,"cagres")
            # self.compare_dags(abstracted_dags, self.ml_child_dag)

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
            abstracted_dags = []
            abstracted_dag = dr(self.current_dag, exposure=None, outcome=None).reduce_dag()
            abstracted_dags.append(abstracted_dag)

            abstract_path = DATA_PATH + '/ReduceDAG/abstracted_dags'
            DataVisualizer.plot_data(abstracted_dag, f"{abstract_path}/reducedag_{self.target_level}_{self.time}")
            # self.display_comparison(self.current_dag, abstracted_dag)
            self.report_dag(abstracted_dags,"ReduceDAG",target_level = self.target_level)
            self.compare_dags(dags=abstracted_dags, oracle_abstracted_dag=self.current_abs_dag, abstraction_method="ReduceDAG", target_level =self.target_level)
            # self.compare_dags(dags=abstracted_dags, oracle_abstracted_dag=self.ml_child_dag, node_removed=True)

        except Exception as e:
            # This will print the FULL error path to your console
            print("\n--- DEBUG ERROR ---")
            traceback.print_exc()

            # This will show a popup so you know exactly what failed
            messagebox.showerror("Abstraction Error", f"Failed to run ReduceDAG:\n{e}")

    def abstract_transitcluster(self):
        import traceback
        try:
            # Check if the "storage" is empty
            if self.current_dag is None:
                messagebox.showwarning("Logic Error", "No DAG loaded! Please load a file first.")
                return
            if isinstance(self.current_dag, dict):
                self.current_dag = next(iter(self.current_dag.values()))

            # Assuming tc_engine is an instance of TransitCluster and G is your networkx DAG
            tc_engine = tc(self.current_dag)
            # 1. Find the clusters (Algorithm 1)
            t_components = tc_engine.find_transit_components(singletons=False)
            t_clusters = tc_engine.find_transit_clusters(self.current_dag,t_components, self.nodes)
            # 2. Iterate and abstract
            # for cluster in t_clusters:
            #     print(f"Processing Transit Cluster: {cluster['vertices']}")
            #     # Generate the abstracted/contracted graph
            #     abstracted_dag = tc_engine.grouped_graph(grouping=cluster)
            #     abstract_path = DATA_PATH  + '/transitcluster/abstracted'
            #     DataVisualizer.plot_data(abstracted_dag, abstract_path)
            #     # self.display_comparison(self.current_dag, abstracted_dag)
            grouped_dags = []
           # 3. THE LOOP: (Matches R: for (tc in tcluster))
           # 'i' increments automatically (0, 1, 2...).

            for i, cluster in enumerate(t_clusters):
                # --- THE 'R-PARITY' ORDERING STEP ---
                # We filter using 'self.nodes' (The Master Ruler).
                # This guarantees U_1 is first and Y_1 is last, matching your R image.
                cluster['vertices'] = [n for n in self.current_dag.nodes() if n in cluster['vertices']]
                print(f"Iteration {i + 1}: Processing Cluster {cluster['vertices']}")

                # 4. ABSTRACTION: (Matches R: g_abstract <- grouped_graph(tc))
                # This merges the 13 nodes into that single 'Blue Bubble' (Macro-Node).
                abstracted_dag = tc_engine.grouped_graph(cluster,self.current_dag)
                abstract_path = f"{DATA_PATH}/transitcluster/abstracted_dags/cluster_{i + 1}"
                # if '' not in abstracted_dag and (size(abstracted_dag.nodes) == size(self.ml_teen_dag.nodes)): #or size(abstracted_dag.nodes) == size(self.ml_child_dag.nodes)):
                # if '' not in abstracted_dag and (size(abstracted_dag.nodes) == size(
                #         self.ml_teen_dag.nodes)):
                if '' not in abstracted_dag and (abs(size(abstracted_dag.nodes) - size(self.current_abs_dag.nodes)) <= self.tolerance):
                    grouped_dags.append(abstracted_dag)
                    DataVisualizer.plot_data(abstracted_dag, f"{abstract_path}/tc_{self.target_level}_cluster_{i+1}_{self.time}")
                # 5. PLOT: (Matches R: plot(g_abstract))
                # Use i + 1 directly in the path to avoid overwriting files.

                # if i == 77:

            # self.report_dag(grouped_dags,"transitcluster")

            self.report_dag(grouped_dags, "TransitClusters", target_level = self.target_level)
            self.compare_dags(dags=grouped_dags, oracle_abstracted_dag=self.current_abs_dag, abstraction_method="TransitClusters", target_level = self.target_level)
            # self.compare_dags(dags=grouped_dags, oracle_abstracted_dag=self.ml_child_dag)
            # for i, g in enumerate(grouped_dags):
            #     # print(f"\nCoarsening step {i}")
            #     # print("Nodes:", list(g.nodes()))
            #     # print("Edges:", list(g.edges()))
            #     #
            #     # shd_result = shd_reporter.shd_if_same_partition(
            #     #     oracle_abs_dag=self.ml_teen_dag,
            #     #     candidate_abs_dag=g
            #     # )
            #     shd_result = cr.calculate_shd(
            #         oracle_abs_dag=self.ml_teen_dag,
            #         candidate_abs_dag=g)
            #
            #     jaccard_result = cr.calculate_jaccard(oracle_abs_dag=self.ml_teen_dag, candidate_abs_dag=g)
            #
            #     ri_ari_result = cr.calculate_ri_ari(
            #         oracle_dag=self.ml_teen_dag,
            #         candidate_dag=g
            #     )
            #
            #     print(f"\nCluster {i}")
            #     print("SHD:", shd_result["shd"])
            #     print("Jaccard:", jaccard_result["JS"])
            #     print("RI:", ri_ari_result["RI"])
            #     print("ARI:", ri_ari_result["ARI"])

        except Exception as e:
            # This will print the FULL error path to your console
            print("\n--- DEBUG ERROR ---")
            traceback.print_exc()

            # This will show a popup so you know exactly what failed
            messagebox.showerror("Abstraction Error", f"Failed to run Transit Cluster:\n{e}")

    def abstract_repare(self):
        import traceback
        try:
            # Check if the "storage" is empty
            if self.current_dag is None:
                messagebox.showwarning("Logic Error", "No DAG loaded! Please load a file first.")
                return
            if isinstance(self.current_dag, dict):
                self.current_dag = next(iter(self.current_dag.values()))

            # order, adj = GraphUtils.get_dag_metadata(self.current_dag)
            # repare_model =  repare_dag.PartitionDagModelOracle()
            # abstracted_dag = repare_model.fit(order, adj)
            # 1. Initialize with the DiGraph object
            # mapper = GraphMapper(self.current_dag)
            mapper = GraphMapper(self.current_dag)
            # 2. Get the numerical versions
            order, adj = mapper.get_indexed_data()
            # order = [3, 1, 2, 4]
            # adj = {1: [2], 3: [2], 2: [4]}transit
            # 3. Fit the model
            repare_model = repare_dag.PartitionDagModelOracle()
            abstracted_dag_numeric = repare_model.fit(order, adj)

            # 4. Map back to strings for visualization
            # abstracted_dag = mapper.relabel_abstracted_dag(abstracted_dag_numeric)
            abstract_path = f"{DATA_PATH}/repare/abstracted_dag"
            # DataVisualizer.plot_data(abstracted_dag, abstract_path)
            coarsening_history = repare_model.get_coarsening_history()
            # coarsening_relabeled = [
            #     mapper.relabel_abstracted_dag(g)
            #     for g in coarsening_history
            # ]
            coarsening_relabeled = []
            for i, g in enumerate(coarsening_history):
                if abs(len(g.nodes) - len(self.current_abs_dag.nodes)) <= self.tolerance:
                # if len(g.nodes) == len(self.ml_child_dag.nodes):
                    abstract_coarsening_path = f"{DATA_PATH}/RePaRe/abstracted_dags/coarsening_{i + 1}"
                    g_relabeled = mapper.relabel_abstracted_dag(g)
                    coarsening_relabeled.append(g_relabeled)
                    DataVisualizer.plot_data(g_relabeled, f"{abstract_coarsening_path}/repare_{self.target_level}_coarsening_{i+1}_{self.time}")
            self.report_dag(coarsening_relabeled,"RePaRe", target_level = self.target_level)
            self.compare_dags(dags=coarsening_relabeled, oracle_abstracted_dag=self.current_abs_dag, abstraction_method="RePaRe", target_level = self.target_level)
            # self.compare_dags(dags=coarsening_relabeled, oracle_abstracted_dag=self.ml_child_dag)
            # for i, g in enumerate(coarsening_relabeled):
            #     # print(f"\nCoarsening step {i}")
            #     # print("Nodes:", list(g.nodes()))
            #     # print("Edges:", list(g.edges()))
            #     #
            #     # shd_result = shd_reporter.shd_if_same_partition(
            #     #     oracle_abs_dag=self.ml_teen_dag,
            #     #     candidate_abs_dag=g
            #     # )
            #     shd_result = cr.calculate_shd(date
            #         oracle_abs_dag=self.ml_teen_dag,
            #         candidate_abs_dag=g)
            #
            #     jaccard_result = cr.calculate_jaccard(oracle_abs_dag=self.ml_teen_dag,candidate_abs_dag=g)
            #
            #     ri_ari_result = cr.calculate_ri_ari(
            #         oracle_dag=self.ml_teen_dag,
            #         candidate_dag=g
            #     )
            #
            #     print(f"\nCandidate {i}")
            #     print("SHD:", shd_result["shd"])
            #     print("Jaccard:", jaccard_result["JS"])
            #     print("RI:", ri_ari_result["RI"])
            #     print("ARI:", ri_ari_result["ARI"])
            #
            #     DataVisualizer.plot_data(g, f"{abstract_path}_{i}")

        except Exception as e:
            # This will print the FULL error path to your console
            print("\n--- DEBUG ERROR ---")
            traceback.print_exc()

            # This will show a popup so you know exactly what failed
            messagebox.showerror("Abstraction Error", f"Failed to run Repare:\n{e}")

    def configure_abstraction_for_L2(self):
        # This method will generate the report for L2 (Teen DAG)
        # You can customize the logic to select the appropriate abstracted DAGs for L2
        # For example, you might want to filter abstracted_dags based on their node count or other criteria
        self.current_dag = self.oracle_dag
        self.current_abs_dag = self.oracle_abstracted_L2
        self.tolerance = 0
        self.target_level = "L2"
        self.time = datetime.now()

    def configure_abstraction_for_L1(self):
        self.current_dag = self.oracle_dag
        self.current_abs_dag = self.oracle_abstracted_L1
        self.tolerance = 0
        self.target_level = "L1"
        self.time = datetime.now()

    def report_dag(self, abstracted_dags, abstraction_method, target_level = "L2"):
        reporter = GraphReport(abstracted_dags)
        abstraction_report_path = f"{self.report_dir}/{abstraction_method}"
        os.makedirs(abstraction_report_path, exist_ok=True)
        general_results_path = f"{abstraction_report_path}/{target_level}_{self.time}"
        # report_path_general = report_pat
        # General graph report
        df_general = reporter.report()
        # print(df_general)
        # df_general.to_html(f"{abstraction_method}_{target_level}_df_general.html")
        df_general.to_excel(f"{general_results_path}_df_general.xlsx", index=False)
        df_general.to_latex(f"{general_results_path}_df_general.tex", index=False)

        # Coarsening-specific size report
        df_abstraction = reporter.abstraction_level_report()
        # print(df_abstraction)
        # df_abstraction.to_html(f"{abstraction_method}_{target_level}_df_abstraction_{self.time}.html")
        df_abstraction.to_excel(f"{general_results_path}_df_abstraction.xlsx", index=False)
        df_abstraction.to_latex(f"{general_results_path}_df_abstraction.tex", index=False)

        # Node-level abstract size report
        df_nodes = reporter.node_size_report()
        # print(df_nodes)
        # df_nodes.to_html(f"{abstraction_method}_{target_level}_df_nodes_{self.time}.html")
        df_nodes.to_excel(f"{general_results_path}_df_nodes.xlsx", index=False)
        df_nodes.to_latex(f"{general_results_path}_df_nodes.tex", index=False)

    def compare_dags (self, dags, oracle_abstracted_dag, abstraction_method="CaGreS", target_level = "L2"):
        results = []
        results_dir = f"{self.report_dir}/{abstraction_method}"
        os.makedirs(results_dir, exist_ok=True)
        general_results_path = f"{results_dir}/{target_level}_{self.time}"
        for i, g in enumerate(dags):
            # print(f"\nCoarsening step {i}")
            # print("Nodes:", list(g.nodes()))
            # print("Edges:", list(g.edges()))
            #
            # shd_result = shd_reporter.shd_if_same_partition(
            #     oracle_abs_dag=oracle_abstracted_dag,
            #     candidate_abs_dag=g
            # )
            # shd_result_oracles = cr.calculate_shd(
            #     oracle_abs_dag=self.ml_grad_dag,
            #     candidate_abs_dag=self.ml_teen_dag
            # )
            original_nodes = eu.infer_original_nodes(oracle_abstracted_dag, g)
            adj_oracle = eu.graph_to_adj_matrix(oracle_abstracted_dag, original_nodes, sep='_')
            adj_abstracted = eu.graph_to_adj_matrix(g, original_nodes, sep='_')

            # shd_result = cr.calculate_shd(
            #     oracle_abs_dag=oracle_abstracted_dag,
            #     candidate_abs_dag=g
            # )
            shd_result = cr.calculate_shd_with_adj(adj_oracle, adj_abstracted)

            # jaccard_score_oracles = cr.calculate_jaccard(self.ml_grad_dag, self.ml_teen_dag)
            # jaccard_score = cr.calculate_jaccard(oracle_abstracted_dag, g)
            jaccard_score = cr.calculate_jaccard_with_adj(adj_oracle, adj_abstracted)

            # ci_coverage_oracles = cr.calculate_ci_coverage(oracle_abs_dag=self.ml_grad_dag, candidate_abs_dag=self.ml_teen_dag)
            # ci_coverage = cr.calculate_ci_coverage(oracle_abs_dag=oracle_abstracted_dag,candidate_abs_dag=g)
            ci_coverage = cr.calculate_ci_coverage_with_adj(adj_oracle, adj_abstracted, original_nodes)

            # ri_ari_result = ri_ari_reporter.ri_ari_report(
            #     oracle_dag=oracle_abstracted_dag,
            #     candidate_dag=g
            # )
            # ri_ari_result_oracles = cr.calculate_ri_ari(oracle_dag=self.ml_grad_dag, candidate_dag=g)V            print(f"Abstracted DAG-{i + 1}")
            ri_ari_result = {}
            if abstraction_method != "ReduceDAG":
                ri_ari_result = cr.calculate_ri_ari(oracle_dag=oracle_abstracted_dag, candidate_dag=g)
                # print("RI:", ri_ari_result["RI"])
                # print("ARI:", ri_ari_result["ARI"])
            else:
                ri_ari_result = {
                    "RI": None,
                    "ARI": None
                }


            results.append({
                "Method": abstraction_method,
                "Target Level": target_level,
                "Abstracted DAG": i+1,
                "Candidate Nodes": g.number_of_nodes(),
                "Oracle Nodes": oracle_abstracted_dag.number_of_nodes(),
                "Node Difference": abs(g.number_of_nodes() - oracle_abstracted_dag.number_of_nodes()),
                "Candidate Edges": g.number_of_edges(),
                "Oracle Edges": oracle_abstracted_dag.number_of_edges(),
                "Edge Difference": abs(g.number_of_edges() - oracle_abstracted_dag.number_of_edges()),
                "RI": ri_ari_result.get("RI", np.nan),
                "ARI": ri_ari_result.get("ARI", np.nan),
                "SHD": shd_result["shd"],
                "Jaccard": jaccard_score["JS"],
                "CI-F1": ci_coverage["f1"],
                "CI-Precision": ci_coverage["precision"],
                "CI-Recall": ci_coverage["recall"]
            })
            df = pd.DataFrame(results)
            df = df.round(3)
            # print(df)
            # df.to_html(f"{abstraction_method}_{target_level}_df_results_{self.time}.html")
            # df.to_excel(f"{abstraction_method}_{target_level}_df_results_{self.time}.xlsx", index=False)
            df.to_excel(f"{general_results_path}_df_results.xlsx", index=False)

            summary_table = (
                df.groupby(["Method", "Target Level"])
                .agg(
                    Best_CI_F1=("CI-F1", "max"),
                    Mean_CI_F1=("CI-F1", "mean"),
                    Std_CI_F1=("CI-F1", "std"),
                    Best_ARI=("ARI", "max"),
                    Mean_ARI=("ARI", "mean"),
                    Std_ARI=("ARI", "std"),
                    Best_Jaccard=("Jaccard", "max"),
                    Mean_Jaccard=("Jaccard", "mean"),
                    Best_SHD=("SHD", "min"),
                    Mean_SHD=("SHD", "mean")
                )
                .reset_index()
            )
            # print(summary_table)
            # summary_table = summary_table.round(3)
            # print(summary_table)

            # summary_table.to_csv(f"{abstraction_method}_{target_level}_df_summary_results.csv", index=False)

            # summary_table.to_excel(f"{abstraction_method}_{target_level}_df_summary_results_{self.time}.xlsx", index=False)
            summary_table.to_excel(f"{general_results_path}_df_summary_results.xlsx", index=False)
            # summary_table.to_latex(f"{abstraction_method}_{target_level}_df_summary_results_{self.time}.tex", index=False)
            summary_table.to_latex(f"{general_results_path}_df_summary_results.tex", index=False)

            #Best causal preservation: use CI-F1
            best_by_ci = df.loc[[df['CI-F1'].idxmax()]]
            # best_by_ci.to_excel(f"{abstraction_method}_{target_level}_best_by_ci_{self.time}.xlsx", index=False)
            best_by_ci.to_excel(f"{general_results_path}_best_by_ci.xlsx", index=False)

            # best_by_ci.to_latex(f"{abstraction_method}_{target_level}_best_by_ci_{self.time}.tex", index=False)
            best_by_ci.to_latex(f"{general_results_path}_best_by_ci.tex", index=False)

            #Best grouping similarity: use ARI
            if abstraction_method != "ReduceDAG":
                best_by_ari = df.loc[[df["ARI"].idxmax()]]
                # best_by_ari.to_excel(f"{abstraction_method}_{target_level}_best_by_ari_{self.time}.xlsx", index=False)
                best_by_ari.to_excel(f"{general_results_path}_best_by_ari.xlsx", index=False)

                # best_by_ari.to_latex(f"{abstraction_method}_{target_level}_best_by_ari_{self.time}.xlsx", index=False)
                best_by_ari.to_latex(f"{general_results_path}_best_by_ari.tex", index=False)

            #Best structural similarity: use lowest SHD or highest Jaccard
            best_by_shd = df.loc[[df["SHD"].idxmin()]]
            # best_by_shd.to_excel(f"{abstraction_method}_{target_level}_best_by_shd_{self.time}.xlsx", index=False)
            # best_by_shd.to_latex(f"{abstraction_method}_{target_level}_best_by_shd_{self.time}.tex", index=False)
            best_by_shd.to_excel(f"{general_results_path}_best_by_shd.xlsx", index=False)
            best_by_shd.to_latex(f"{general_results_path}_best_by_shd.tex", index=False)



            best_by_jaccard = df.loc[[df["Jaccard"].idxmax()]]
            # best_by_jaccard.to_excel(f"{abstraction_method}_{target_level}_best_by_jaccard_{self.time}.xlsx", index=False)
            # best_by_jaccard.to_latex(f"{abstraction_method}_{target_level}_best_by_jaccard_{self.time}.tex", index=False)
            best_by_jaccard.to_excel(f"{general_results_path}_best_by_jaccard.xlsx", index=False)
            best_by_jaccard.to_latex(f"{general_results_path}_best_by_jaccard.tex", index=False)

            best_runs = (
                df.sort_values("CI-F1", ascending=False)
                .groupby(["Method", "Target Level"])
                .head(1)
            )
            # print(best_runs)
            # best_runs.to_excel(f"{abstraction_method}_{target_level}_df_best_results_{self.time}.xlsx", index=False)
            # best_runs.to_latex(f"{abstraction_method}_{target_level}_df_best_results_{self.time}.tex", index=False)
            best_runs.to_excel(f"{general_results_path}_best_results.xlsx", index=False)
            best_runs.to_latex(f"{general_results_path}_best_results.tex", index=False)

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
        ax2.set_title("Abstracted Model")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = LauncherApp(root)
    root.mainloop()