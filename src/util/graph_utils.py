import os
import re
import xml.etree.ElementTree as ET

import networkx as nx
import pydot


class GraphUtils:
    # ... [previous methods: from_r_formula, get_info] ...

    data_path = '/home/taylanozgur/BackUp/taylanozgur/METU/CogS/Thesis/MyProjects/DataVisualizer/data/'

    @staticmethod
    def save_graph(G, path, format="dot"):
        """
        Saves the graph to a specified path.

        Args:
            G (nx.DiGraph): The graph to save.
            path (str): File path (e.g., "output/my_dag.graphml").
            format (str): "graphml", "dot", or "adjlist".
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        fmt = format.lower()
        try:
            if fmt == "graphml":
                # Best for preserving node attributes and research data
                nx.write_graphml(G, path)
            elif fmt == "dot":
                # Best for Dagitty/R compatibility (requires pydot or pygraphviz)
                from networkx.drawing.nx_pydot import write_dot
                write_dot(G, path)
            elif fmt == "adjlist":
                # Simple text format
                nx.write_adjlist(G, path)
            else:
                raise ValueError(f"Unsupported format: {format}")

            print(f"Graph successfully saved to {path} in {fmt} format.")

        except ImportError:
            if fmt == "dot":
                print("Error: Saving to DOT requires 'pydot'. Run: pip install pydot")
        except Exception as e:
            print(f"An error occurred while saving: {e}")

    @staticmethod
    def from_r_formula(formula_string):
        """
        Parses an R-style graph.formula string into a NetworkX DiGraph.

        Args:
            formula_string (str): The raw string from R, e.g., "U_1 -+ X_1, U_1 -+ R_1"

        Returns:
            nx.DiGraph: A directed acyclic graph object.
        """
        # 1. Clean the string: remove newlines, R function wrappers, and extra spaces
        clean_str = re.sub(r'graph\.formula\(|\)', '', formula_string)
        clean_str = clean_str.replace('\n', ' ').replace('\t', ' ')

        # 2. Split by comma to get individual edge definitions
        parts = [p.strip() for p in clean_str.split(',') if p.strip()]

        edges = []
        for part in parts:
            # Handle R's directed edge notation '-+'
            if '-+' in part:
                u, v = part.split('-+')
                edges.append((u.strip(), v.strip()))
            # Handle standard directed notation '->' just in case
            elif '->' in part:
                u, v = part.split('->')
                edges.append((u.strip(), v.strip()))

        # 3. Build and return the graph
        G = nx.DiGraph()
        G.add_edges_from(edges)
        return G

    @staticmethod
    def from_r_formula(formula_string, master_nodes=None):
        """
        Parses an R-style graph.formula string into a NetworkX DiGraph.
        Now supports explicit node ordering for R-parity.
        """
        # 1. Clean the string
        clean_str = re.sub(r'graph\.formula\(|\)', '', formula_string)
        clean_str = clean_str.replace('\n', ' ').replace('\t', ' ')

        # 2. Extract edges
        parts = [p.strip() for p in clean_str.split(',') if p.strip()]
        edges = []
        for part in parts:
            if '-+' in part:
                u, v = part.split('-+')
                edges.append((u.strip(), v.strip()))
            elif '->' in part:
                u, v = part.split('->')
                edges.append((u.strip(), v.strip()))

        G = nx.DiGraph()

        # --- CRITICAL FIX FOR YOUR THESIS ---
        if master_nodes:
            # We add the nodes FIRST in the exact R-order.
            # This fixes the internal hashing sequence in NetworkX.
            G.add_nodes_from(master_nodes)

        G.add_edges_from(edges)

        # 3. Final alignment: Ensure adjacency lists are sorted by the master_nodes
        if master_nodes:
            G = GraphUtils.force_r_parity(G, master_nodes)

        return G



    @staticmethod
    def load_from_cg(file_path):
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

    @staticmethod
    def load_from_dot(file_path):
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

    @staticmethod
    def load_from_graphml(file_path):
        # 1. Parse the XML file directly
        tree = ET.parse(file_path)
        root = tree.getroot()
        ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}

        # 2. Extract mapping and nodes in one pass
        # We store them in a list first to ensure we keep the XML document order
        raw_nodes = []
        for node_tag in root.findall('.//g:node', ns):
            node_id = node_tag.get('id')
            # Extract the 'v_name' (e.g., U_1)
            data_tag = node_tag.find('./g:data[@key="v_name"]', ns)
            name = data_tag.text if data_tag is not None else node_id
            raw_nodes.append({'id': node_id, 'name': name})

        # 3. Sort nodes numerically (n0, n1, n2... n12)
        # This is the "Gold Standard" order from R
        raw_nodes.sort(key=lambda x: int(re.search(r'\d+', x['id']).group()))
        master_nodes = [n['name'] for n in raw_nodes]
        id_to_name = {n['id']: n['name'] for n in raw_nodes}

        # 4. Initialize a fresh NetworkX DiGraph
        # We add nodes FIRST to set the internal index
        G = nx.DiGraph()
        G.add_nodes_from(master_nodes)

        # 5. Extract and add edges in the exact order they appear in the XML
        # We also sort these by the source node's numeric index to be safe
        edges_to_add = []
        for edge_tag in root.findall('.//g:edge', ns):
            source_id = edge_tag.get('source')
            target_id = edge_tag.get('target')
            edges_to_add.append((source_id, target_id))

        for s_id, t_id in edges_to_add:
            G.add_edge(id_to_name[s_id], id_to_name[t_id])

        return G

    @staticmethod
    def force_r_parity(graph, master_nodes):
        node_to_idx = {node: i for i, node in enumerate(master_nodes)}
        for u in list(graph.nodes()):
            # Sort neighbors by their original R-index (n0 < n1 < n2)
            sorted_neighbors = sorted(graph[u], key=lambda n: node_to_idx.get(n, 999))
            graph.adj[u] = {v: graph[u][v] for v in sorted_neighbors}
        return graph
    @staticmethod
    def get_info(G):
        """Prints basic DAG information for verification."""
        is_dag = nx.is_directed_acyclic_graph(G)
        print(f"Nodes: {G.number_of_nodes()}")
        print(f"Edges: {G.number_of_edges()}")
        print(f"Is Directed Acyclic Graph: {is_dag}")

    @staticmethod
    def to_digraph_string(G: nx.DiGraph) -> str:
        edges_str_list = []
        for u, v in G.edges():
            edges_str_list.append(f"{u} -> {v}")

        edges_str = "; ".join(edges_str_list)
        return f"digraph {{{edges_str};}}"
    @staticmethod
    def get_dag_metadata(G):
        """
        Given a NetworkX DiGraph, returns the topological order
        and the adjacency list in a dictionary format.
        """
        # 1. Generate 'order' (Topological Sort)
        # We use lexicographical_topological_sort to ensure the order
        # is deterministic and matches the 'sorted' requirement in your code.
        order = list(nx.lexicographical_topological_sort(G))

        # 2. Generate 'adj' (Adjacency List)
        # We only include nodes that actually have successors to match
        # the format: {node: [children]}
        adj = {n: list(G.successors(n)) for n in G.nodes() if list(G.successors(n))}

        return order, adj




class GraphMapper:
    def __init__(self, dag):
        # 1. NetworkX uses .nodes() instead of .keys()
        self.labels = sorted(list(dag.nodes()))
        self.dag = dag
        # 2. Create the Bi-Map
        self.label_to_id = {label: i for i, label in enumerate(self.labels)}
        self.id_to_label = {i: label for i, label in enumerate(self.labels)}

    def get_indexed_data(self):
        """Returns the order (list of ints) and adj (dict of list of ints)."""
        # Numerical Order (topological sort of IDs)
        # We sort the labels first to ensure lexicographical consistency
        order = [self.label_to_id[node] for node in nx.lexicographical_topological_sort(self.dag)]

        # Numerical Adjacency List
        adj = {}
        for node in self.dag.nodes():
            parent_id = self.label_to_id[node]
            # Get successors from the DiGraph and map them to IDs
            adj[parent_id] = [self.label_to_id[child] for child in self.dag.successors(node)]

        return order, adj

    def relabel_abstracted_dag(self, abstracted_dag):
        """Converts the tuples of IDs back to tuples of Labels."""
        new_dag = nx.DiGraph()
        for node_tuple in abstracted_dag.nodes():
            # Translate each ID in the tuple back to its label and sort for consistency
            new_node = tuple(sorted([self.id_to_label[i] for i in node_tuple]))
            new_dag.add_node(new_node)

        for u, v in abstracted_dag.edges():
            new_u = tuple(sorted([self.id_to_label[i] for i in u]))
            new_v = tuple(sorted([self.id_to_label[i] for i in v]))
            new_dag.add_edge(new_u, new_v)

        return new_dag