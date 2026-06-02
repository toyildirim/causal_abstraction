import networkx as nx
import pandas as pd


class GraphReport:
    """
    Generic reporting utility for NetworkX graphs/DAGs.

    Works with:
        - a single nx.Graph / nx.DiGraph
        - a list of graphs
        - a dict of graphs, e.g. {"original": G, "coarsened_1": G1}
    """

    def __init__(self, graphs):
        self.graphs = self._normalize_graph_input(graphs)

    def _normalize_graph_input(self, graphs):
        """
        Converts input into a dictionary:
            {graph_name: graph}
        """
        if isinstance(graphs, nx.Graph):
            return {"graph_0": graphs}

        if isinstance(graphs, list):
            return {f"graph_{i}": g for i, g in enumerate(graphs)}

        if isinstance(graphs, dict):
            return graphs

        raise TypeError(
            "graphs must be a NetworkX graph, a list of graphs, or a dictionary of graphs."
        )

    def summarize_graph(self, G):
        """
        Returns a dictionary of size and structure information for one graph.
        """

        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        is_directed = G.is_directed()

        if n_nodes > 1:
            density = nx.density(G)
        else:
            density = 0.0

        degrees = dict(G.degree())
        in_degrees = dict(G.in_degree()) if is_directed else {}
        out_degrees = dict(G.out_degree()) if is_directed else {}

        if degrees:
            avg_degree = sum(degrees.values()) / n_nodes
            max_degree = max(degrees.values())
            min_degree = min(degrees.values())
        else:
            avg_degree = 0
            max_degree = 0
            min_degree = 0

        if is_directed:
            if in_degrees:
                avg_in_degree = sum(in_degrees.values()) / n_nodes
                avg_out_degree = sum(out_degrees.values()) / n_nodes
                max_in_degree = max(in_degrees.values())
                max_out_degree = max(out_degrees.values())
            else:
                avg_in_degree = 0
                avg_out_degree = 0
                max_in_degree = 0
                max_out_degree = 0

            weak_components = list(nx.weakly_connected_components(G))
            strong_components = list(nx.strongly_connected_components(G))

            n_weak_components = len(weak_components)
            n_strong_components = len(strong_components)

            largest_weak_component_size = (
                max(len(c) for c in weak_components) if weak_components else 0
            )

            largest_strong_component_size = (
                max(len(c) for c in strong_components) if strong_components else 0
            )

            is_dag = nx.is_directed_acyclic_graph(G)

        else:
            avg_in_degree = None
            avg_out_degree = None
            max_in_degree = None
            max_out_degree = None

            components = list(nx.connected_components(G))
            n_weak_components = len(components)
            n_strong_components = None

            largest_weak_component_size = (
                max(len(c) for c in components) if components else 0
            )

            largest_strong_component_size = None
            is_dag = None

        return {

            "nodes": n_nodes,
            "edges": n_edges,
            "density": density,
            "is_directed": is_directed,
            "is_dag": is_dag,
            "avg_degree": avg_degree,
            "min_degree": min_degree,
            "max_degree": max_degree,
            "avg_in_degree": avg_in_degree,
            "avg_out_degree": avg_out_degree,
            "max_in_degree": max_in_degree,
            "max_out_degree": max_out_degree,
            "weak_components": n_weak_components,
            "strong_components": n_strong_components,
            "largest_weak_component_size": largest_weak_component_size,
            "largest_strong_component_size": largest_strong_component_size,
        }

    def report(self):
        """
        Returns a pandas DataFrame summarizing all graphs.
        """
        rows = []

        for name, G in self.graphs.items():
            summary = self.summarize_graph(G)
            summary["graph"] = name
            rows.append(summary)

        df = pd.DataFrame(rows)

        cols = ["graph"] + [c for c in df.columns if c != "graph"]
        return df[cols]

    def print_report(self):
        """
        Prints the report as a readable table.
        """
        df = self.report()
        # print(df.to_string(index=False))

    def node_size_report(self):
        """
        Useful for coarsened graphs.

        Reports the size of each abstract/coarsened node.
        For example:
            ('A', 'B', 'C') has size 3
            ('D',) has size 1
        """
        rows = []

        for graph_name, G in self.graphs.items():
            for node in G.nodes():
                if isinstance(node, tuple):
                    node_size = len(node)
                    node_label = str(node)
                elif isinstance(node, set):
                    node_size = len(node)
                    node_label = str(node)
                else:
                    node_size = 1
                    node_label = str(node)

                rows.append({
                    "graph": graph_name,
                    "node": node_label,
                    "node_size": node_size,
                })

        return pd.DataFrame(rows)

    def abstraction_level_report(self):
        """
        Reports how coarse each graph is based on:
            - number of abstract nodes
            - average abstract node size
            - maximum abstract node size
        """
        rows = []

        for graph_name, G in self.graphs.items():
            node_sizes = []

            for node in G.nodes():
                if isinstance(node, tuple):
                    node_sizes.append(len(node))
                elif isinstance(node, set):
                    node_sizes.append(len(node))
                else:
                    node_sizes.append(1)

            if node_sizes:
                avg_abstract_node_size = sum(node_sizes) / len(node_sizes)
                max_abstract_node_size = max(node_sizes)
                min_abstract_node_size = min(node_sizes)
            else:
                avg_abstract_node_size = 0
                max_abstract_node_size = 0
                min_abstract_node_size = 0

            rows.append({
                "graph": graph_name,
                "abstract_nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "avg_abstract_node_size": avg_abstract_node_size,
                "min_abstract_node_size": min_abstract_node_size,
                "max_abstract_node_size": max_abstract_node_size,
            })

        return pd.DataFrame(rows)