import pyreadr
import pickle
import os
import networkx as nx
import pydot
import pandas as pd

from file_utils import FileUtil

class RdaConverter:
    """Handles the translation between R Data formats and Python formats."""

    @staticmethod
    def convert_to_dict(file_path):
        """
        Reads RDA and detects if objects are data tables or
        nested DOT/DAG strings.
        """
        try:
            result = pyreadr.read_r(file_path)
            processed_data = {}

            for key, content in result.items():
                raw_val = None

                # Attempt to extract a string if it's a list or a DataFrame
                if isinstance(content, list) and len(content) > 0:
                    raw_val = str(content[0]).strip()
                elif isinstance(content, pd.DataFrame) and not content.empty:
                    # Common pyreadr behavior for R character vectors
                    raw_val = str(content.iloc[0, 0]).strip()

                # Centralized check: If it looks like a graph, parse it
                if raw_val and ('dag {' in raw_val or 'digraph {' in raw_val):
                    processed_data[key] = RdaConverter._parse_dot_string(raw_val)
                else:
                    # Default: Keep as DataFrame (Sensor data/Tables)
                    processed_data[key] = content

            return processed_data
        except Exception as e:
            raise Exception(f"RDA Conversion Error: {e}")

    @staticmethod
    def _parse_dot_string(dot_string):
        """Helper to turn a raw DOT/DAG string into a NetworkX DiGraph."""
        # 1. Clean up the string (remove extra quotes/newlines common in R lists)
        clean_str = dot_string.strip().strip("'").strip('"')

        # 2. Standardize the header for Graphviz compatibility
        if clean_str.startswith('dag {'):
            clean_str = clean_str.replace('dag {', 'digraph {', 1)

        try:
            # 3. Parse using pydot
            dot_graphs = pydot.graph_from_dot_data(clean_str)
            if dot_graphs:
                # Convert to NetworkX
                nx_graph = nx.nx_pydot.from_pydot(dot_graphs[0])
                # Ensure it's a DiGraph
                return nx.DiGraph(nx_graph)
        except Exception as e:
            print(f"Pydot parsing failed: {e}")

        return dot_string # Fallback if parsing fails

