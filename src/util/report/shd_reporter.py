import networkx as nx
from dodiscover.metrics import structure_hamming_dist
import numpy as np
from sklearn.metrics import jaccard_score

def node_to_block(node, sep="_"):
    if isinstance(node, tuple):
        return set(node)
    if isinstance(node, set):
        return node
    if isinstance(node, frozenset):
        return set(node)
    if isinstance(node, str) and sep in node:
        return set(node.split(sep))
    return {node}


def block_to_key(block):
    return tuple(sorted(block, key=str))


def normalize_graph_nodes(G, sep="_"):
    mapping = {
        node: block_to_key(node_to_block(node, sep=sep))
        for node in G.nodes()
    }
    return nx.relabel_nodes(G, mapping, copy=True)


def partition_signature(G, sep="_"):
    """
    Returns an order-independent representation of the graph's partition.
    """
    return frozenset(
        frozenset(node_to_block(node, sep=sep))
        for node in G.nodes()
    )


def same_partition(G1, G2, sep="_"):
    return partition_signature(G1, sep=sep) == partition_signature(G2, sep=sep)


def shd_if_same_partition(oracle_abs_dag, candidate_abs_dag, sep="_", double_for_anticausal=True):
    """
    Computes SHD only when the two abstract DAGs have the same clusters.
    Otherwise returns None.
    """
    if not same_partition(oracle_abs_dag, candidate_abs_dag, sep=sep):
        return {
            "same_partition": False,
            "shd": None,
            "message": "SHD skipped because abstract partitions differ."
        }

    oracle_norm = normalize_graph_nodes(oracle_abs_dag, sep=sep)
    candidate_norm = normalize_graph_nodes(candidate_abs_dag, sep=sep)

    shd = structure_hamming_dist(
        true_graph=oracle_norm,
        pred_graph=candidate_norm,
        double_for_anticausal=double_for_anticausal
    )

    return {
        "same_partition": True,
        "shd": shd,
        "message": "SHD computed because abstract partitions match."
    }

def get_node_to_macro_map(G, sep="_"):
    """
    Creates a mapping from each micro-node to its corresponding macro-node object.
    Example: {'AccuracyEvaluation': 'Accuracy_Fairness_Usefulness'}
    """
    mapping = {}
    for macro_node in G.nodes():
        # Uses your logic to split strings or handle sets/tuples
        block = node_to_block(macro_node, sep=sep)
        for micro_node in block:
            mapping[micro_node] = macro_node
    return mapping

def graph_to_adj_matrix(G, micro_node_list, sep="_"):
    """
    Converts an abstracted DAG into a micro-level 25x25 matrix.
    """
    n = len(micro_node_list)
    matrix = np.zeros((n, n), dtype=int)

    # 1. Get the dictionary mapping (The "Translation Table")
    node_map = get_node_to_macro_map(G, sep=sep)

    # 2. Build the matrix using direct dictionary lookups
    for i, u in enumerate(micro_node_list):
        for j, v in enumerate(micro_node_list):
            m_u = node_map.get(u)
            m_v = node_map.get(v)

            # If both micro-nodes are part of the abstraction and connected in G
            if m_u is not None and m_v is not None:
                if G.has_edge(m_u, m_v):
                    matrix[i, j] = 1

    return matrix