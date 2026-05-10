from sklearn.metrics import rand_score, adjusted_rand_score
from dodiscover.metrics import structure_hamming_dist
import numpy as np
import networkx as nx
from sklearn.metrics import jaccard_score


def node_to_block(node, sep="_"):
    if isinstance(node, (tuple, list, set, frozenset)):
        return set(node)

    elif isinstance(node, dict):
        # Extracts the micro-node identifiers from keys
        return set(node.keys())

    elif isinstance(node, str):
        if sep in node:
            return set(node.split(sep))
        return {node}

    else:
        # Fallback for any other type (int, float, etc.)
        return {node}


def infer_original_nodes(*graphs):
    nodes = set()

    for G in graphs:
        for abstract_node in G.nodes():
            if abstract_node is not None or abstract_node != '':
                nodes.update(node_to_block(abstract_node))

    return sorted(nodes, key=str)


def dag_to_labels(G, original_nodes):
    labels = [-1] * len(original_nodes)

    for cluster_id, abstract_node in enumerate(G.nodes()):
        block = node_to_block(abstract_node)

        for i, node in enumerate(original_nodes):
            if node in block:
                labels[i] = cluster_id

    if -1 in labels:
        missing = [original_nodes[i] for i, label in enumerate(labels) if label == -1]
        raise ValueError(f"Missing nodes in partition: {missing}")

    return labels


# def ri_ari_report(oracle_dag, candidate_dag, original_nodes=None):
#     if original_nodes is None:
#         original_nodes = infer_original_nodes(oracle_dag, candidate_dag)
#
#     oracle_labels = dag_to_labels(oracle_dag, original_nodes)
#     candidate_labels = dag_to_labels(candidate_dag, original_nodes)
#
#     return {
#         "RI": rand_score(oracle_labels, candidate_labels),
#         "ARI": adjusted_rand_score(oracle_labels, candidate_labels),
#         "original_nodes": original_nodes,
#         "oracle_labels": oracle_labels,
#         "candidate_labels": candidate_labels,
#     }

def calculate_ri_ari(oracle_dag, candidate_dag, original_nodes=None):
    if original_nodes is None:
        original_nodes = infer_original_nodes(oracle_dag, candidate_dag)

    oracle_labels = dag_to_labels(oracle_dag, original_nodes)
    candidate_labels = dag_to_labels(candidate_dag, original_nodes)

    return {
        "RI": rand_score(oracle_labels, candidate_labels),
        "ARI": adjusted_rand_score(oracle_labels, candidate_labels),
        "original_nodes": original_nodes,
        "oracle_labels": oracle_labels,
        "candidate_labels": candidate_labels,
    }
# def shd_if_same_partition(oracle_abs_dag, candidate_abs_dag, sep="_", double_for_anticausal=True):
#     """
#     Computes SHD only when the two abstract DAGs have the same clusters.
#     Otherwise returns None.
#     """
#     if not same_partition(oracle_abs_dag, candidate_abs_dag, sep=sep):
#         return {
#             "same_partition": False,
#             "shd": None,
#             "message": "SHD skipped because abstract partitions differ."
#         }
#
#     oracle_norm = normalize_graph_nodes(oracle_abs_dag, sep=sep)
#     candidate_norm = normalize_graph_nodes(candidate_abs_dag, sep=sep)
#
#     shd = structure_hamming_dist(
#         true_graph=oracle_norm,
#         pred_graph=candidate_norm,
#         double_for_anticausal=double_for_anticausal
#     )
#
#     return {
#         "same_partition": True,
#         "shd": shd,
#         "message": "SHD computed because abstract partitions match."
#     }

def calculate_shd(oracle_abs_dag, candidate_abs_dag, sep="_", double_for_anticausal=True):
    """
    Computes SHD only when the two abstract DAGs have the same clusters.
    Otherwise returns None.
    """
    original_nodes = infer_original_nodes(oracle_abs_dag, candidate_abs_dag)
    adj_oracle = graph_to_adj_matrix(oracle_abs_dag, original_nodes, sep=sep)
    adj_abstracted = graph_to_adj_matrix(candidate_abs_dag, original_nodes, sep=sep)

    # 1. Convert your matrices to temporary DiGraphs
    g_oracle_reshaped = nx.from_numpy_array(adj_oracle, create_using=nx.DiGraph)
    g_abstracted_reshaped = nx.from_numpy_array(adj_abstracted, create_using=nx.DiGraph)

    shd = structure_hamming_dist(
        true_graph=g_oracle_reshaped,
        pred_graph=g_abstracted_reshaped,
        double_for_anticausal=double_for_anticausal
    )

    return {
        "shd": shd,
        "message": "SHD computed for each partitions."
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


# def create_lifted_matrix(G, canonical_list, node_map):
#     """
#     EXECUTION PHASE: Zero calls to node_to_block.
#     """
#     n = len(canonical_list)
#     matrix = np.zeros((n, n), dtype=int)
#
#     for i, u in enumerate(canonical_list):
#         for j, v in enumerate(canonical_list):
#             m_u = node_map.get(u)  # O(1) lookup
#             m_v = node_map.get(v)
#
#             if m_u and m_v and G.has_edge(m_u, m_v):
#                 matrix[i, j] = 1
#
#     return matrix

def get_graph_partitions(G, sep="_"):
    """
    Returns both the signature (for ARI) and the mapping (for SHD).
    """
    node_map = {}
    blocks = []

    for macro_node in G.nodes():
        # Identify the micro-nodes inside this cluster
        block_set = frozenset(node_to_block(macro_node, sep=sep))
        blocks.append(block_set)

        # Build the lookup dictionary
        for micro_node in block_set:
            node_map[micro_node] = macro_node

    signature = frozenset(blocks)
    return signature, node_map

# def calculate_shd(oracle_abs_dag, candidate_abs_dag, sep="_", double_for_anticausal=True):
#     # 1. Infer the "Universe" of 25 nodes
#     canonical_list = infer_original_nodes(oracle_abs_dag, candidate_abs_dag)
#
#     # 2. Get the mappings for both
#     sig_oracle, map_oracle = get_graph_partitions(oracle_abs_dag)
#     sig_algo, map_algo = get_graph_partitions(candidate_abs_dag)
#
#     # 3. Create the matrices using the Canonical List as the 'Ruler'
#     m_oracle = graph_to_adj_matrix(oracle_abs_dag, canonical_list, map_oracle)
#     m_algo = graph_to_adj_matrix(candidate_abs_dag, canonical_list, map_algo)
#
#     # 4. Calculate SHD (Hamming Distance)
#     # shd = np.sum(np.abs(m_oracle - m_algo))
#     shd = structure_hamming_dist(
#         true_graph=m_oracle,
#         pred_graph=m_algo,
#         double_for_anticausal=double_for_anticausal
#     )
#
#     return {
#         "same_partition": True,
#         "shd": shd,
#         "message": "SHD computed for given adj matrices."
#     }
# def calculate_jaccard (m_oracle, m_algo):
def calculate_jaccard(oracle_abs_dag, candidate_abs_dag, sep ="_"):
    original_nodes = infer_original_nodes(oracle_abs_dag, candidate_abs_dag)
    adj_oracle = graph_to_adj_matrix(oracle_abs_dag, original_nodes, sep=sep)
    adj_abstracted = graph_to_adj_matrix(candidate_abs_dag, original_nodes, sep=sep)
    # Flatten the matrices to 1D vectors
    js = jaccard_score(adj_oracle.flatten(), adj_abstracted.flatten())
    return {
        "JS": js,
        "message": "Jaccard Score computed for the given graph adj matrices."
    }

# def analyze_graphs_for_metrics(*graphs, sep="_"):
#     """
#     ONE PASS: Parses all nodes using node_to_block exactly once.
#     Returns: (canonical_list, list_of_partition_data)
#     """
#     all_micro_nodes = set()
#     graph_data = []  # Stores (signature, node_map) for each graph
#
#     for G in graphs:
#         node_map = {}
#         blocks = []
#
#         for macro_node in G.nodes():
#             # The ONLY place node_to_block is called in the entire workflow
#             micro_nodes_in_block = node_to_block(macro_node, sep=sep)
#
#             # Update the 'Universe' of nodes
#             all_micro_nodes.update(micro_nodes_in_block)
#
#             # Prepare ARI data
#             block_frozenset = frozenset(micro_nodes_in_block)
#             blocks.append(block_frozenset)
#
#             # Prepare SHD mapping
#             for micro_node in micro_nodes_in_block:
#                 node_map[micro_node] = macro_node
#
#         graph_data.append({
#             'signature': frozenset(blocks),
#             'node_map': node_map
#         })
#
#     # Create the 'Ruler' for matrix alignment
#     canonical_list = sorted(list(all_micro_nodes), key=str)
#
#     return canonical_list, graph_data