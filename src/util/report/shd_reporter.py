import networkx as nx
from dodiscover.metrics import structure_hamming_dist


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