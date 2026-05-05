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


def normalize_graph_nodes_for_shd(G, sep="_"):
    mapping = {
        node: block_to_key(node_to_block(node, sep=sep))
        for node in G.nodes()
    }
    return nx.relabel_nodes(G, mapping, copy=True)


def shd_with_normalized_nodes(
    oracle_abs_dag,
    candidate_abs_dag,
    sep="_",
    double_for_anticausal=True
):
    oracle_norm = normalize_graph_nodes_for_shd(oracle_abs_dag, sep=sep)
    candidate_norm = normalize_graph_nodes_for_shd(candidate_abs_dag, sep=sep)

    if set(oracle_norm.nodes()) != set(candidate_norm.nodes()):
        raise ValueError(
            "SHD cannot be computed directly because the normalized abstract node sets differ.\n"
            f"Only in oracle: {set(oracle_norm.nodes()) - set(candidate_norm.nodes())}\n"
            f"Only in candidate: {set(candidate_norm.nodes()) - set(oracle_norm.nodes())}"
        )

    return structure_hamming_dist(
        true_graph=oracle_norm,
        pred_graph=candidate_norm,
        double_for_anticausal=double_for_anticausal
    )