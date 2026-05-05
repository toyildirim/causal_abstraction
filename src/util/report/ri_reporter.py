from sklearn.metrics import rand_score, adjusted_rand_score


def node_to_block(node):
    if isinstance(node, tuple):
        return set(node)
    if isinstance(node, set):
        return node
    if isinstance(node, frozenset):
        return set(node)
    if isinstance(node, str) and "_" in node:
        return set(node.split("_"))

    return {node}

def infer_original_nodes(*graphs):
    nodes = set()

    for G in graphs:
        for abstract_node in G.nodes():
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


def ri_ari_report(oracle_dag, candidate_dag, original_nodes=None):
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