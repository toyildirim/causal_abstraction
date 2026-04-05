import networkx as nx


def to_adj_mat(G):
    """Converts a NetworkX graph to a pandas DataFrame adjacency matrix."""
    # Ensure the matrix order matches the node list for consistency
    nodes = list(G.nodes())
    adj_mat = nx.to_pandas_adjacency(G, nodelist=nodes, dtype=int)
    return adj_mat


def to_dag_string(adj_mat, coords=None, subset=None):
    """
    Converts an adjacency matrix to a dagitty-style string.
    In Python, we usually keep this as a nx.DiGraph,
    but this mimics the R logic for string generation.
    """
    V = list(adj_mat.index)
    if subset is not None:
        V = [v for v in V if v in subset]
        adj_mat = adj_mat.loc[V, V]

    lines = ["dag {"]

    # Add coordinates and special labels (Exposure/Outcome)
    if coords is not None:
        for v in V:
            pos_str = f'pos="{coords[v][0]},{coords[v][1]}"'
            if v == "A":
                lines.append(f'  A [{pos_str}, exposure]')
            elif v == "Y":
                lines.append(f'  Y [{pos_str}, outcome]')
            else:
                lines.append(f'  {v} [{pos_str}]')

    # Add edges
    for i, row_node in enumerate(V):
        for j, col_node in enumerate(V):
            if adj_mat.iloc[i, j] == 1:
                lines.append(f'  {row_node} -> {col_node}')

    lines.append("}")
    return "\n".join(lines)


def get_induced_graph(G, subset):
    """Returns a subgraph containing only the nodes in 'subset'."""
    return G.subgraph(subset).copy()


def project_out(G, v):
    """
    Projects out a variable with 0 or 1 child.
    If v has parents and one child, it connects parents directly to that child.
    """
    parents = list(G.predecessors(v))
    children = list(G.successors(v))

    if len(children) >= 2:
        raise ValueError(f"Node '{v}' has {len(children)} children. Project_out requires < 2.")

    new_G = G.copy()

    # If there is exactly one child, bridge the gap
    if len(children) == 1 and len(parents) > 0:
        child = children[0]
        for parent in parents:
            new_G.add_edge(parent, child)

    # Remove the projected node
    new_G.remove_node(v)
    return new_G