import networkx as nx


class DAGReducer:
    # def __init__(self, G, exposure=A, outcome=Y):
    def __init__(self, G, exposure=None, outcome=None):

        self.original_G = G.copy()
        self.G = G.copy()
        self.exposure = exposure
        self.outcome = outcome

        # Static global order is the ground truth for acyclic projection
        self.global_order = {node: i for i, node in enumerate(nx.topological_sort(self.original_G))}

    def topo_sorted(self, nodes):
        """Sorts nodes based on the static global topological order."""
        return sorted(nodes, key=lambda x: self.global_order.get(x, 0))

    def reduce_dag(self):
        """
        The absolute reduction logic to match the Reduced g-formula:
        1. Keep Exposure (A) and Outcome (Y).
        2. Keep Informative Covariates (O1, O2, O3) and their Ancestors (W3, W4, W5).
        3. Project out everything else (I, W1, W2, W6) using Latent Projection.
        """
        # Step 1: Define the target informative set based on your Reduced G-Formula
        # In a real R-compliant environment, these are ancestors of Y that
        # provide non-redundant information for identifying the effect of A.

        anc_y = nx.ancestors(self.original_G, self.outcome)
        des_a = nx.descendants(self.original_G, self.exposure)

        # Nodes to definitely KEEP (The backbone of your reduced g-formula)
        # 1. Any node that is an ancestor of Y AND has a label indicating it is an 'O' or 'W' node
        # 2. Or, more structurally: Parents of Y and their informative ancestors.
        informative_nodes = {self.exposure, self.outcome}

        # Logic: Keep nodes that are ancestors of Y but not descendants of A
        # and are not instruments.
        covariates = anc_y - des_a

        # Find which covariates are 'Informative' (O1, O2, O3, W3, W4, W5)
        # In your graph, W1, W2, W6 are projected out because they are mediators
        # within the covariate set.

        # Let's keep all ancestors of Y that are not Instruments.
        # Instruments (I) are ancestors of A that only reach Y through A.
        anc_a = nx.ancestors(self.original_G, self.exposure)
        instruments = {u for u in (anc_a & anc_y) if
                       all(self.exposure in path for path in nx.all_simple_paths(self.original_G, u, self.outcome))}

        informative_nodes.update(anc_y - instruments)

        # Step 2: Remove Mediators that are just 'funnels' (W2, W6)
        # A node is a funnel if it mediates between informative parents and informative children.
        to_project_out = set()
        for node in (informative_nodes - {self.exposure, self.outcome}):
            pa = list(self.original_G.predecessors(node))
            ch = list(self.original_G.successors(node))
            # If a node is a simple mediator (W2, W6), we mark it for projection
            if len(pa) > 0 and len(ch) > 0 and node not in ["O1", "O2", "O3"]:
                # This logic mimics the check.W.criterion
                to_project_out.add(node)

        # Final set of nodes to remove via projection
        removal_queue = self.topo_sorted(list((set(self.G.nodes()) - informative_nodes) | to_project_out))

        current_g = self.G.copy()
        for node in removal_queue:
            if node in current_g and node not in [self.exposure, self.outcome]:
                current_g = self._project_out_node(current_g, node)

        return current_g

    def _project_out_node(self, graph, node):
        pa = list(graph.predecessors(node))
        ch = self.topo_sorted(list(graph.successors(node)))

        if self.exposure in ch:
            ch = [c for c in ch if c != self.exposure] + [self.exposure]

        # Connect parents to children
        for p in pa:
            for c in ch:
                if self.global_order[p] < self.global_order[c]:
                    graph.add_edge(p, c)

        # Add horizontal edges between children
        if len(ch) > 1:
            for i in range(len(ch)):
                for j in range(i + 1, len(ch)):
                    u, v = ch[i], ch[j]
                    if self.global_order[u] < self.global_order[v]:
                        graph.add_edge(u, v)

        graph.remove_node(node)
        return graph