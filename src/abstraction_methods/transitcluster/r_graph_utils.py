from enum import Enum, auto

import networkx as nx
import rpy2.robjects as robjects


class NeighborhoodMode(Enum):
    ANCESTORS = auto()
    DESCENDANTS = auto()
    PARENTS = auto()
    CHILDREN = auto()

class GraphUtils:
    """
    Static utility class to replicate R's igraph behavior in Python.
    Ensures all outputs follow the 'Master Index' order of the original graph.
    """

    @staticmethod
    def get_ordered_neighborhood(graph, nodes, x, mode: NeighborhoodMode):
        """
        Mimics R: v[neighborhood(g, nodes=x, order=..., mode=...)]$name
        """
        # 1. Initialize result with the starting node (Distance 0 in R)
        result = [x]

        # 2. Define the reachability check based on the mode
        if mode == NeighborhoodMode.DESCENDANTS:
            # Can x reach n?
            check = lambda n: nx.has_path(graph, x, n)
        elif mode == NeighborhoodMode.ANCESTORS:
            # Can n reach x?
            check = lambda n: nx.has_path(graph, n, x)
        elif mode == NeighborhoodMode.CHILDREN:
            # Is n a direct successor?
            check = lambda n: n in graph.successors(x)
        elif mode == NeighborhoodMode.PARENTS:
            # Is n a direct predecessor?
            check = lambda n: n in graph.predecessors(x)
        else:
            raise ValueError("Mode must be 'ancestors', 'descendants', 'parents', or 'children'")

        # 3. Scan the Master List to maintain R-style indexing
        # We skip x because it's already at the head [0]
        others = [n for n in nodes if n != x and check(n)]

        result.extend(others)
        return result


    # @staticmethod
    # def get_all_neighborhoods(graph, nodes, mode: NeighborhoodMode):
    #     """
    #     Optimized R-compliant mapping.
    #     Simple filter for Pa/Ch, BFS for An/De.
    #     """
    #     # FAST PATH: Parents and Children (Distance 1)
    #     if mode == NeighborhoodMode.PARENTS:
    #         return {x: [n for n in nodes if n in graph.predecessors(x)] for x in nodes}
    #
    #     if mode == NeighborhoodMode.CHILDREN:
    #         return {x: [n for n in nodes if n in graph.successors(x)] for x in nodes}
    #
    #     # BFS PATH: Ancestors and Descendants (Full Traversal)
    #     all_results = {}
    #     is_reverse = mode == NeighborhoodMode.ANCESTORS
    #     search_graph = graph.reverse() if is_reverse else graph
    #
    #     for x in nodes:
    #         result = [x]  # Include self for an/de
    #         visited = {x}
    #         current_layer = [x]
    #
    #         while current_layer:
    #             next_layer = []
    #             for node in current_layer:
    #                 if node in search_graph:
    #                     # Find neighbors and sort by master index for tie-breaking
    #                     for neighbor in search_graph.successors(node):
    #                         if neighbor not in visited:
    #                             visited.add(neighbor)
    #                             next_layer.append(neighbor)
    #
    #             if next_layer:
    #                 next_layer.sort(key=lambda n: nodes.index(n))
    #                 result.extend(next_layer)
    #             current_layer = next_layer
    #
    #         all_results[x] = result
    #
    #     return all_results
    @staticmethod
    def get_all_neighborhoods(graph, nodes, mode: NeighborhoodMode):
        """
        Unified R-compliant neighborhood generator.
        - PARENTS/CHILDREN: Fast list comprehension filter.
        - ANCESTORS/DESCENDANTS: Layered BFS with Master Index tie-breaking.
        """

        # --- FAST PATH: PARENTS & CHILDREN (Distance 1, Excludes Self) ---
        if mode == NeighborhoodMode.PARENTS:
            return {x: [n for n in nodes if n in graph.predecessors(x)] for x in nodes}

        if mode == NeighborhoodMode.CHILDREN:
            return {x: [n for n in nodes if n in graph.successors(x)] for x in nodes}

        # --- BFS PATH: ANCESTORS & DESCENDANTS (Multi-hop, Includes Self) ---
        all_results = {}

        # Determine search direction based on Enum

        if mode == NeighborhoodMode.ANCESTORS or mode == NeighborhoodMode.DESCENDANTS:
            if mode == NeighborhoodMode.ANCESTORS:
                search_graph = graph.reverse()
            else:
                search_graph = graph
            for x in nodes:
                result = [x]  # Distance 0: Start with the node itself
                visited = {x}
                current_layer = [x]

                while current_layer:
                    next_layer_candidates = []
                    for node in current_layer:
                        if node in search_graph:
                            for neighbor in search_graph.successors(node):
                                if neighbor not in visited:
                                    visited.add(neighbor)
                                    next_layer_candidates.append(neighbor)

                    if next_layer_candidates:
                        # THE R TIE-BREAKER: Sort by the master list index
                        next_layer_candidates.sort(key=lambda n: nodes.index(n))
                        result.extend(next_layer_candidates)

                    current_layer = next_layer_candidates

                all_results[x] = result
        # else:
        #     # for x in nodes:
        #     #     result = GraphUtils.get_bfs_nodes(graph, nodes, x)
        #     #     all_results[x] = result
        #     all_results = {x: GraphUtils.get_bfs_nodes(graph, x, mode='out') for x in nodes}
        return all_results
    # @staticmethod
    # def get_all_neighborhoods(graph, nodes, mode: NeighborhoodMode):
    #     """
    #     Traverses all nodes and returns a dict of R-ordered neighborhoods.
    #     Matches R's v[neighborhood(g, nodes=V(g), ...)]$name
    #     """
    #     all_mappings = {}
    #
    #     for x in nodes:
    #         # R-style: The starting node x is always first (Distance 0)
    #         result = [x]
    #
    #         # Identify the check based on mode
    #         if mode == NeighborhoodMode.ANCESTORS:
    #             check = lambda n: nx.has_path(graph, n, x)
    #         elif mode == NeighborhoodMode.DESCENDANTS:
    #             check = lambda n: nx.has_path(graph, x, n)
    #         elif mode == NeighborhoodMode.PARENTS:
    #             check = lambda n: n in graph.predecessors(x)
    #         elif mode == NeighborhoodMode.CHILDREN:
    #             check = lambda n: n in graph.successors(x)
    #
    #         # Scan Master List to preserve index order
    #         others = [n for n in nodes if n != x and check(n)]
    #         result.extend(others)
    #
    #         all_mappings[x] = result
    #
    #     return all_mappings

    @staticmethod
    def unique_ordered(list_of_lists):
        """
        Mimics R's unique() on a list of vectors.
        Preserves the order of first appearance.
        """
        seen = []
        for item in list_of_lists:
            if item not in seen and len(item) > 0:
                seen.append(item)
        return seen

    # @staticmethod
    # def get_bfs_descendants(graph, master_list, start_node):
    #     """
    #     Direct BFS implementation replicating igraph::neighborhood
    #     """
    #     result = [start_node]
    #     visited = {start_node}
    #     queue = [start_node]  # Using a list as a queue for layer-by-layer processing
    #
    #     # We use a while loop to process layers
    #     current_layer = [start_node]
    #
    #     while current_layer:
    #         next_layer = []
    #         # Discover all potential candidates from the current layer
    #         for node in current_layer:
    #             # Find children
    #             successors = list(graph.successors(node))
    #             for s in successors:
    #                 if s not in visited:
    #                     visited.add(s)
    #                     next_layer.append(s)
    #
    #         if next_layer:
    #             # THE R TIE-BREAKER:
    #             # Sort the discovered nodes by their index in the master list
    #             next_layer.sort(key=lambda n: master_list.index(n))
    #             result.extend(next_layer)
    #
    #         current_layer = next_layer
    #
    #     return result

    # @staticmethod
    # def get_bfs_nodes(graph, start_node, mode='out'):
    #     # This mirrors R's igraph::neighborhood behavior
    #     # mode='out' for successors/descendants, mode='in' for predecessors/ancestors
    #     edges = graph.out_edges if mode == 'out' else graph.in_edges
    #
    #     visited = []
    #     queue = [start_node]
    #     seen = {start_node}
    #
    #     while queue:
    #         curr = queue.pop(0)
    #         visited.append(curr)
    #         # Sort neighbors alphabetically ONLY so discovery is consistent
    #         neighbors = sorted([v if mode == 'out' else u for u, v in edges(curr)])
    #         for n in neighbors:
    #             if n not in seen:
    #                 seen.add(n)
    #                 queue.append(n)
    #     return visited
    @staticmethod
    def cluster_edge_subgraph(graph, incoming_nodes, outgoing_nodes):
        """
        Python equivalent of R's subgraph.edges(g, e_to_keep, delete.vertices = FALSE)
        """
        # 1. Identify edges to REMOVE
        # R: e[.to(incoming)] -> Edges pointing TO nodes in the 'incoming' set
        # R: e[.from(outgoing)] -> Edges coming FROM nodes in the 'outgoing' set
        edges_to_remove = {
            (u, v) for u, v in graph.edges()
            if v in incoming_nodes or u in outgoing_nodes
        }

        # 2. Identify edges to KEEP
        edges_to_keep = [e for e in graph.edges() if e not in edges_to_remove]

        # 3. Create the edge-induced subgraph
        # IMPORTANT: NetworkX's edge_subgraph drops nodes with no remaining edges.
        subgraph = graph.edge_subgraph(edges_to_keep).copy()

        # 4. REPLICATE 'delete.vertices = FALSE'
        # Add back all original nodes so isolated nodes aren't lost
        subgraph.add_nodes_from(graph.nodes())

        return subgraph

    @staticmethod
    def force_r_parity(graph, master_nodes):
        """
        Rebuilds the graph's internal adjacency to strictly
        follow the R-Vertex integer order (V(g)).
        """
        node_to_idx = {node: i for i, node in enumerate(master_nodes)}

        # We must modify the internal _adj to control traversal order
        for u in list(graph.nodes()):
            # Sort neighbors by their position in the R-Master-Index
            sorted_neighbors = sorted(graph[u], key=lambda n: node_to_idx.get(n, 999))

            # Rebuild the dictionary to preserve this insertion order
            # (Python 3.7+ dicts are ordered by default)
            new_adj = {v: graph[u][v] for v in sorted_neighbors}
            graph.adj[u] = new_adj

        return graph

    @staticmethod
    def get_r_connected_component(graph, start_nodes, master_nodes):
        """
        Replicates R's A <- uu(connected(XY, g_ne))
        Finds the weakly connected component and returns it in R-Vertex order.
        """
        # 1. Convert to undirected to match 'mode = "all"'
        undirected_g = graph.to_undirected()

        # 2. Re-apply parity to the undirected version
        # (Crucial: to_undirected can scramble neighbor order)
        undirected_g = GraphUtils.force_r_parity(undirected_g, master_nodes)

        reachable_pool = set()
        for node in start_nodes:
            if node in undirected_g:
                # Get all nodes in the component
                comp = nx.node_connected_component(undirected_g, node)
                reachable_pool.update(comp)

        # 3. REPLICATE R's v[co_ind]$name
        # This filters the master list to ensure the output
        # is a sorted sub-sequence of the original graph.
        return [n for n in master_nodes if n in reachable_pool]

    @staticmethod
    def uu_rel(node_list, rel_dict, master_nodes):
        """
        Pure list-based union that maintains R-Vertex order.
        Equivalent to R's unique(unlist(rel_dict[node_list])) sorted by V(g).
        """
        return [
            n for n in master_nodes
            if any(n in rel_dict.get(root, []) for root in node_list)
        ]

    @staticmethod
    def get_ordered_subcomponents(subgraph, master_nodes):
        """
        Replicates R's components(induced_subgraph(g, A))
        Returns a list of components, where each component is R-ordered,
        and the list itself is sorted by the first appearing node.
        """
        # Get raw components from NetworkX
        raw_comps = [list(c) for c in nx.weakly_connected_components(subgraph)]

        # 1. Sort nodes WITHIN each component by Master Order
        ordered_comps = [
            [n for n in master_nodes if n in c] for c in raw_comps
        ]

        # 2. Sort the LIST of components by the index of their first node
        # (This ensures we process Component 1 before Component 2 if C1
        # contains the 'earliest' node in the graph).
        return sorted(ordered_comps, key=lambda c: master_nodes.index(c[0]))

    @staticmethod
    def get_r_vertex_order(r_graph_obj):
        """
        Connects to the R instance and pulls the exact
        internal vertex name sequence: V(g)$name.
        """
        # 1. Define the R command to extract names
        # rpy2 allows us to execute R code strings directly
        get_names = robjects.r('function(g) { igraph::V(g)$name }')

        # 2. Call the R function on your graph object
        r_names_vector = get_names(r_graph_obj)

        # 3. Convert R character vector to Python list
        return list(r_names_vector)

    @staticmethod
    def sync_networkx_to_r(nx_graph, r_graph_obj):
        """
        Uses R as the source of truth to re-order the
        NetworkX nodes and adjacency structure.
        """
        # Get the 'Gold Standard' order from R
        true_order = GraphUtils.get_r_vertex_order(r_graph_obj)

        # Re-apply parity to the existing NetworkX graph
        # This uses the force_r_parity logic we built earlier
        return GraphUtils.force_r_parity(nx_graph, true_order)