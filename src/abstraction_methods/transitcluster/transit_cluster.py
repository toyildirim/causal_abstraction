import networkx as nx

from abstraction_methods.transitcluster.r_graph_utils import GraphUtils as gu, NeighborhoodMode


class TransitCluster:
    def __init__(self, G=None, nodes=None):
        self.G = G
        self.nodes = nodes if nodes is not None else (list(G.nodes()) if G is not None else None)
    def find_transit_components(self, G=None, prohibit=None, singletons=False):
        # Use provided G or the instance's G
        graph = G if G is not None else self.G
        if graph is None:
            raise ValueError("No graph provided to find_transit_components.")
        nodes = self.nodes
        n = len(nodes)
        restrict = nodes
        if prohibit is not None and len(prohibit) != 0:
            restrict = nodes.remove(prohibit)

        # prohibit = set(prohibit) if prohibit else set()
        # nodes = sorted(list(graph.nodes()))
        ch = gu.get_all_neighborhoods(graph, nodes, NeighborhoodMode.CHILDREN)
        pa = gu.get_all_neighborhoods(graph, nodes, NeighborhoodMode.PARENTS)
        # ch = {x: [n for n in nodes if n in graph.successors(x)] for x in nodes}
        # pa = {x: [n for n in nodes if n in graph.predecessors(x)] for x in nodes}
        # restrict = set(nodes) - prohibit
        an = gu.get_all_neighborhoods(graph,nodes,NeighborhoodMode.ANCESTORS)
        de = gu.get_all_neighborhoods(graph,nodes,NeighborhoodMode.DESCENDANTS)


        # Pre-calculate relatives (Inclusive logic for R-parity)
        # Force alphabetical or "definition order" sorting to match R's vector behavior
        # pa = {node: list(graph.predecessors(node)) for node in nodes}
        # ch = {node: list(graph.successors(node)) for node in nodes}

        # 2. Mirror 'children' and 'parents'
        # R equivalent: v[ch_ind]$name


        # 3. Mirror 'ancestors' and 'descendants'
        # R equivalent: v[an_ind]$name
        # R: ancestors(x, g)
        # Python:
        # an = {x: [n for n in nodes if n == x or n in nx.ancestors(graph, x)] for x in nodes}
        # R: descendants(x, g)
        # Python:
        # de = {x: [n for n in nodes if n == x or n in nx.descendants(graph, x)] for x in nodes}
        # an = {x: [n for n in nodes if n in (nx.ancestors(graph, x) | {x})] for x in nodes}
        # de = {x: [n for n in nodes if n in (nx.descendants(graph, x) | {x})] for x in nodes}

        # For ancestors and descendants, always convert the set to a sorted list
        # an = {node: list({node} | nx.ancestors(graph, node) ) for node in nodes}
        # de = {node: list({node} | nx.descendants(graph, node) ) for node in nodes}

        # 1. Get the raw sets from NetworkX (order-agnostic)
        # 2. Re-order them based on their position in the 'nodes' list
        # an = {}
        # for node in nodes:
        #     # Logic: Who are the ancestors?
        #     anc_set =  nx.ancestors(graph, node) | {node}
        #     # Parity: Keep them in the order they appear in the global 'nodes' list
        #     an[node] = [n for n in nodes if n in anc_set]
        #
        # de = {}
        # for node in nodes:
        #     # Logic: Who are the descendants?
        #     des_set = nx.descendants(graph, node) | {node}
        #     # Parity: Keep them in the order they appear in the global 'nodes' list
        #     de[node] = [n for n in nodes if n in des_set]

        # Pre-convert restrict to a set for O(1) lookups (performance),
        # but we iterate over the LIST to preserve order.
        # r_set = set(restrict) if restrict is not None else None
        r_set = restrict if restrict is not None else None
        # C_SET: Preserves the order found in ch[node]
        c_set = [
            [val for val in ch[node] if r_set is None or val in r_set]
            for node in nodes if ch[node]
        ]

        # P_SET: Preserves the order found in pa[node]
        p_set = [
            [val for val in pa[node] if r_set is None or val in r_set]
            for node in nodes if pa[node]
        ]

        tc = []

        if singletons:
            for x in restrict:
                y = {"vertices": [x], "receivers": [x] if pa[x] else [], "emitters": [x] if ch[x] else []}
                tc.append(y)

        if restrict is not None and len(restrict) == n:
            tc.append({"vertices": nodes, "receivers": [], "emitters": []})
        # To maintain a 'List of Lists' structure
        c_set.append([])
        p_set.append([])

        # # 1. Deduplicate while forcing a sort (this matches R's unique() behavior)
        # unique_c = sorted([sorted(list(s)) for s in {frozenset(s) for s in c_set}])
        # unique_p = sorted([sorted(list(s)) for s in {frozenset(s) for s in p_set}])

        def get_unique_ordered(original_sets):
            seen = []
            for s in original_sets:
                # Convert to list to check content regardless of set-ordering
                if s not in seen: #and len(s) > 0:
                    seen.append(s)
            return seen

        unique_c = get_unique_ordered(c_set)
        unique_p = get_unique_ordered(p_set)

        def uu_rel(node_list, rel_dict, master_nodes):
            """
            Replicates the R 'Named List' structure.
            Returns a dict: { input_node: [ordered_related_nodes] }
            """
            result_map = {}

            for node in node_list:
                if node in rel_dict:
                    # Get the raw relations for this specific node
                    result_map[node] = rel_dict[node]

                    # # Filter and order them based on the Master Ruler
                    # # This replicates: V(g)[ancestors]$name
                    # ordered_relations = [n for n in master_nodes if n in raw_relations]
                    #
                    # result_map[root] = ordered_relations

            return result_map

        for x_orig in unique_c:
            # x_orig = set(x_orig_list)  # Convert back to set for math operations
            print("x_orig-->",x_orig)
            for y_orig in unique_p:
                # y_orig = set(y_orig_list)
                print("y_orig-->", y_orig)
                an_y = uu_rel(y_orig, an, nodes)
                de_x = uu_rel(x_orig, de,nodes)
                # an_de_xy = an_y & de_x
                # 2. Intersect them while keeping the keys (replicating R logic)
                # an_de_xy = []
                # # Usually, in these causal scripts, we intersect nodes that appear in BOTH maps
                # common_keys = [k for k in y_orig if k in x_orig]  # Adjust based on your R logic
                #
                # for k in an_y:
                #     if k in de_x:
                #         # Intersect the lists for this specific node, preserving master order
                #         an_nodes = an_y[k]
                #         de_nodes = de_x[k]
                #         an_de_xy[k] = [n for n in nodes if n in an_nodes and n in de_nodes]

                # 1. Extract all unique ancestors from the an_y dictionary (Equivalent to unlist() in R)
                # This creates a 'pool' of all nodes that are ancestors of any Y
                # all_an_nodes = {node for sublist in an_y.values() for node in sublist}
                all_an_nodes = []
                seen = set()  # We use a set ONLY for O(1) lookups, it doesn't touch the output order

                # Single pass through the dictionary values
                for sublist in an_y.values():
                    for node in sublist:
                        if node not in seen:
                            all_an_nodes.append(node)
                            seen.add(node)

                # Result: Unique nodes in the order they were first 'discovered' in the dict
                print(all_an_nodes)
                # 2. Extract all unique descendants from the de_x dictionary
                # This creates a 'pool' of all nodes that are descendants of any X
                # all_de_nodes = {node for sublist in de_x.values() for node in sublist}
                # 3. RECYCLE: Clear the set for the next task
                seen.clear()

                # 4. Process de_x
                all_de_nodes = []
                for sublist in de_x.values():
                    for node in sublist:
                        if node not in seen:
                            all_de_nodes.append(node)
                            seen.add(node)

                # 5. FINAL CLEANUP: Remove it from memory entirely when the script is done
                del seen
                # 3. The R-Parity Intersection
                # We walk through our 'Master Ruler' and only keep nodes that appear in BOTH pools
                an_de_xy = [n for n in nodes if n in all_an_nodes and n in all_de_nodes]

                print(f"Resulting Vector: {an_de_xy}")
                # Matches R screenshot: "X_1" (if only X_1 is in the intersection)
                        # an_de_xy.append(k)
                # x = x_orig & an_de_xy if y_orig else x_orig
                # y = y_orig & an_de_xy if x_orig else y_orig
                #
                # xy = x | y
                # 1. Intersection for X: Keep nodes in x_orig that are also in an_de_xy
                if y_orig:
                    # We iterate over x_orig because it is already R-ordered
                    # and we only keep items if they exist in the an_de_xy list
                    x = [n for n in x_orig if n in an_de_xy]
                else:
                    x = x_orig

                # 2. Intersection for Y: Keep nodes in y_orig that are also in an_de_xy
                if x_orig:
                    # Again, preserving the order of y_orig
                    y = [n for n in y_orig if n in an_de_xy]
                else:
                    y = y_orig

                # 3. Union for XY: Combine X and Y while maintaining Master Order
                # We iterate over the entire master_nodes list to ensure
                # the combined result follows the R Vertex Sequence.
                xy = [n for n in nodes if n in x or n in y]
                if len(xy) == n or len(xy) == 0:
                    continue

                if x and not all(pa[node] for node in x): continue
                if y and not all(ch[node] for node in y): continue

                if xy:
                    # 1. APPLY THE EDGE CUT (equivalent to your edge_subgraph function)
                    # Remove edges pointing TO x and edges coming FROM y
                    g_ne = self.cluster_edge_subgraph(graph, x, y)

                    # 2. FIND A (Nodes reachable from XY in the 'cut' graph)
                    # In R, 'connected' usually refers to the weak component containing XY
                    # or nodes reachable in the undirected version of the modified graph.
                    # Convert to undirected for weak connectivity check
                     # 1. Identify ALL reachable nodes in the cut graph (g_ne)
                    # We use a set for the 'pool' just to make the 'in' check fast,
                    # but the final result is a LIST.
                    # reachable_pool = set()
                    # g_ne_undirected = g_ne.to_undirected()
                    #
                    # for node in xy:
                    #     if node in g_ne_undirected:
                    #         # Get the component as a set
                    #         component = nx.node_connected_component(g_ne_undirected, node)
                    #         reachable_pool.update(component)
                    # 1. Start with an empty list and a seen-set to preserve order
                    a_ordered = []
                    seen = set()
                    g_ne_undirected = g_ne.to_undirected()

                    # 2. Iterate through your XY nodes in the EXACT order they appear in R
                    for start_node in xy:
                        if start_node in g_ne_undirected and start_node not in seen:
                            # We perform a manual BFS to capture nodes in discovery order
                            # This is the Python equivalent of igraph::neighborhood
                            bfs_nodes = list(nx.bfs_tree(g_ne_undirected, source=start_node).nodes())

                            for node in bfs_nodes:
                                if node not in seen:
                                    a_ordered.append(node)
                                    seen.add(node)

                    # a_ordered should now match your R screenshot ["X_1", "R_1", "A_3", ...]
                    a_list = a_ordered
                    # 2. REPLICATE R's v[co_ind]$name
                    # We iterate through self.nodes to ensure the order is identical to R's V(g)
                    # a_list = [n for n in nodes if n in reachable_pool]


                    # Check restrictions using pure list logic
                    is_subset = True
                    if restrict is not None:
                        for n in a_list:
                            if n not in restrict:
                                is_subset = False
                                break

                    if len(a_list) > 1 and is_subset:
                        # --- 1. SETUP: The Subgraph (Identical to R induced_subgraph) ---
                        sub = graph.subgraph(a_list)

                        # --- 2. CONNECTIVITY: Find the 'Islands' (Identical to R components) ---
                        # We convert the unordered sets into a list of lists immediately
                        raw_groups = list(nx.weakly_connected_components(sub))

                        # --- 3. ORDERING: The 'Master Ruler' Filter (Identical to R memb_v) ---
                        # We iterate over 'nodes' (the Master Ruler) to keep the U_1, X_1... order.
                        # This fixes the 'Y_1' appearing first problem.
                        ordered_components = []
                        for g_set in raw_groups:
                            component_nodes = [n for n in nodes if n in g_set]
                            ordered_components.append(component_nodes)

                        # --- 4. THE LOOP: (Identical to R 'for (k in seq_along...)') ---
                        # 'i' starts at 0 and increments automatically each loop.
                        for i, A_k in enumerate(ordered_components):
                            # k (R-style ID) is simply i + 1
                            k = i + 1
                            n_A_k = len(A_k)

                            # Logic: if (n_A_k > 1 && n_A_k < n)
                            if 1 < n_A_k < len(nodes):
                                # Find intersections within THIS specific island (A_k)
                                X_k = [n for n in x if n in A_k]
                                Y_k = [n for n in y if n in A_k]

                                print(f"Processing Cluster {k}: {A_k}")

                                # --- Run your Transit Component check ---
                                result = self.is_transit_component(X_k, Y_k, A_k, graph)
                                if result:
                                    tc.extend(result)

        # --- DEFINING unique_tc: Deduplicate using a 'Seen' list ---
        unique_tc = []
        for item in tc:
            # Check if this component (vertices/receivers/emitters) is already in unique_tc
            is_duplicate = False
            for existing in unique_tc:
                if (existing["vertices"] == item["vertices"] and
                        existing["receivers"] == item["receivers"] and
                        existing["emitters"] == item["emitters"]):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_tc.append(item)

        return unique_tc

    # def is_transit_component(self, X, Y, A, G):
    #     X_set, Y_set, A_set = set(X), set(Y), set(A)
    #
    #     pa_X_ext = {frozenset(set(G.predecessors(x)) - A_set) for x in X_set}
    #     if (len(pa_X_ext)
    #             > 1): return None
    #
    #     ch_Y_ext = {frozenset(set(G.successors(y)) - A_set) for y in Y_set}
    #     if len(ch_Y_ext) > 1: return None
    #
    #     XY = X_set & Y_set
    #     ex_X, ex_Y = X_set - XY, Y_set - XY
    #
    #     if any(not set(G.successors(x)).issubset(A_set) for x in ex_X): return None
    #     if any(not set(G.predecessors(y)).issubset(A_set) for y in ex_Y): return None
    #
    #     return [{
    #         "vertices": sorted(list(A_set)),
    #         "receivers": sorted(list(X_set)),
    #         "emitters": sorted(list(Y_set))
    #     }]

    def is_transit_component(self,X, Y, A, G):
        # A, X, Y are already lists ordered by master_nodes from the previous steps

        # 1. Check if all receivers (X) have the same external parents
        # R: unique(lapply(pa[X], setdiff, A))
        pa_X_ext = []
        for x in X:
            # Get parents of x that are NOT in A
            ext_parents = [p for p in G.predecessors(x) if p not in A]
            # Standardize order for comparison
            ext_parents.sort(key=lambda n: self.nodes.index(n))
            if ext_parents not in pa_X_ext:
                pa_X_ext.append(ext_parents)

        if len(pa_X_ext) > 1:
            return None

        # 2. Check if all emitters (Y) have the same external children
        # R: unique(lapply(ch[Y], setdiff, A))
        ch_Y_ext = []
        for y in Y:
            # Get children of y that are NOT in A
            ext_children = [c for c in G.successors(y) if c not in A]
            # Standardize order for comparison
            ext_children.sort(key=lambda n: self.nodes.index(n))
            if ext_children not in ch_Y_ext:
                ch_Y_ext.append(ext_children)

        if len(ch_Y_ext) > 1:
            return None

        # 3. Identify exclusive receivers (X not in Y) and exclusive emitters (Y not in X)
        ex_X = [x for x in X if x not in Y]
        ex_Y = [y for y in Y if y not in X]

        # 4. Exclusive receivers must have ALL children inside A
        for x in ex_X:
            if any(child not in A for child in G.successors(x)):
                return None

        # 5. Exclusive emitters must have ALL parents inside A
        for y in ex_Y:
            if any(parent not in A for parent in G.predecessors(y)):
                return None

        # 6. Return the component with lists following the Master Node Order
        # (Since X, Y, and A were passed in as ordered lists, we keep them as is)
        return [{
            "vertices": list(A),
            "receivers": list(X),
            "emitters": list(Y)
        }]

    # def grouped_graph(self, grouping, G=None):
    #     graph = G if G is not None else self.G
    #     if graph is None:
    #         raise ValueError("No graph provided to grouped_graph.")
    #
    #     # These are already R-ordered lists from your transit component discovery
    #     receivers = grouping['receivers']
    #     emitters = grouping['emitters']
    #     component_verts = grouping['vertices']
    #
    #     # 1. Identify External Parents (Parents of receivers not in the component)
    #     pa_ex_raw = []
    #     for r in receivers:
    #         # Get predecessors that are NOT part of the component
    #         pa_ex_raw.extend([p for p in graph.predecessors(r) if p not in component_verts])
    #
    #     # Standardize the order of external parents based on Master List
    #     pa_ex = [n for n in graph.nodes if n in pa_ex_raw]
    #
    #     # 2. Identify External Children (Children of emitters not in the component)
    #     ch_ex_raw = []
    #     for e in emitters:
    #         # Get successors that are NOT part of the component
    #         ch_ex_raw.extend([c for c in graph.successors(e) if c not in component_verts])
    #
    #     # Standardize the order of external children based on Master List
    #     ch_ex = [n for n in graph.nodes if n in ch_ex_raw]
    #
    #     # 3. Create the new graph by removing the component nodes
    #     # We keep all nodes that are NOT in the component_verts list
    #     keep_nodes = [n for n in graph.nodes if n not in component_verts]
    #     grouped = graph.subgraph(keep_nodes).copy()
    #
    #     # 4. Create the Representative (Collapsed) Node
    #     # Use the ordered vertices list to create the name string
    #     representative = "".join(component_verts)
    #     grouped.add_node(representative, description="Collapsed Component")
    #
    #     # 5. Add the "Bridge" Edges to the new representative node
    #     # From external parents TO representative
    #     grouped.add_edges_from([(p, representative) for p in pa_ex])
    #     # From representative TO external children
    #     grouped.add_edges_from([(representative, c) for c in ch_ex])
    #
    #     return grouped

    # def grouped_graph(self, grouping, G=None):
    #     graph = G if G is not None else self.G
    #     component_verts = grouping['vertices']
    #
    #     # 1. Identify External Neighbors (Definition 1)
    #     # Parents of the cluster that are NOT in the cluster
    #     pa_ex = [p for p in graph.nodes if p in graph.predecessors(component_verts) and p not in component_verts]
    #     # Children of the cluster that are NOT in the cluster
    #     ch_ex = [c for c in graph.nodes if c in graph.successors(component_verts) and c not in component_verts]
    #
    #     # 2. Create the Induced Subgraph (Definition 1)
    #     # Remove all vertices in T
    #     keep_nodes = [n for n in graph.nodes if n not in component_verts]
    #     grouped = graph.subgraph(keep_nodes).copy()
    #
    #     # 3. Add the Representative (Definition 1)
    #     representative = "".join(component_verts)
    #     grouped.add_node(representative)
    #
    #     # 4. Re-attach External Edges (Corollary 5)
    #     grouped.add_edges_from([(p, representative) for p in pa_ex])
    #     grouped.add_edges_from([(representative, c) for c in ch_ex])

        # return grouped

    # def grouped_graph(self, grouping, G=None):
    #     """
    #     Implements Definition 1 (Clustering) while strictly preserving
    #     the Master Ruler node order[cite: 128, 160].
    #     """
    #     graph = G if G is not None else self.G
    #
    #     # T is the set of vertices in the transit cluster [cite: 125, 128]
    #     component_verts = grouping['vertices']
    #     receivers = grouping['receivers']
    #     emitters = grouping['emitters']
    #
    #     # 1. Identify External Parents (Pa_G*(T) \ T) [cite: 128, 137]
    #     # We use a list comprehension to act as a filter against the Master Ruler.
    #     # This ensures parents appear in the correct causal sequence.
    #     pa_ex_raw = []
    #     for r in receivers:
    #         pa_ex_raw.extend([p for p in graph.predecessors(r) if p not in component_verts])
    #
    #     # Remove duplicates while maintaining order by filtering the Master Ruler
    #     pa_ex = [n for n in self.nodes if n in pa_ex_raw]
    #
    #     # 2. Identify External Children (Ch_G*(T) \ T) [cite: 128, 140]
    #     ch_ex_raw = []
    #     for e in emitters:
    #         ch_ex_raw.extend([c for c in graph.successors(e) if c not in component_verts])
    #
    #     # Remove duplicates while maintaining order by filtering the Master Ruler
    #     ch_ex = [n for n in self.nodes if n in ch_ex_raw]
    #
    #     # 3. Define the Representative Vertex Label [cite: 128, 160]
    #     # Uses the already-ordered component_verts list
    #     representative = "".join(component_verts)
    #
    #     # 4. Construct the Induced Graph G' [cite: 124, 128]
    #     # Create a fresh DiGraph to ensure total isolation
    #     keep_nodes = [n for n in self.nodes if n in graph.nodes and n not in component_verts]
    #     grouped = nx.DiGraph()
    #
    #     # Add nodes and edges between the remaining nodes [cite: 124]
    #     grouped.add_nodes_from((n, graph.nodes[n]) for n in keep_nodes)
    #     for u, v in graph.edges:
    #         if u in keep_nodes and v in keep_nodes:
    #             grouped.add_edge(u, v)
    #
    #     # 5. Integrate the Representative Node [cite: 128, 160]
    #     grouped.add_node(representative, description="Collapsed Component")
    #
    #     # Add edges from external parents TO the representative [cite: 160]
    #     grouped.add_edges_from([(p, representative) for p in pa_ex])
    #
    #     # Add edges from the representative TO external children [cite: 160]
    #     grouped.add_edges_from([(representative, c) for c in ch_ex])
    #
    #     return grouped

    def grouped_graph(self, grouping, G=None):
        # Use provided graph or the internal one
        g = G if G is not None else self.G

        # R uses 'vertices' for clusters, but your first element uses 'nodes'
        # We check both to match the R behavior perfectly

        v_to_hide = grouping.get('vertices', [])
        if set(v_to_hide) == set(self.nodes):
            v_to_hide = []
        v_receivers = grouping.get('receivers', [])
        v_emitters = grouping.get('emitters', [])

        # R: pa_ex <- setdiff(parents(grouping$receivers, g), grouping$vertices)
        pa_ex_raw = []
        for r in v_receivers:
            # Python fix: predecessors() takes 1 node, so we iterate
            pa_ex_raw.extend([p for p in g.predecessors(r) if p not in v_to_hide])
        # Keep Master Ruler order
        pa_ex = [n for n in self.nodes if n in pa_ex_raw]

        # R: ch_ex <- setdiff(children(grouping$emitters, g), grouping$vertices)
        ch_ex_raw = []
        for e in v_emitters:
            ch_ex_raw.extend([c for c in g.successors(e) if c not in v_to_hide])
        # Keep Master Ruler order
        ch_ex = [n for n in self.nodes if n in ch_ex_raw]

        # R: v <- igraph::V(g)
        # R: keep <- setdiff(v, vertices_in_grouping)
        # If v_to_hide is empty, keep will be the entire graph

        keep = [n for n in self.nodes if n in g.nodes and n not in v_to_hide]

        # R: grouped <- igraph::induced_subgraph(g, keep)
        # Induced subgraph in R is a new graph; .copy() ensures this in Python
        grouped = g.subgraph(keep).copy()

        # R: representative <- paste0(grouping$vertices, collapse = "")
        # representative = "".join(v_to_hide)
        representative = "_".join(v_to_hide)
        # R: grouped <- grouped + vertex(representative)
        grouped.add_node(representative, description="Collapsed Component")

        # R: grouped <- grouped + edges(...)
        for p in pa_ex:
            grouped.add_edge(p, representative)
        for c in ch_ex:
            grouped.add_edge(representative, c)

        return grouped
    def find_transit_clusters(self, G, tgr, nodes):
        """
        Python equivalent of find_transit_clusters R function.
        Iteratively expands base components into larger clusters.
        """
        A = list(tgr)
        B = list(tgr)

        # R's rev(seq_along(tgr)) goes from len down to 1
        for i in range(len(tgr) - 1, -1, -1):
            B.pop(i)  # Remove the current element for the next comparison
            # Expand the current component against remaining components in B
            expanded = self.expand_cluster(tgr[i], A, B, G, nodes)
            A.extend(expanded)

        # Deduplicate the list of dictionaries
        return self.unique_clusters(A)


    def expand_cluster(self, t, A, B, G,nodes):
        if nodes is None:
            nodes = self.nodes
        B_prime = list(B)
        # Use lists for existence checks to avoid set hashing
        A_v_lists = [item["vertices"] for item in A]
        current_A = []

        for i in range(len(B) - 1, -1, -1):
            B_prime.pop(i)
            s = B[i]

            # Combine vertices and restore R-order via Master List
            st_v_raw = s["vertices"] + t["vertices"]
            st_vertices = [n for n in nodes if n in st_v_raw]

            # Check if this exact ordered list already exists in A
            if st_vertices not in A_v_lists:
                # --- EXTERNAL INTERFACE CHECK ---

                # Parents of S receivers NOT in S receivers
                pa_re_S = [p for node in s["receivers"] for p in G.predecessors(node) if p not in s["receivers"]]
                pa_re_S_ordered = [n for n in nodes if n in pa_re_S]

                # Parents of T receivers NOT in T receivers
                pa_re_T = [p for node in t["receivers"] for p in G.predecessors(node) if p not in t["receivers"]]
                pa_re_T_ordered = [n for n in nodes if n in pa_re_T]

                # Children of S emitters NOT in S emitters
                ch_em_S = [c for node in s["emitters"] for c in G.successors(node) if c not in s["emitters"]]
                ch_em_S_ordered = [n for n in nodes if n in ch_em_S]

                # Children of T emitters NOT in T emitters
                ch_em_T = [c for node in t["emitters"] for c in G.successors(node) if c not in t["emitters"]]
                ch_em_T_ordered = [n for n in nodes if n in ch_em_T]

                # Compare ordered interfaces
                if pa_re_S_ordered == pa_re_T_ordered and ch_em_S_ordered == ch_em_T_ordered:
                    # Merge and Re-order
                    st_clust = {
                        "vertices": st_vertices,
                        "receivers": [n for n in nodes if n in s["receivers"] or n in t["receivers"]],
                        "emitters": [n for n in nodes if n in s["emitters"] or n in t["emitters"]]
                    }

                    current_A.append(st_clust)
                    A_v_lists.append(st_vertices)

                    recursive_exp = self.expand_cluster(st_clust, A + current_A, B_prime, G, nodes)
                    current_A.extend(recursive_exp)

        return current_A


    def unique_clusters(self, clusters):
        unique = []
        for c in clusters:
            is_duplicate = False
            for u in unique:
                if (u["vertices"] == c["vertices"] and
                        u["receivers"] == c["receivers"] and
                        u["emitters"] == c["emitters"]):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(c)
        return unique



    def cluster_edge_subgraph(self, G, incoming, outgoing):
        # 1. Identify edges to remove using list comprehension
        # This captures (u -> v) where v is incoming or u is outgoing
        edges_to_remove = [
            (u, v) for u, v in G.edges()
            if v in incoming or u in outgoing
        ]

        # 2. Create copy and remove
        H = G.copy()
        H.remove_edges_from(edges_to_remove)

        return H

    def get_bfs_nodes(self, graph, start_node, mode='out'):
        # This mirrors R's igraph::neighborhood behavior
        # mode='out' for successors/descendants, mode='in' for predecessors/ancestors
        edges = graph.out_edges if mode == 'out' else graph.in_edges

        visited = []
        queue = [start_node]
        seen = {start_node}

        while queue:
            curr = queue.pop(0)
            visited.append(curr)
            # Sort neighbors alphabetically ONLY so discovery is consistent
            neighbors = sorted([v if mode == 'out' else u for u, v in edges(curr)])
            for n in neighbors:
                if n not in seen:
                    seen.add(n)
                    queue.append(n)
        return visited

