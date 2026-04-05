import os
import sys
from itertools import combinations, product

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from gFormula import *

class DAGReducer:
    """
    A comprehensive Python implementation of the reduce DAG algorithm,
    ported from the R dagitty/causaleffect packages.

    This class provides methods to identify uninformative variables in a DAG
    and reduce it to a minimal set of informative variables.
    """

    def __init__(self, G, exposure=None, outcome=None):
        """
        Initialize the DAGReducer.

        Args:
            G: A NetworkX DiGraph representing the causal DAG
            exposure: The exposure/treatment node (or its label attribute)
            outcome: The outcome node (or its label attribute)
        """
        self.G = G

        # 1. If not provided, try to find them in the 'label' attribute
        if exposure is None:
            exposure_nodes = [n for n, d in G.nodes(data=True) if d.get('label') == 'exposure']
            self.exposure = exposure_nodes[0] if exposure_nodes else None
        else:
            self.exposure = exposure

        if outcome is None:
            outcome_nodes = [n for n, d in G.nodes(data=True) if d.get('label') == 'outcome']
            self.outcome = outcome_nodes[0] if outcome_nodes else None
        else:
            self.outcome = outcome

        # 2. Validation
        if not self.exposure or not self.outcome:
            print("Warning: No exposure/outcome found. Goal-oriented reduction will be disabled.")

        print(f"\nVariable Sets:")
        print(f"  V-set: {self.get_vset()}")
        print(f"  I-set: {self.get_iset()}")
        print(f"  W-set: {self.get_wset()}")
        print(f"  M-set: {self.get_mset()}")
        print(f"  N-set: {self.get_nset()}")

    def topo_sorted(self, nodes):
        """
        Topologically sorts a subset of nodes based on the global topological order.

        Args:
            nodes: List or set of nodes to sort

        Returns:
            List of nodes sorted in topological order
        """
        if not nodes:
            return []
        full_order = list(nx.topological_sort(self.G))
        node_idx = {node: i for i, node in enumerate(full_order)}
        return sorted(nodes, key=lambda x: node_idx[x])

    def get_vset(self):
        """
        Returns the complete vertex set sorted topologically.

        Returns:
            Sorted list of all nodes in the graph
        """
        return self.topo_sorted(list(self.G.nodes()))

    def get_iset2(self):
        """
        Returns the set of instruments (I-set).

        Instruments are ancestors of both A and Y that are d-separated from Y
        when conditioning on A (i.e., they only affect Y through A).

        Returns:
            Sorted list of instrument nodes
        """
        anc_a = nx.ancestors(self.G, self.exposure) if self.exposure in self.G else set()
        anc_y = nx.ancestors(self.G, self.outcome) if self.outcome in self.G else set()
        I = (anc_a & anc_y) - {self.exposure}

        valid_I = []
        for u in I:
            # Check if u is d-separated from Y when conditioning on A
            # If u is d-separated from Y | A, then u is an instrument
            # This means all paths from u to Y must go through A
            if self.check_d_sep(u, self.outcome, {self.exposure}):
                valid_I.append(u)
        return self.topo_sorted(valid_I)

    def get_iset(self):
        """
        Returns the set of instruments (I-set) identical to the R implementation.
        Checks if ancestors are 'intersected by A' using directed reachability.
        """
        if self.exposure not in self.G or self.outcome not in self.G:
            return []

        anc_a = (nx.ancestors(self.G, self.exposure) | {self.exposure})
        anc_y = (nx.ancestors(self.G, self.outcome) | {self.outcome})

        # .I is the intersection of ancestors of A and Y
        candidate_I = (anc_a & anc_y) - {self.exposure}

        # Use a subgraph view to avoid memory-heavy G.copy()
        # This view virtually removes the exposure node
        view_without_A = nx.subgraph_view(self.G, filter_node=lambda n: n != self.exposure)

        valid_I = []
        for u in candidate_I:
            # R logic: !any(dagitty::paths(g, u, Y, Z=A, directed=TRUE)$open)
            # Python equivalent: Is Y unreachable from u if A is removed?
            if not nx.has_path(view_without_A, u, self.outcome):
                valid_I.append(u)

        return self.topo_sorted(valid_I)
    def get_wset(self):
        """
        Returns the W-set (covariates).

        Covariates are ancestors of Y that are not descendants of A and not instruments.

        Returns:
            Sorted list of covariate nodes
        """
        anc_y = (nx.ancestors(self.G, self.outcome) | {self.outcome}) if self.outcome in self.G else set()
        des_a = (nx.descendants(self.G, self.exposure) | {self.exposure}) if self.exposure in self.G else set()
        W = (anc_y - des_a) - set(self.get_iset())
        return self.topo_sorted(W)

    def get_mset(self):
        """
        Returns the M-set (mediators).

        Mediators are nodes on causal paths between A and Y.

        Returns:
            Sorted list of mediator nodes
        """
        anc_y = (nx.ancestors(self.G, self.outcome) | {self.outcome}) if self.outcome in self.G else set()
        des_a = (nx.descendants(self.G, self.exposure) | {self.exposure}) if self.exposure in self.G else set()
        M = (anc_y & des_a) - {self.exposure}
        return self.topo_sorted(M)

    def get_nset(self):
        """
        Returns the N-set (non-ancestors of outcome).

        Non-ancestors are nodes that do not have a path to the outcome.

        Returns:
            Sorted list of non-ancestor nodes
        """
        anc_y = (nx.ancestors(self.G, self.outcome) | {self.outcome}) if self.outcome in self.G else set()
        N = set(self.G.nodes()) - anc_y
        return self.topo_sorted(N)

    def get_oset(self):
        """
        Returns the O-set (optimal adjustment set).

        The O-set consists of parents of mediators that are not the exposure
        and not descendants of mediators.

        Returns:
            Sorted list of optimal adjustment set nodes
        """
        M = self.get_mset()
        if M:
            pa_m = set().union(*[set(self.G.predecessors(m)) for m in M])
            de_m = set().union(*[nx.descendants(self.G, m) for m in M])
        else:
            pa_m = set()
            de_m = set()
        O = pa_m - ({self.exposure} | de_m)
        return self.topo_sorted(O)

    def check_d_sep(self, X, Y, Z=None):
        """
        Checks d-separation between X and Y conditioned on Z.

        Args:
            X: Single node or list of nodes
            Y: Single node or list of nodes
            Z: Conditioning set (list of nodes or None)

        Returns:
            Boolean indicating if X and Y are d-separated given Z
        """
        Z = set(Z) if Z else set()
        X_set = {X} if isinstance(X, str) else set(X)
        Y_set = {Y} if isinstance(Y, str) else set(Y)
        return nx.is_d_separator(self.G, X_set - Z, Y_set - Z, Z)

    def get_omin(self):
        """
        Finds the minimal adjustment set (O-min).

        This is the minimal subset of O that d-separates the exposure from
        the rest of O.

        Returns:
            Sorted list of nodes in the minimal adjustment set
        """
        O = self.get_oset()
        if not O:
            return []
        for size in range(len(O) + 1):
            for subset in combinations(O, size):
                if self.check_d_sep(self.exposure, set(O) - set(subset), subset):
                    return self.topo_sorted(list(subset))
        return O

    def get_uninformative_variables(self):
        """
        Identifies all uninformative variables in the DAG.

        A variable is uninformative if removing it doesn't affect the causal
        effect estimation between exposure and outcome.

        Returns:
            List of uninformative variable nodes
        """
        uninf = set(self.get_nset()) | set(self.get_iset())

        # Check W-set nodes for uninformativeness
        for w in self.get_wset():
            if self._check_w_criterion(w):
                uninf.add(w)

        # Check M-set nodes for uninformativeness
        for m in self.get_mset():
            if self._check_m_criterion(m):
                uninf.add(m)

        return list(uninf)

    def get_informative_variables(self):
        """
        Returns all informative variables (complement of uninformative).

        Returns:
            List of informative variable nodes
        """
        return [v for v in self.get_vset() if v not in self.get_uninformative_variables()]

    def _check_w_criterion(self, w):
        """
        Checks if a W-node (covariate) is uninformative.

        Fully implements R's check.W.criterion function.

        Args:
            w: The node to check

        Returns:
            Boolean indicating if w is uninformative
        """
        O = set(self.get_oset())
        W = set(self.get_wset())

        # If w is in the optimal adjustment set, it's always informative
        if w in O:
            return False

        # If O-set is empty, w is informative (it's part of required confounding adjustment)
        if not O:
            return False

        # Get children of w that are also in W
        children_in_W = self.topo_sorted([c for c in self.G.successors(w) if c in W])
        if not children_in_W:
            # Leaf node: Check if it has any effect on outcome
            # If it doesn't connect to O through any path, it's uninformative
            return True

        # Check the last child in topological order
        w_last_ch = children_in_W[-1]
        z1 = (set(self.G.predecessors(w_last_ch)) | {w_last_ch}) - {w}

        if not self.check_d_sep(w, O, z1):
            return False

        # Run iterative checks for all children
        return self._run_iterative_checks(w, children_in_W, O)

    def _check_m_criterion(self, m):
        """
        Checks if an M-node (mediator) is uninformative.

        Fully implements R's check.M.criterion function.

        Args:
            m: The mediator node to check

        Returns:
            Boolean indicating if m is uninformative
        """
        Omin = set(self.get_omin())
        # Target set S = {A, Y, Omin}
        S = {self.exposure, self.outcome} | Omin
        M = set(self.get_mset())

        # Outcome node is always informative
        if m == self.outcome:
            return False

        # If m has direct path to outcome (A -> M -> Y), keep it
        # A mediator on the path from A to Y is informative
        if m in M and self.exposure in nx.ancestors(self.G, m):
            # Check if there's a path from m to outcome
            try:
                path = nx.shortest_path(self.G, m, self.outcome)
                if len(path) > 1:  # There's an actual path
                    return False  # Keep mediators on causal path
            except nx.NetworkXNoPath:
                pass

        # Get children of m that are also in M
        children_in_M = self.topo_sorted([c for c in self.G.successors(m) if c in M])
        if not children_in_M:
            return True  # Leaf mediators might be uninformative

        # Check the last child in topological order
        m_last_ch = children_in_M[-1]
        z1 = (set(self.G.predecessors(m_last_ch)) | {m_last_ch}) - {m}

        if not self.check_d_sep(m, S, z1):
            return False

        # Run iterative checks for all children against target set S
        return self._run_iterative_checks(m, children_in_M, S)

    def _run_iterative_checks2(self, node, children_in_set, target_set):
        """
        Implements the iterative checks: conditions i, ii, and iii from R code.

        Args:
            node: The parent node being checked
            children_in_set: List of children in topological order
            target_set: The conditioning/target set

        Returns:
            Boolean indicating if all conditions are satisfied
        """
        for t in range(len(children_in_set)):
            cur = children_in_set[t]
            prev = node if t == 0 else children_in_set[t - 1]

            pa_cur = set(self.G.predecessors(cur))
            pa_prev = set(self.G.predecessors(prev))

            # Condition i: Previous node is a parent of current child
            cond_i = prev in pa_cur

            # Condition ii: Current parents are a subset of (prev parents + prev)
            cond_ii = pa_cur.issubset(pa_prev | {prev})

            # Condition iii: Difference in parents is d-separated from target by pa_cur
            diff_set = pa_prev - pa_cur
            cond_iii = self.check_d_sep(list(diff_set), target_set, list(pa_cur))

            if not (cond_i and cond_ii and cond_iii):
                return False

        return True

    def _run_iterative_checks(self, node, children_in_set, target_set):
        # Ensure target_set is a list for consistency with your check_d_sep
        targets = list(target_set) if isinstance(target_set, (set, list)) else [target_set]

        for t in range(len(children_in_set)):
            cur = children_in_set[t]
            prev = node if t == 0 else children_in_set[t - 1]

            pa_cur = set(self.G.predecessors(cur))
            pa_prev = set(self.G.predecessors(prev))

            # Condition i: prev -> cur exists
            cond_i = prev in pa_cur

            # Condition ii: pa(cur) ⊆ {pa(prev) ∪ prev}
            # This ensures no 'new' information enters the chain
            cond_ii = pa_cur.issubset(pa_prev | {prev})

            # Condition iii: (pa(prev) \ pa(cur)) ⊥ target | pa(cur)
            diff_set = pa_prev - pa_cur

            if not diff_set:
                # If no parents were removed, the independence is vacuously true
                cond_iii = True
            else:
                # Check d-separation for the nodes being 'dropped'
                cond_iii = self.check_d_sep(list(diff_set), targets, list(pa_cur))

            if not (cond_i and cond_ii and cond_iii):
                return False

        return True
    def _project_out(self, node_to_remove):
        """
        Removes a node from the graph via latent projection.
        Connects all parents to all children and removes the node.

        Args:
            node_to_remove: The node to project out

        Returns:
            Modified graph (modifies in-place on self.G)
        """
        if node_to_remove not in self.G:
            return self.G

        parents = list(self.G.predecessors(node_to_remove))
        children = list(self.G.successors(node_to_remove))

        # Connect all parents to all children
        for p in parents:
            for c in children:
                self.G.add_edge(p, c)

        # Remove the node
        self.G.remove_node(node_to_remove)
        return self.G

    def _project_out_n_and_i(self):
        """
        Implements R's project.out.N.and.I() function.

        First removes N-set (non-ancestors), then iteratively removes I-set.
        Updates self.G in place.
        """
        # First, remove all N-set nodes (non-ancestors of outcome)
        N = self.get_nset()
        for node in N:
            if node in self.G:
                self._project_out(node)

        # Then, iteratively remove I-set nodes
        I = self.get_iset()
        while I:
            node = I[0]  # Remove first element (or could be last)
            if node in self.G:
                self._project_out(node)
            # Recalculate I-set after removing each node
            I = self.get_iset()

        return self.G

    def reduce_dag(self, verbose=True):
        """
        Performs the main DAG reduction algorithm.

        Removes all uninformative variables and creates a new DAG where all
        remaining variables are informative.

        Args:
            verbose: If True, prints reduction details and identifying equation

        Returns:
            Reduced NetworkX DiGraph
        """
        # Step 0: Report uninformative variables BEFORE any modifications (like R does)
        if verbose:
            original_uninf = self.get_uninformative_variables()
            if len(original_uninf) == 0:
                print("All variables are informative.\n")
            else:
                print(f"Uninformative variables {{" +
                      ", ".join(str(v) for v in original_uninf) +
                      "}} are eliminated.\n")

        # Create a copy to avoid modifying the original
        abstracted_g = self.G.copy()

        # Create a temporary reducer for the working graph
        temp_reducer = DAGReducer(abstracted_g, exposure=self.exposure, outcome=self.outcome)

        # Step 1: Project out N-set and I-set first (as in R's project.out.N.and.I)
        temp_reducer._project_out_n_and_i()

        # Step 2: Get uninformative variables from the modified graph
        V_uninf = temp_reducer.get_uninformative_variables()

        # Step 3: Remove remaining uninformative variables via latent projection
        for node in V_uninf:
            if node in abstracted_g:
                parents = list(abstracted_g.predecessors(node))
                children = list(abstracted_g.successors(node))

                # Connect all parents to all children
                for p in parents:
                    for c in children:
                        abstracted_g.add_edge(p, c)

                # Remove the node
                abstracted_g.remove_node(node)

        # Generate and print results if verbose
        if verbose:

            try:
                identifying_equation = g_formula(abstracted_g)
                print(f"Reduced g-formula:")
                print(f"{identifying_equation}\n")
            except Exception as e:
                print(f"Could not generate g-formula: {e}\n")

        return abstracted_g

    @staticmethod
    def analyze_all_causal_paths(G):
        """
        Analyzes all possible causal paths in a DAG.

        Automatically discovers sources (nodes with in-degree 0) and sinks
        (nodes with out-degree 0), then analyzes each source->sink pair.

        Args:
            G: A NetworkX DiGraph

        Returns:
            Dictionary mapping (exposure, outcome) pairs to their analysis results
        """
        # Find all possible exposures (sources) and outcomes (sinks)
        sources = [n for n, d in G.in_degree() if d == 0]
        sinks = [n for n, d in G.out_degree() if d == 0]

        all_results = {}

        # Iterate over all possible (Exposure, Outcome) pairs
        for exp, out in product(sources, sinks):
            if exp == out:
                continue

            print(f"--- Analyzing Path: {exp} -> {out} ---")

            # Initialize reducer for this specific context
            reducer = DAGReducer(G, exposure=exp, outcome=out)

            # Find uninformative nodes for THIS specific pair
            uninf = reducer.get_uninformative_variables()

            # Store for later comparison
            try:
                reduced = reducer.reduce_dag(verbose=False)
                g_eq = g_formula(reduced)
            except Exception as e:
                g_eq = f"Error: {e}"

            all_results[(exp, out)] = {
                'uninformative': uninf,
                'g_formula': g_eq
            }

        return all_results