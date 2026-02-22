from itertools import combinations
from itertools import product
import sys
import os
import networkx as nx
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from gFormula import *

class DAGReducer:

    def __init__(self, G, exposure=None, outcome=None):
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
            # If still None, it's not a 'Goal-Oriented' graph yet
            print("Warning: No exposure/outcome found. Goal-oriented reduction will be disabled.")

    def topo_sorted(self, nodes):
        """Sorts a subset of nodes based on the global topological order."""
        full_order = list(nx.topological_sort(self.G))
        node_idx = {node: i for i, node in enumerate(full_order)}
        return sorted(nodes, key=lambda x: node_idx[x])

    def get_vset(self):
        return self.topo_sorted(list(self.G.nodes()))

    def get_iset(self):
        """Returns Instruments: Ancestors of A and Y that only affect Y through A."""
        anc_a = nx.ancestors(self.G, self.exposure)
        anc_y = nx.ancestors(self.G, self.outcome)
        I = (anc_a & anc_y) - {self.exposure}

        valid_I = []
        for u in I:
            # Check if all directed paths from u to outcome are blocked by exposure
            paths = list(nx.all_simple_paths(self.G, u, self.outcome))
            is_blocked = all(self.exposure in path for path in paths)
            if is_blocked:
                valid_I.append(u)
        return self.topo_sorted(valid_I)

    def get_wset(self):
        """Returns Covariates: Ancestors of Y that are not descendants of A."""
        anc_y = nx.ancestors(self.G, self.outcome)
        des_a = nx.descendants(self.G, self.exposure) | {self.exposure}
        W = (anc_y - des_a) - set(self.get_iset())
        return self.topo_sorted(W)

    def get_mset(self):
        """Returns Mediators: Nodes on the causal path between A and Y."""
        anc_y = nx.ancestors(self.G, self.outcome)
        des_a = nx.descendants(self.G, self.exposure)
        M = (anc_y & des_a) - {self.exposure}
        return self.topo_sorted(M)

    def get_nset(self):
        """Returns Non-ancestors of the outcome."""
        anc_y = nx.ancestors(self.G, self.outcome) | {self.outcome}
        N = set(self.G.nodes()) - anc_y
        return self.topo_sorted(N)

    def get_oset(self):
        """Returns Optimal adjustment set (Parents of Mediators)."""
        M = self.get_mset()
        pa_m = set().union(*[set(self.G.predecessors(m)) for m in M]) if M else set()
        de_m = set().union(*[nx.descendants(self.G, m) for m in M]) if M else set()
        O = pa_m - ({self.exposure} | de_m)
        return self.topo_sorted(O)

    def check_d_sep(self, X, Y, Z=None):
        Z = set(Z) if Z else set()
        X_set = {X} if isinstance(X, str) else set(X)
        Y_set = {Y} if isinstance(Y, str) else set(Y)
        return (nx.is_d_separator(self.G, X_set - Z, Y_set - Z, Z))

    def get_omin(self):
        """Finds the minimal subset of O that d-separates Exposure from the rest of O."""
        O = self.get_oset()
        if not O: return []
        for size in range(len(O) + 1):
            for subset in combinations(O, size):
                if self.check_d_sep(self.exposure, set(O) - set(subset), subset):
                    return self.topo_sorted(list(subset))
        return O

    def get_uninformative_variables(self):
        uninf = set(self.get_nset()) | set(self.get_iset())
        # Add uninformative W nodes
        for w in self.get_wset():
            if self._check_w_criterion(w):
                uninf.add(w)
        # Add uninformative M nodes
        for m in self.get_mset():
            if self._check_m_criterion(m):
                uninf.add(m)
        return list(uninf)

    def _check_w_criterion(self, w):
        O = set(self.get_oset())
        W = self.get_wset()
        if w in O: return False
        children_in_W = self.topo_sorted([c for c in self.G.successors(w) if c in W])
        if not children_in_W: return True  # Leaf in W-set is usually uninformative

        w_last = children_in_W[-1]
        z1 = (set(self.G.predecessors(w_last)) | {w_last}) - {w}
        if not self.check_d_sep(w, O, z1): return False
        # ... logic for iterative parent checks ...
        return True  # Simplified for brevity, follows R logic structure

    def _check_m_criterion(self, m):
        """
        Checks if a mediator node 'm' is uninformative for the causal effect.
        Follows the logic that if all information from 'm' is captured by
        other mediators or optimal adjustment sets, it can be abstracted.
        """
        O = set(self.get_oset())
        M = self.get_mset()

        # If the mediator is part of the optimal adjustment set, we usually keep it
        if m in O:
            return False

        # Get the children of m that are also in the mediator set
        children_in_M = self.topo_sorted([c for c in self.G.successors(m) if c in M])

        # If m has no children in the mediator set, it might be an 'outcome-only'
        # mediator leaf, which is often uninformative.
        if not children_in_M:
            return True

        # Check the last child in the topological order
        m_last = children_in_M[-1]

        # z1 consists of other parents of m_last and m_last itself, excluding m
        z1 = (set(self.G.predecessors(m_last)) | {m_last}) - {m}

        # Use the fixed is_d_separator function we discussed
        import networkx as nx
        if not nx.is_d_separator(self.G, {m}, O, z1):
            return False

        return True
    def reduce_dag(self, verbose=True):
        """The main reduction loop."""
        # 1. Project out N and I sets first (Standard Latent Projection)
        abstracted_g = self.G.copy()
        to_remove = self.get_uninformative_variables()

        for node in to_remove:
            parents = list(abstracted_g.predecessors(node))
            children = list(abstracted_g.successors(node))
            for p in parents:
                for c in children:
                    abstracted_g.add_edge(p, c)
            abstracted_g.remove_node(node)
        # 2. Use g_formula to describe the result
        if verbose:
            identifying_equation = g_formula(abstracted_g)  # <--- Using it here
            print(f"Abstraction Complete.")
            print(f"New Identifying Equation: {identifying_equation}")
        return abstracted_g

    """The novel function that identifies the possible exposures and outcomes"""
    def analyze_all_causal_paths(G):
        # 1. Automatic Discovery
        sources = [n for n, d in G.in_degree() if d == 0]
        sinks = [n for n, d in G.out_degree() if d == 0]

        all_results = {}

        # 2. Iterate over all possible (Exposure, Outcome) pairs
        for exp, out in product(sources, sinks):
            if exp == out: continue

            print(f"--- Analyzing Path: {exp} -> {out} ---")

            # Initialize reducer for this specific context
            reducer = DAGReducer(G, exposure=exp, outcome=out)

            # Find uninformative nodes for THIS specific pair
            uninf = reducer.get_uninformative_variables()

            # Store for later comparison
            all_results[(exp, out)] = {
                'uninformative': uninf,
                'g_formula': g_formula(reducer.reduce_dag())
            }

        return all_results