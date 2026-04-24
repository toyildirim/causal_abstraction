from collections import deque
import networkx as nx
import numpy as np


class PartitionDagModelKnownDag(object):
    """A class to abstract a known DAG by partitioning nodes deterministically."""

    def __init__(self) -> None:
        self.dag = nx.DiGraph()

    def fit(self, dag) -> object:
        # Extract order and adj from the input DAG
        self.input_dag = dag
        nodes = list(dag.nodes())
        if not all(isinstance(n, int) for n in nodes):
            # Map nodes to integers if necessary
            node_map = {node: i for i, node in enumerate(sorted(nodes))}
            self.node_map = node_map
            self.reverse_map = {v: k for k, v in node_map.items()}
            mapped_dag = nx.relabel_nodes(dag, node_map)
            self.order = list(range(mapped_dag.number_of_nodes()))
            self.adj = nx.to_numpy_array(mapped_dag)
        else:
            self.order = list(range(dag.number_of_nodes()))
            self.adj = nx.to_numpy_array(dag)
            self.node_map = None

        # always use lexicographical order for node partitions
        init_partition = [set(self.order)]  # trivial coarsening
        self.dag.add_node(tuple(sorted(self.order)))
        self.refinable = deque(init_partition)
        while len(self.refinable) > 0:
            self._recurse()
        return self

    def _refine(self):
        to_refine = self.refinable.popleft()
        # Sort the set according to the order for deterministic splitting
        sorted_to_refine = sorted(to_refine, key=lambda x: self.order.index(x))
        u_len = len(sorted_to_refine) // 2
        u = set(sorted_to_refine[:u_len])
        v = set(sorted_to_refine[u_len:])
        if len(u) > 1:
            self.refinable.append(u)
        if len(v) > 1:
            self.refinable.append(v)
        return to_refine, u, v

    def _is_adj(self, pa, ch):
        for el_pa in pa:
            for el_ch in ch:
                if el_ch in self.adj[el_pa]:
                    return True
        return False

    def _recurse(self):
        to_refine, u, v = self._refine()
        if not u or not v:
            return
        self.dag.add_node(tuple(u))
        self.dag.add_node(tuple(v))
        for pa in self.dag.predecessors(tuple(to_refine)):
            for ch in (u, v):
                if self._is_adj(set(pa), ch):
                    self.dag.add_edge(pa, tuple(ch))
        for pa in (u, v):
            for ch in self.dag.successors(tuple(to_refine)):
                if self._is_adj(pa, set(ch)):
                    self.dag.add_edge(tuple(pa), ch)
        if self._is_adj(u, v):
            self.dag.add_edge(tuple(u), tuple(v))
        self.dag.remove_node(tuple(to_refine))
