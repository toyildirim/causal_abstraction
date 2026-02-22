import networkx as nx

def g_formula(g):
    """
    Python implementation of the g-formula for identifying the counterfactual mean.

    Args:
        g (nx.DiGraph): A directed acyclic graph where nodes have 'exposure'
                        or 'outcome' labels in their attributes, or are identified
                        via separate lists.
    Returns:
        str: A character string representing the causal formula.
    """
    # 1. Identify Exposure (A) and Outcome (Y)
    # Based on your previous context, we'll assume they are stored in graph attributes
    # or you can pass them as arguments.
    A = [n for n, d in g.nodes(data=True) if 'exposure' in d]
    Y = [n for n, d in g.nodes(data=True) if 'outcome' in d]

    if not A or not Y:
        # Fallback if labels aren't set: assume specific names or prompt user
        raise ValueError("Graph must have nodes labeled 'exposure' and 'outcome'")

    # G-formula identifies the joint distribution minus the intervention variable
    V = list(nx.topological_sort(g))
    V_no_A = [v for v in V if v not in A]

    # We need to build a factor matrix/dependency tracker similar to the R logic
    # to determine where the summation (Σ) symbols should be placed.
    factors = {v: set([v]) | set(g.predecessors(v)) for v in V_no_A}

    formula_terms = []

    # Iterate through variables in reverse topological order (inner to outer)
    for v in reversed(V_no_A):
        parents = list(g.predecessors(v))

        # Replace the exposure variable with the fixed intervention value 'a'
        processed_parents = ["A=a" if p in A else str(p) for p in parents]

        if processed_parents:
            term = f"P({v} | {', '.join(processed_parents)})"
        else:
            term = f"P({v})"

        # If this is the outcome node, format it specifically
        if v in Y:
            term = f"{v} {term.replace('P(', 'P[')}"
            term = term.replace(')', ']')

        # Logic for Summation Σ: In G-formula, we sum over all non-outcome,
        # non-exposure variables to marginalize the distribution.
        if v not in Y:
            term = f"∑_{{{v}}} {term}"

        formula_terms.append(term)

    # Join terms together (G-formula is a product of these terms)
    return " ".join(reversed(formula_terms))