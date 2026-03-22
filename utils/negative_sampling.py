import random


def sample_hard_negatives(graph, num_samples):
    """
    2-hop hard negative sampling with fallback to random non-edges.
    """
    negatives    = set()
    nodes        = list(graph.nodes())
    max_attempts = num_samples * 100
    attempts     = 0

    while len(negatives) < num_samples and attempts < max_attempts:
        attempts += 1
        u  = random.choice(nodes)
        Nu = set(graph.neighbors(u))
        if not Nu:
            continue
        two_hop = set()
        for n in Nu:
            two_hop.update(graph.neighbors(n))
        two_hop.discard(u)
        two_hop -= Nu
        if not two_hop:
            continue
        v = random.choice(list(two_hop))
        if not graph.has_edge(u, v):
            negatives.add((u, v))

    if len(negatives) < num_samples:
        existing = set(graph.edges())
        while len(negatives) < num_samples:
            u, v = random.sample(nodes, 2)
            if (u, v) not in existing and (v, u) not in existing:
                negatives.add((u, v))

    return list(negatives)