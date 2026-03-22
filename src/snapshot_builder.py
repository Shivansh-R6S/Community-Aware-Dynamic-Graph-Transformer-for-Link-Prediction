import pandas as pd
import networkx as nx


def build_rolling_cumulative_snapshots(
    df, num_nodes,
    window_size_days=30, step_size_days=15, memory_days=60
):
    print("\nBuilding snapshots...")
    df["datetime"]  = pd.to_datetime(df["timestamp"], unit="s")
    start_time      = df["datetime"].min()
    end_time        = df["datetime"].max()
    snapshots       = []
    current_start   = start_time
    sid             = 1

    while current_start < end_time:
        window_end   = current_start + pd.Timedelta(days=window_size_days)
        memory_start = max(start_time, current_start - pd.Timedelta(days=memory_days))
        mask         = (df["datetime"] >= memory_start) & (df["datetime"] < window_end)
        edges        = df[mask]
        if len(edges) > 0:
            G = nx.Graph()
            G.add_nodes_from(range(num_nodes))
            G.add_edges_from(edges[["source", "target"]].values)
            snapshots.append(G)
            print(f"  Snapshot {sid:02d} | Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
            sid += 1
        current_start += pd.Timedelta(days=step_size_days)

    print(f"\nTotal snapshots: {len(snapshots)}")
    return snapshots