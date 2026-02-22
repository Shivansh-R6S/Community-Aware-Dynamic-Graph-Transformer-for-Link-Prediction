import pandas as pd
import networkx as nx


def build_rolling_cumulative_snapshots(
    df,
    num_nodes,
    window_size_days=30,
    step_size_days=15,
    memory_days=60
):
    """
    Build rolling window + cumulative memory snapshots.

    Each snapshot includes:
    [current_start - memory_days, current_start + window_size_days]

    Returns:
        List[nx.Graph]
    """

    print("\nBuilding Rolling + Cumulative Snapshots...")

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    start_time = df["datetime"].min()
    end_time = df["datetime"].max()

    print("Start time:", start_time)
    print("End time:", end_time)

    snapshots = []
    current_start = start_time

    snapshot_id = 1

    while current_start < end_time:

        window_end = current_start + pd.Timedelta(days=window_size_days)
        memory_start = current_start - pd.Timedelta(days=memory_days)

        if memory_start < start_time:
            memory_start = start_time

        mask = (
            (df["datetime"] >= memory_start) &
            (df["datetime"] < window_end)
        )

        window_edges = df[mask]

        if len(window_edges) > 0:

            G = nx.Graph()
            G.add_nodes_from(range(num_nodes))

            edges = window_edges[["source", "target"]].values
            G.add_edges_from(edges)

            snapshots.append(G)

            print(
                f"Snapshot {snapshot_id} | "
                f"Nodes: {G.number_of_nodes()} | "
                f"Edges: {G.number_of_edges()}"
            )

            snapshot_id += 1

        current_start += pd.Timedelta(days=step_size_days)

    print(f"\nTotal snapshots created: {len(snapshots)}")

    return snapshots


if __name__ == "__main__":

    from data_loader import load_fb_forum
    import os

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "data", "raw", "fb-forum.txt")

    df, node_mapping = load_fb_forum(data_path)
    num_nodes = len(node_mapping)

    snapshots = build_rolling_cumulative_snapshots(
        df,
        num_nodes,
        window_size_days=30,
        step_size_days=15,
        memory_days=60
    )

    print("\nSnapshot Summary:")
    for i, snap in enumerate(snapshots[:5]):
        print(
            f"Snapshot {i+1}: "
            f"Nodes={snap.number_of_nodes()}, "
            f"Edges={snap.number_of_edges()}"
        )
