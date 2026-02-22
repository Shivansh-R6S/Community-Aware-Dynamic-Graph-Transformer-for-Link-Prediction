import os
import pandas as pd


def load_fb_forum(data_path):
    """
    Load FB-Forum dataset.
    Format: source target timestamp (whitespace separated)
    """

    print("Loading FB-Forum dataset...")

    df = pd.read_csv(
        data_path,
        sep=r"\s+",
        engine="python",
        header=None,
        names=["source", "target", "timestamp"]
    )

    print(f"Raw edges: {len(df)}")

    # Remove self-loops
    df = df[df["source"] != df["target"]]

    # Sort by time
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Edges after cleaning: {len(df)}")

    # Map node IDs to continuous indices
    unique_nodes = pd.unique(df[["source", "target"]].values.ravel())
    node_mapping = {node: idx for idx, node in enumerate(unique_nodes)}

    df["source"] = df["source"].map(node_mapping)
    df["target"] = df["target"].map(node_mapping)

    print(f"Total unique nodes: {len(unique_nodes)}")
    print("Timestamp sorted?:", df["timestamp"].is_monotonic_increasing)

    return df, node_mapping


if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "data", "raw", "fb-forum.txt")

    df, node_mapping = load_fb_forum(data_path)

    print("\nFirst 5 rows:")
    print(df.head())
