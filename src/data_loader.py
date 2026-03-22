import os
import pandas as pd


def load_fb_forum(data_path):
    print("Loading FB-Forum dataset...")
    df = pd.read_csv(
        data_path, sep=r"\s+", engine="python",
        header=None, names=["source", "target", "timestamp"]
    )
    print(f"Raw edges: {len(df)}")
    df = df[df["source"] != df["target"]]
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Edges after cleaning: {len(df)}")
    unique_nodes = pd.unique(df[["source", "target"]].values.ravel())
    node_mapping = {node: idx for idx, node in enumerate(unique_nodes)}
    df["source"] = df["source"].map(node_mapping)
    df["target"] = df["target"].map(node_mapping)
    print(f"Total unique nodes: {len(unique_nodes)}")
    return df, node_mapping