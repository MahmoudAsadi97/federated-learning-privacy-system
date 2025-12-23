import argparse
import json
import os
import numpy as np
import pandas as pd


def create_non_iid_clients(df, num_clients, alpha=0.3):
    """
    Dirichlet-based non-IID partitioning over the full dataset.
    Guaranteed to assign samples to every client.
    """

    n = len(df)
    indices = np.arange(n)
    np.random.shuffle(indices)

    # Dirichlet proportions
    proportions = np.random.dirichlet(alpha * np.ones(num_clients))
    splits = (proportions * n).astype(int)

    # Fix rounding issues
    remainder = n - splits.sum()
    splits[0] += remainder

    clients = {}
    start = 0
    for cid, size in enumerate(splits):
        end = start + size
        clients[cid] = indices[start:end].tolist()
        start = end

    return clients


def main(args):
    print("Loading data...")
    df = pd.read_csv(args.data_path, index_col=0, parse_dates=True)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise RuntimeError("Index is not a DatetimeIndex — check preprocessing")

    if len(df) == 0:
        raise RuntimeError("Dataset is empty")

    print(f"Loaded {len(df)} rows")

    clients = create_non_iid_clients(df, args.num_clients)

    os.makedirs(args.output_dir, exist_ok=True)
    metadata = {}

    for cid, idxs in clients.items():
        client_df = df.iloc[idxs].sort_index()

        if len(client_df) == 0:
            raise RuntimeError(f"Client {cid} has zero samples")

        client_dir = os.path.join(args.output_dir, f"client_{cid:02d}")
        os.makedirs(client_dir, exist_ok=True)

        client_df.to_csv(os.path.join(client_dir, "data.csv"))

        metadata[f"client_{cid:02d}"] = {
            "num_samples": int(len(client_df)),
            "mean_target": float(client_df["target"].mean()),
            "std_target": float(client_df["target"].std()),
        }

        print(
            f"Client {cid:02d} | "
            f"samples={len(client_df)} | "
            f"mean={metadata[f'client_{cid:02d}']['mean_target']:.3f} | "
            f"std={metadata[f'client_{cid:02d}']['std_target']:.3f}"
        )

    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved metadata to {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()
    main(args)
