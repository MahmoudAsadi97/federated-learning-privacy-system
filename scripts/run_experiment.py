import os
import json
import argparse
from datetime import datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader

from fl_privacy_system.models.model import EnergyRegressor
from fl_privacy_system.utils.dataset import make_supervised
from fl_privacy_system.utils.seed import set_global_seed
from fl_privacy_system.fl.client import FLClient
from fl_privacy_system.fl.server import FLServer
from fl_privacy_system.evaluation.metrics import mse


DATA_DIR = "data/clients"


def parse_args():
    p = argparse.ArgumentParser(description="Federated Learning Experiment Runner")

    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)

    p.add_argument("--dp", action="store_true")
    p.add_argument("--noise-multiplier", type=float, default=1.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=1e-5)

    p.add_argument("--outdir", type=str, default="runs")
    p.add_argument("--tag", type=str, default="")

    return p.parse_args()


def load_clients(batch_size):
    datasets = []

    for client_name in sorted(os.listdir(DATA_DIR)):
        client_path = os.path.join(DATA_DIR, client_name, "data.csv")
        if not os.path.exists(client_path):
            continue

        df = pd.read_csv(client_path, index_col=0, parse_dates=True)
        dataset = make_supervised(df["target"], window=1)
        datasets.append(dataset)

    if len(datasets) == 0:
        raise RuntimeError("No client datasets found — did you generate clients?")

    return datasets


def evaluate_global(model, datasets, batch_size):
    loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=False)
        for ds in datasets
    ]
    losses = [mse(model, loader) for loader in loaders]
    return sum(losses) / len(losses)


def main():
    args = parse_args()
    set_global_seed(42)

    print("Loading federated clients...")
    client_datasets = load_clients(args.batch_size)
    print(f"Loaded {len(client_datasets)} clients")

    global_model = EnergyRegressor(input_dim=1)
    server = FLServer(global_model)

    final_loss = None
    final_eps_mean = None
    final_eps_max = None

    for rnd in range(args.rounds):
        print(f"\n--- Federated round {rnd} ---")
        client_states = []
        round_eps = []

        for cid, dataset in enumerate(client_datasets):
            client = FLClient(
                client_id=cid,
                model=server.distribute(),
                data=dataset,
                epochs=args.local_epochs,
                batch_size=args.batch_size,
                dp=args.dp,
                noise_multiplier=args.noise_multiplier,
                max_grad_norm=args.max_grad_norm,
            )

            state = client.train()
            client_states.append(state)
            round_eps.append(client.epsilon)

            if client.epsilon is not None:
                print(f"Client {cid} ε = {client.epsilon:.2f}")

        server.aggregate(client_states)

        final_loss = evaluate_global(
            server.global_model, client_datasets, args.batch_size
        )
        print(f"Global MSE after round {rnd}: {final_loss:.4f}")

        epsilons = [e for e in round_eps if e is not None]
        if epsilons:
            final_eps_mean = float(sum(epsilons) / len(epsilons))
            final_eps_max = float(max(epsilons))
            print(
                f"ε mean = {final_eps_mean:.2f} | ε max = {final_eps_max:.2f}"
            )

    torch.save(server.global_model.state_dict(), "models/federated_model_V2.pt")
    print("\nSaved federated model to federated_model.pt")

    # ---- Save run summary ----
    os.makedirs(args.outdir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.tag:
        run_id = f"{run_id}_{args.tag}"

    summary = {
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "dp": bool(args.dp),
        "noise_multiplier": args.noise_multiplier,
        "max_grad_norm": args.max_grad_norm,
        "delta": args.delta,
        "final_global_mse": final_loss,
        "final_eps_mean": final_eps_mean,
        "final_eps_max": final_eps_max,
    }

    summary_path = os.path.join(args.outdir, f"{run_id}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved run summary → {summary_path}")


if __name__ == "__main__":
    main()
