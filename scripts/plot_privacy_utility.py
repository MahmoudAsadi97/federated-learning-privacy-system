import json
from pathlib import Path
import matplotlib.pyplot as plt


RUNS_DIR = Path("runs")


def main():
    eps = []
    mse = []
    labels = []

    for f in sorted(RUNS_DIR.glob("*.json")):
        with open(f) as fp:
            d = json.load(fp)

        if not d["dp"]:
            continue

        eps.append(d["final_eps_mean"])
        mse.append(d["final_global_mse"])
        labels.append(f.stem)

    plt.figure()
    plt.scatter(eps, mse)

    for e, m, lbl in zip(eps, mse, labels):
        plt.annotate(lbl, (e, m), textcoords="offset points", xytext=(5, 5))

    plt.xlabel("ε (mean across clients)")
    plt.ylabel("Final Global MSE")
    plt.title("Privacy–Utility Trade-off (DP Federated Learning)")
    plt.grid(True)

    out = RUNS_DIR / "privacy_utility.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved plot → {out}")


if __name__ == "__main__":
    main()
