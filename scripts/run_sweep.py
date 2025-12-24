import subprocess
import sys
from pathlib import Path

NOISE_MULTIPLIERS = [0.3, 0.6, 1.0, 2.0]

ROUNDS = 5
LOCAL_EPOCHS = 1
BATCH_SIZE = 128
MAX_GRAD_NORM = 1.0

OUTDIR = "runs"


def run():
    Path(OUTDIR).mkdir(exist_ok=True)

    for nm in NOISE_MULTIPLIERS:
        tag = f"nm{nm}".replace(".", "_")
        print(f"\n=== Running sweep: noise_multiplier={nm} ===")

        cmd = [
            sys.executable,
            "scripts/run_experiment.py",
            "--dp",
            "--noise-multiplier",
            str(nm),
            "--max-grad-norm",
            str(MAX_GRAD_NORM),
            "--rounds",
            str(ROUNDS),
            "--local-epochs",
            str(LOCAL_EPOCHS),
            "--batch-size",
            str(BATCH_SIZE),
            "--outdir",
            OUTDIR,
            "--tag",
            tag,
        ]

        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    run()
