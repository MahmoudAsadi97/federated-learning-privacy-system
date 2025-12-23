import os
import zipfile
import urllib.request
import pandas as pd

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00235/household_power_consumption.zip"
)

RAW_DIR = "data/raw"
ZIP_PATH = os.path.join(RAW_DIR, "household_power.zip")
TXT_PATH = os.path.join(RAW_DIR, "household_power_consumption.txt")
OUTPUT_PATH = os.path.join(RAW_DIR, "energy.csv")


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    # Download dataset if not present
    if not os.path.exists(ZIP_PATH):
        print("Downloading dataset...")
        urllib.request.urlretrieve(URL, ZIP_PATH)

    # Extract
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(RAW_DIR)

    print("Loading raw data...")

    # Load raw file WITHOUT datetime parsing
    df = pd.read_csv(
        TXT_PATH,
        sep=";",
        na_values="?",
        low_memory=False,
    )

    # Explicit, robust datetime parsing (future-proof)
    df["timestamp"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        dayfirst=True,
        errors="coerce",
    )

    # Drop original columns
    df = df.drop(columns=["Date", "Time"])

    # Drop invalid rows
    df = df.dropna(subset=["timestamp", "Global_active_power"])

    # Sort by time
    df = df.sort_values("timestamp")

    # Select target
    df = df[["timestamp", "Global_active_power"]]
    df = df.rename(columns={"Global_active_power": "target"})
    df = df.set_index("timestamp")

    # Save cleaned dataset
    df.to_csv(OUTPUT_PATH)

    print(f"Saved cleaned dataset to {OUTPUT_PATH}")
    print("Preview:")
    print(df.head())


if __name__ == "__main__":
    main()
