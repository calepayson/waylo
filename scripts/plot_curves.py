# scripts/plot_curves.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_results(results_dir: Path, title: str):
    loss = results_dir / "loss_history.csv"
    maph = results_dir / "map_history.csv"

    if not loss.exists() or not maph.exists():
        print(f"[SKIP] {results_dir} is missing CSVs.")
        return

    # Plot loss
    df_loss = pd.read_csv(loss)
    plt.plot(df_loss["epoch"], df_loss["loss"])
    plt.title(f"{title} Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid()
    plt.savefig(results_dir / "loss_curve.png", dpi=200)
    plt.close()

    # Plot mAP
    df_map = pd.read_csv(maph)
    plt.plot(df_map["epoch"], df_map["map"])
    plt.title(f"{title} mAP")
    plt.xlabel("epoch")
    plt.ylabel("mAP")
    plt.grid()
    plt.savefig(results_dir / "map_curve.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    PROJECT = Path(__file__).resolve().parent.parent

    plot_results(PROJECT / "results_voc", "VOC")
    plot_results(PROJECT / "results_waymo", "Waymo")
