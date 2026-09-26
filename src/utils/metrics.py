"""Helpers for exporting and plotting training metrics (loss history).

These helpers are dependency-optional:
- ``csv`` / ``json`` come from the standard library.
- ``matplotlib`` is optional: ``plot_history`` returns ``False`` and prints a
  hint if it is not installed, without breaking the training loop.
"""

import csv
import json
from pathlib import Path
from typing import Sequence


def write_history_csv(path: str | Path, history: Sequence[dict], fieldnames: Sequence[str]) -> None:
    """Write the loss history to a CSV file (header + one row per eval point)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_history_json(path: str | Path, history: Sequence[dict]) -> None:
    """Write the loss history to a JSON file (list of objects)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(list(history), f, indent=2)


def plot_history(
    path: str | Path,
    history: Sequence[dict],
    loss_keys: Sequence[str] = ('train_loss', 'val_loss'),
) -> bool:
    """Plot loss curves with matplotlib (Agg backend).

    Returns ``True`` if a plot was produced, ``False`` if matplotlib is not
    installed (a hint is printed in that case).
    """
    if not history:
        return False
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  -> matplotlib not available -> no loss curve generated")
        return False

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x_key = 'epoch' if 'epoch' in history[0] else 'iter'
    xs = [row.get(x_key) for row in history]

    fig, ax = plt.subplots()
    for key in loss_keys:
        if key in history[0]:
            ax.plot(xs, [row.get(key) for row in history], marker='o', label=key)
    ax.set_xlabel(x_key)
    ax.set_ylabel('loss')
    ax.set_title('Training progress')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True