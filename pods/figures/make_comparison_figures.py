"""Build side-by-side comparisons of detection examples across models, one PNG per task."""

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

FIGURES_DIR = Path(__file__).resolve().parent
DEFAULT_EVAL_DIR = FIGURES_DIR.parents[2] / "eval"

MODELS = [
    ("dinov2_small", "DINOv2 Small"),
    ("dinov2_base", "DINOv2 Base"),
    ("dinov3_small", "DINOv3 Small"),
    ("dinov3_base", "DINOv3 Base"),
    ("mammo_clip", "Mammo-CLIP"),
    ("mammo_fm", "MammoFM"),
]

TASKS = {
    "mass": "Mass",
    "suspicious_calcification": "Suspicious Calcification",
}


def find_example(model_dir: Path, index: int) -> Path:
    matches = sorted(model_dir.glob(f"{index:05d}_*.png"))
    if not matches:
        raise FileNotFoundError(f"No example {index:05d} in {model_dir}")
    return matches[0]


def make_figure(eval_dir: Path, task: str, examples: list[int], output: Path, dpi: int):
    models = [
        (eval_dir / f"{key}_{task}", name)
        for key, name in MODELS
        if (eval_dir / f"{key}_{task}").is_dir()
    ]
    if not models:
        print(f"Skipping {task}: no eval folders found")
        return

    n_rows, n_cols = len(examples), len(models)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.2 * n_cols, 2.2 * n_rows + 0.4),
        squeeze=False,
    )

    for row, index in enumerate(examples):
        for col, (model_dir, name) in enumerate(models):
            ax = axes[row, col]
            ax.imshow(mpimg.imread(find_example(model_dir, index)))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(name, fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"Ex. {row + 1}", fontsize=9)

    fig.legend(
        handles=[
            Patch(edgecolor="lime", facecolor="none", linewidth=2, label="Ground truth"),
            Patch(edgecolor="red", facecolor="none", linewidth=2, label="Prediction"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=0.3, h_pad=0.3)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    parser.add_argument(
        "--examples", nargs="+", type=int, default=[0, 2, 4],
        help="Example indices (the 00000_ prefix of the filenames)",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        make_figure(
            args.eval_dir,
            task,
            args.examples,
            args.output_dir / f"{task}_examples.png",
            args.dpi,
        )


if __name__ == "__main__":
    main()