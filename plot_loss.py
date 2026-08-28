import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")           # không cần màn hình, chạy được trên cluster
import matplotlib.pyplot as plt


def get_args():
    """Đọc tham số dòng lệnh.

    Input : argv
    Output: Namespace {ckpt_dir, out}
    """
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="checkpoints/codebert-deprecated",
                   help="thư mục chứa các checkpoint-*")
    p.add_argument("--out", default="logs/loss_curve.png")
    return p.parse_args()


def load_history(ckpt_dir):
    """Lấy log_history từ checkpoint mới nhất.

    Input : ckpt_dir - thư mục chứa các checkpoint-*
    Output: list[dict] mỗi dict là một dòng log Trainer đã in
    """
    ckpts = glob.glob(os.path.join(ckpt_dir, "checkpoint-*"))
    if not ckpts:
        raise SystemExit(f"khong tim thay checkpoint-* trong {ckpt_dir}")
    newest = max(ckpts, key=lambda c: int(c.rsplit("-", 1)[-1]))
    with open(os.path.join(newest, "trainer_state.json"), encoding="utf-8") as f:
        return newest, json.load(f)["log_history"]


def series(history, key):
    """Tách một chuỗi (step, giá trị) ra khỏi log_history.

    Input : history - list[dict], key - tên trường cần lấy
    Output: (list step, list giá trị)
    """
    pts = [(h["step"], h[key]) for h in history if key in h]
    return [s for s, _ in pts], [v for _, v in pts]


def main():
    """Đọc trainer_state.json rồi ghi biểu đồ ra file PNG."""
    args = get_args()
    ckpt, history = load_history(args.ckpt_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(*series(history, "loss"), label="train loss", alpha=.7)
    ax1.plot(*series(history, "eval_loss"), label="eval loss", marker="o", ms=3)
    ax1.set_xlabel("step"); ax1.set_ylabel("loss"); ax1.legend(); ax1.grid(alpha=.3)
    ax1.set_title("Loss")

    steps, acc = series(history, "eval_masked_acc")
    ax2.plot(steps, acc, marker="o", ms=3, color="tab:green")
    ax2.set_xlabel("step"); ax2.set_ylabel("masked accuracy"); ax2.grid(alpha=.3)
    ax2.set_title(f"Masked accuracy (cao nhat {max(acc):.4f})" if acc else "Masked accuracy")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"doc  : {ckpt}/trainer_state.json")
    print(f"ghi  : {args.out}")


if __name__ == "__main__":
    main()
