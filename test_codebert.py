import argparse
import json

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from train_codebert import DeprecatedApiMLM, MaskingCollator, masked_accuracy


def get_args():
    """Đọc tham số dòng lệnh.

    Input : argv (test_codebert.sh truyền vào tường minh)
    Output: Namespace chứa toàn bộ tham số đánh giá
    """
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../Data-Collection/codellama/U_dep_test.json")
    p.add_argument("--model", default="checkpoints/codebert-deprecated")
    p.add_argument("--target", default="replacement",
                   choices=["replacement", "deprecated"],
                   help="che API thay thế (code đã sửa) hay API deprecated")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true")
    return p.parse_args()


def retarget(sample):
    """Trỏ trường "deprecated api" sang API thay thế của mẫu.

    Input : sample - dict một mẫu
    Output: bản sao nông với "deprecated api" = [replacement api, expected call]

    Nhờ vậy surface_forms và find_spans của train_codebert dùng lại được y
    nguyên, chỉ khác tập API cần tìm.
    """
    apis = {sample.get("replacement api"), sample.get("expected call")}
    out = dict(sample)
    out["deprecated api"] = sorted(a for a in apis if a)
    return out


def main():
    """Nạp model và dữ liệu, chạy một vòng eval, in kết quả."""
    args = get_args()
    set_seed(args.seed)

    with open(args.data, encoding="utf-8") as f:
        samples = json.load(f)
    if args.target == "replacement":
        samples = [retarget(s) for s in samples]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    test_set = DeprecatedApiMLM(samples, tokenizer, args.max_length)
    print(f"nạp {len(samples)} mẫu, dùng được {len(test_set)}")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/test_codebert",
            per_device_eval_batch_size=args.batch_size,
            fp16=args.fp16,
            remove_unused_columns=False,  # giữ target_mask cho collator
            report_to="none",
            seed=args.seed,
        ),
        # 0.0 = chỉ che token API, không che ngẫu nhiên, giống hệt vòng eval
        data_collator=MaskingCollator(tokenizer, 0.0),
        compute_metrics=masked_accuracy,
        # argmax trước khi tích luỹ, nếu không eval giữ logits (N, 512, 50265)
        preprocess_logits_for_metrics=lambda logits, labels: logits.argmax(-1),
    )
    metrics = trainer.evaluate(test_set)

    print(f"\nmodel      : {args.model}")
    print(f"target     : {args.target}")
    print(f"eval_loss  : {metrics['eval_loss']:.4f}")
    print(f"masked_acc : {metrics['eval_masked_acc']:.4f}")


if __name__ == "__main__":
    main()
