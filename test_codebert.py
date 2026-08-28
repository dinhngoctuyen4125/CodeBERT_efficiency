"""Đo accuracy đoán token API của model đã finetune trên U_dep_test.json.

Chỉ chạy inference, không train và không cập nhật trọng số. Cách che token
giống hệt vòng eval khi train (chỉ che token API, không che ngẫu nhiên) nên
con số so sánh trực tiếp được với eval_masked_acc trong log training.

U_dep_test.json là code đã được sửa (dùng API thay thế), nên mặc định script
nhắm vào API thay thế — xem --target.
"""
import argparse
import json

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from train_codebert import DeprecatedApiMLM, MaskingCollator


def get_args():
    """Đọc tham số dòng lệnh.

    Input : argv (test_codebert.sh truyền vào tường minh)
    Output: Namespace chứa tham số đánh giá
    """
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../Data-Collection/codellama/U_dep_test.json")
    p.add_argument("--model", default="tummitum/codebert-deprecated")
    p.add_argument("--target", default="replacement",
                   choices=["replacement", "deprecated"],
                   help="che API thay thế (code đã sửa) hay API deprecated")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
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


@torch.no_grad()
def masked_accuracy(model, dataset, collator, batch_size, device, fp16):
    """Chạy inference và đếm token bị che mà model đoán đúng.

    Input : model, dataset, collator, batch_size, device, fp16
    Output: (số token đúng, tổng số token bị che)
    """
    model.eval().to(device)
    correct = total = 0
    for i in range(0, len(dataset), batch_size):
        batch = collator([dataset[j] for j in range(i, min(i + batch_size, len(dataset)))])
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and device == "cuda"):
            preds = model(**batch).logits.argmax(-1)
        keep = labels != -100
        correct += (preds[keep] == labels[keep]).sum().item()
        total += int(keep.sum())
    return correct, total


def main():
    """Nạp model và dữ liệu, đo accuracy, in kết quả."""
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.data, encoding="utf-8") as f:
        samples = json.load(f)
    if args.target == "replacement":
        samples = [retarget(s) for s in samples]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    dataset = DeprecatedApiMLM(samples, tokenizer, args.max_length)
    # 0.0 = chỉ che token API, không che ngẫu nhiên, giống hệt vòng eval
    collator = MaskingCollator(tokenizer, 0.0)

    correct, total = masked_accuracy(model, dataset, collator,
                                     args.batch_size, device, args.fp16)
    if not total:
        raise SystemExit("không tìm thấy token API nào để đo")

    print(f"model      : {args.model}")
    print(f"data       : {args.data}")
    print(f"target     : {args.target}")
    print(f"mẫu        : {len(dataset)}/{len(samples)}")
    print(f"token che  : {total}")
    print(f"masked_acc : {correct / total:.4f}  ({correct}/{total})")


if __name__ == "__main__":
    main()
