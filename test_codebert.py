import argparse
import json

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from train_codebert import (
    DeprecatedApiMLM,
    MaskingCollator,
    find_spans,
    surface_forms,
    token_window,
)


def get_args():
    """Đọc tham số dòng lệnh.

    Input : argv (test_codebert.sh truyền vào tường minh)
    Output: Namespace chứa tham số đánh giá
    """
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../Data-Collection/codellama/U_dep_test.json")
    p.add_argument("--model", default="tummitum/codebert-deprecated")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--fp16", action="store_true")
    return p.parse_args()


class DeprecatedLineMLM(DeprecatedApiMLM):
    """Che API cũ trong "y_neg"; "probing input" chỉ đóng vai ngữ cảnh.

    Input : samples - list[dict] các mẫu, tokenizer, max_length
    Output: mỗi phần tử là dict {input_ids, attention_mask, target_mask}

    Không dùng "function" vì trong U_dep_test nó đã được sửa sang API mới.
    Span nằm trong phần "probing input" bị loại: đó cũng là code đã sửa, và
    dạng rút gọn của API cũ (norm) khớp nhầm vào đuôi API mới
    (torch.linalg.norm).
    """

    def __init__(self, samples, tokenizer, max_length):
        self.features = []
        for s in samples:
            context = s.get("probing input") or ""
            text = context + (s.get("y_neg") or "")
            spans = [(a, b) for a, b in find_spans(text, surface_forms(s))
                     if a >= len(context)]
            if not spans:
                continue
            enc = tokenizer(text, return_offsets_mapping=True)
            target = [
                int(any(o_s < e and s_ < o_e for s_, e in spans) and o_e > o_s)
                for o_s, o_e in enc["offset_mapping"]
            ]
            if not any(target):
                continue
            ids = enc["input_ids"]
            if len(ids) > max_length:
                ids, target = token_window(ids, target, max_length,
                                           tokenizer.bos_token_id,
                                           tokenizer.eos_token_id)
            self.features.append({
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "target_mask": target,
            })


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
    """Nạp model và dữ liệu, đo accuracy trên API deprecated, in kết quả."""
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.data, encoding="utf-8") as f:
        samples = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    dataset = DeprecatedLineMLM(samples, tokenizer, args.max_length)
    # 0.0 = chỉ che token API, không che ngẫu nhiên, giống hệt vòng eval
    collator = MaskingCollator(tokenizer, 0.0)

    correct, total = masked_accuracy(model, dataset, collator,
                                     args.batch_size, device, args.fp16)
    if not total:
        raise SystemExit("không tìm thấy token API nào để đo")

    print(f"model      : {args.model}")
    print(f"data       : {args.data}")
    print(f"mẫu        : {len(dataset)}/{len(samples)}")
    print(f"token che  : {total}")
    print(f"masked_acc : {correct / total:.4f}  ({correct}/{total})")


if __name__ == "__main__":
    main()
