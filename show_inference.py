"""In ra vài mẫu lúc test để xem input, dự đoán của model, và ground truth.

Tương tự show_masked.py nhưng dành cho giai đoạn inference:
  - input : đoạn code đã che <mask> (đúng như model nhận lúc test)
  - infer : kết quả model dự đoán tại mỗi vị trí <mask>
  - groundtruth : token gốc trước khi che

Dùng lại DeprecatedLineMLM và MaskingCollator của test/train_codebert.

  python show_inference.py                           # 3 mẫu đầu, CPU
  python show_inference.py --n 5 --fp16              # 5 mẫu, GPU fp16
"""
import argparse
import textwrap

import json
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, set_seed

from test_codebert import DeprecatedLineMLM
from train_codebert import MaskingCollator


def get_args():
    """Đọc tham số dòng lệnh.

    Input : argv
    Output: Namespace {data, model, n, max_length, seed, fp16}
    """
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../Data-Collection/codellama/U_dep_test.json")
    p.add_argument("--model", default="tummitum/codebert-deprecated")
    p.add_argument("--n", type=int, default=3, help="số mẫu cần in")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true")
    return p.parse_args()


def show(tokenizer, batch, preds, i=0):
    """In input đã che, dự đoán của model, và ground truth cho một mẫu.

    Input : tokenizer, batch - dict do MaskingCollator trả về,
            preds - tensor dự đoán (argmax logits), i - chỉ số mẫu
    Output: không trả về gì, chỉ in ra màn hình
    """
    ids = batch["input_ids"][i].tolist()
    labels = batch["labels"][i].tolist()
    pred_ids = preds[i].tolist()

    mask_positions = [pos for pos, lab in enumerate(labels) if lab != -100]

    # --- 1. Decode input (đoạn code với <mask>) ---
    print("\n--- INPUT (code đã che <mask>) ---")
    text = tokenizer.decode(ids)
    for special in (tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token):
        if special:
            text = text.replace(special, "")
    print(text)

    # --- 2. Dự đoán vs Ground truth tại từng vị trí bị che ---
    print(f"\n--- VỊ TRÍ BỊ CHE: {len(mask_positions)} token ---")
    print(f"{'pos':>5}  {'ground truth':30s}  {'dự đoán':30s}  {'đúng?'}")
    print("-" * 78)
    n_correct = 0
    for pos in mask_positions:
        gt_token = tokenizer.decode([labels[pos]]).strip()
        pr_token = tokenizer.decode([pred_ids[pos]]).strip()
        match = gt_token == pr_token
        n_correct += int(match)
        mark = "✓" if match else "✗"
        print(f"{pos:5d}  {gt_token:30s}  {pr_token:30s}  {mark}")

    acc = n_correct / len(mask_positions) if mask_positions else 0
    print(f"\n=> accuracy mẫu này: {n_correct}/{len(mask_positions)} = {acc:.2%}")

    # --- 3. Decode bản ground truth (khôi phục token gốc) ---
    restored = list(ids)
    for pos in mask_positions:
        restored[pos] = labels[pos]
    print("\n--- GROUND TRUTH (code gốc, chưa che) ---")
    text_gt = tokenizer.decode(restored)
    for special in (tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token):
        if special:
            text_gt = text_gt.replace(special, "")
    print(text_gt)

    # --- 4. Decode bản infer (thay <mask> bằng dự đoán) ---
    inferred = list(ids)
    for pos in mask_positions:
        inferred[pos] = pred_ids[pos]
    print("\n--- INFER (code model điền vào) ---")
    text_inf = tokenizer.decode(inferred)
    for special in (tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token):
        if special:
            text_inf = text_inf.replace(special, "")
    print(text_inf)

    # --- 5. Tensor thô ---
    for name in ("input_ids", "attention_mask", "labels"):
        tensor = batch[name]
        print(f"\n--- {name}  shape={tuple(tensor.shape)}  dtype={tensor.dtype} ---")
        print(textwrap.fill(str(tensor[i].tolist()), width=100))


@torch.no_grad()
def main():
    """Nạp model và dữ liệu test, chạy inference rồi in kết quả từng mẫu."""
    args = get_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.data, encoding="utf-8") as f:
        samples = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model).eval().to(device)

    dataset = DeprecatedLineMLM(samples[:args.n], tokenizer, args.max_length)
    collator = MaskingCollator(tokenizer, 0.0)  # chỉ che API, không che ngẫu nhiên

    # gói chung một batch
    batch = collator([dataset[i] for i in range(len(dataset))])
    batch_device = {k: v.to(device) for k, v in batch.items()}

    # inference
    input_for_model = {k: v for k, v in batch_device.items() if k != "labels"}
    with torch.autocast("cuda", dtype=torch.float16,
                        enabled=args.fp16 and device == "cuda"):
        logits = model(**input_for_model).logits
    preds = logits.argmax(-1).cpu()

    for i in range(len(dataset)):
        print(f"\n{'=' * 78}\nMẪU {i}/{len(dataset)}\n{'=' * 78}")
        show(tokenizer, batch, preds, i)


if __name__ == "__main__":
    main()
