"""In ra vài mẫu sau khi masking để xem model thực sự nhận input/label gì.

Dùng lại DeprecatedApiMLM và MaskingCollator của train_codebert nên những gì
in ra đúng bằng những gì Trainer đưa vào model.

  --random_mask_prob 0.1  -> giống lúc train (API + 10% ngẫu nhiên)
  --random_mask_prob 0.0  -> giống lúc eval  (chỉ API)
"""
import argparse
import json

from transformers import AutoTokenizer, set_seed

from train_codebert import DeprecatedApiMLM, MaskingCollator


def get_args():
    """Đọc tham số dòng lệnh.

    Input : argv
    Output: Namespace {data, model, n, random_mask_prob, seed}
    """
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../Data-Collection/codellama/D_forget.json")
    p.add_argument("--model", default="microsoft/codebert-base-mlm")
    p.add_argument("--n", type=int, default=3, help="số mẫu cần in")
    p.add_argument("--random_mask_prob", type=float, default=0.0)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def show(tokenizer, ids_in, ids_out, labels):
    """In input đã che và bảng label của một mẫu.

    Input : tokenizer, ids_in - token gốc, ids_out - sau khi che, labels
    Output: không trả về gì, chỉ in ra màn hình
    """
    mask_id = tokenizer.mask_token_id
    print("\n--- INPUT đưa vào model ---")
    # không dùng skip_special_tokens: nó xoá luôn <mask>, tức đúng chỗ cần xem
    text = tokenizer.decode(ids_out)
    for special in (tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token):
        text = text.replace(special, "")
    print(text)

    print("--- LABELS (chỉ vị trí != -100) ---")
    for i, lab in enumerate(labels):
        if lab == -100:
            continue
        token = tokenizer.convert_ids_to_tokens([lab])[0]
        shown = "<mask>" if ids_out[i] == mask_id else \
                tokenizer.convert_ids_to_tokens([ids_out[i]])[0]
        # token API bị che 100%, token ngẫu nhiên theo 80/10/10 nên có thể
        # giữ nguyên hoặc bị thay bằng token bừa
        print(f"  vị trí {i:4}  input={shown:<14} label={lab:<6} = {token!r}")
    n = sum(1 for lab in labels if lab != -100)
    print(f"  => {n} vị trí có loss / {len(ids_in)} token")


def main():
    """Dựng dataset, che vài mẫu đầu rồi in input và label."""
    args = get_args()
    set_seed(args.seed)

    with open(args.data, encoding="utf-8") as f:
        samples = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = DeprecatedApiMLM(samples[:args.n], tokenizer, args.max_length)
    collator = MaskingCollator(tokenizer, args.random_mask_prob)

    for i in range(len(dataset)):
        feature = dataset[i]
        batch = collator([feature])
        print(f"\n{'=' * 78}\nMẪU {i}   (random_mask_prob={args.random_mask_prob})\n{'=' * 78}")
        show(tokenizer, feature["input_ids"],
             batch["input_ids"][0].tolist(), batch["labels"][0].tolist())


if __name__ == "__main__":
    main()
