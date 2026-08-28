"""In ra vài mẫu sau khi masking để xem model thực sự nhận input/label gì.

Dùng lại DeprecatedApiMLM và MaskingCollator của train_codebert nên những gì
in ra đúng bằng những gì Trainer đưa vào model.

  --random_mask_prob 0.1  -> giống lúc train (API + 10% ngẫu nhiên)
  --random_mask_prob 0.0  -> giống lúc eval  (chỉ API)
"""
import argparse
import json
import textwrap

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


def show(tokenizer, batch, i=0):
    """In đúng ba tensor mà model nhận, kèm bản decode để đọc.

    Input : tokenizer, batch - dict do MaskingCollator trả về, i - chỉ số mẫu
    Output: không trả về gì, chỉ in ra màn hình
    """
    ids = batch["input_ids"][i].tolist()
    labels = batch["labels"][i].tolist()

    print("\n--- decode để đọc (không phải thứ model nhận) ---")
    # không dùng skip_special_tokens: nó xoá luôn <mask>, tức đúng chỗ cần xem
    text = tokenizer.decode(ids)
    for special in (tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token):
        text = text.replace(special, "")
    print(text)

    for name in ("input_ids", "attention_mask", "labels"):
        tensor = batch[name]
        print(f"\n--- {name}  shape={tuple(tensor.shape)}  dtype={tensor.dtype} ---")
        print(textwrap.fill(str(tensor[i].tolist()), width=100))

    n = sum(1 for lab in labels if lab != -100)
    print(f"\n=> {n}/{len(ids)} vị trí có loss   "
          f"(<mask> = id {tokenizer.mask_token_id}, -100 = bỏ qua)")


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
        show(tokenizer, batch)


if __name__ == "__main__":
    main()
