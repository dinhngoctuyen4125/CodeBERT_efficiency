import argparse
import json
import random
import re

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


def get_args():
    """Đọc tham số dòng lệnh.

    Input : argv (train_codebert.sh truyền vào tường minh)
    Output: Namespace chứa toàn bộ hyperparameter
    """
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../Data-Collection/codellama/D_forget.json")
    p.add_argument("--model", default="microsoft/codebert-base-mlm")
    p.add_argument("--output_dir", default="checkpoints/codebert-deprecated")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--epochs", type=float, default=10.0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--random_mask_prob", type=float, default=0.0,
                   help="che ngẫu nhiên thêm trên token không phải API")
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--eval_steps", type=int, default=250)
    p.add_argument("--patience", type=int, default=3,
                   help="số lần eval không cải thiện trước khi dừng")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--report_to", default="none")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Tìm vị trí API deprecated trong code
# --------------------------------------------------------------------------- #
def surface_forms(sample):
    """Liệt kê mọi cách API deprecated có thể được viết trong code.

    Input : sample - dict một mẫu trong D_forget.json
    Output: list[str] các form, sắp dài trước ngắn sau
            vd. ['numpy.product', 'np.product', 'product']
    """
    deprecated = set(sample.get("deprecated api") or [])
    forms = set()
    for alias, full in (sample.get("alias dict") or {}).items():
        if full in deprecated:
            forms.add(alias)
    for api in deprecated:
        forms.add(api)
        forms.add(api.split(".")[-1])
    return sorted({f for f in forms if f}, key=len, reverse=True)


def find_spans(text, forms):
    """Tìm vị trí các form API xuất hiện trong đoạn code.

    Input : text  - str đoạn code
            forms - list[str] từ surface_forms
    Output: list[(start, end)] span ký tự, sắp tăng dần, không chồng lấn

    Lookbehind chỉ chặn ký tự chữ (không chặn dấu chấm), nên vẫn bắt được lời
    gọi phương thức như es.render(); form ngắn nằm trong form dài đã khớp thì
    danh sách taken lo. Lookahead chặn khớp nhầm tf.div trong tf.divide.
    """
    spans, taken = [], []
    for form in forms:
        for m in re.finditer(r"(?<!\w)" + re.escape(form) + r"(?![\w])", text):
            if any(s < m.end() and m.start() < e for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            spans.append((m.start(), m.end()))
    return sorted(spans)


def build_text(sample):
    """Chọn đoạn code để train, đảm bảo có chứa lời gọi deprecated.

    Input : sample - dict một mẫu
    Output: (text, forms) - text lấy từ trường function; nếu function không
            chứa lời gọi thì lấy probing input + y_neg
    """
    forms = surface_forms(sample)
    code = sample.get("function") or ""
    if find_spans(code, forms):
        return code, forms
    # vài mẫu chỉ còn dòng deprecated trong y_neg
    return (sample.get("probing input") or "") + (sample.get("y_neg") or ""), forms


def token_window(ids, target, max_length, bos_id, eos_id):
    """Cắt chuỗi token dài, giữ cửa sổ quanh token API đầu tiên.

    Input : ids, target - hai list cùng độ dài, đã gồm <s> ... </s>
            max_length  - số token tối đa
            bos_id, eos_id - id của <s> và </s>
    Output: (ids, target) dài đúng max_length, vẫn có <s> đầu và </s> cuối

    Phải cắt theo token chứ không theo ký tự, vì truncation của tokenizer đo
    bằng token; cắt theo ký tự sẽ để lọt đoạn ngắn nhưng nhiều token và lời
    gọi API bị cắt mất.
    """
    body = max_length - 2
    start = min(max(1, target.index(1) - body // 2), max(1, len(ids) - 1 - body))
    return ([bos_id] + ids[start:start + body] + [eos_id],
            [0] + target[start:start + body] + [0])


# --------------------------------------------------------------------------- #
# Dữ liệu
# --------------------------------------------------------------------------- #
class DeprecatedApiMLM(Dataset):
    """Tokenize sẵn toàn bộ mẫu và đánh dấu token nào là API deprecated.

    Input : samples - list[dict] các mẫu, tokenizer, max_length
    Output: mỗi phần tử là dict {input_ids, attention_mask, target_mask},
            dài tối đa max_length và luôn còn ít nhất một token API
    """

    def __init__(self, samples, tokenizer, max_length):
        self.features = []
        for s in samples:
            text, forms = build_text(s)
            spans = find_spans(text, forms)
            if not spans:
                continue
            # tokenize không truncation, để còn thấy token API ở cuối chuỗi dài
            enc = tokenizer(text, return_offsets_mapping=True)
            # token là target nếu khoảng ký tự của nó chồng lên một span API;
            # offset (0, 0) là token đặc biệt, không bao giờ bị che
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

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]


class MaskingCollator:
    """Gộp nhiều mẫu thành một batch và đặt mask lên đó.

    Input : tokenizer, random_mask_prob - tỉ lệ che ngẫu nhiên thêm
            (bản dùng cho eval đặt 0.0 nên chỉ che token API)
    """

    def __init__(self, tokenizer, random_mask_prob):
        self.pad_id = tokenizer.pad_token_id
        self.mask_id = tokenizer.mask_token_id
        self.mlm = DataCollatorForLanguageModeling(
            tokenizer, mlm_probability=random_mask_prob)

    def __call__(self, features):
        """Pad về độ dài lớn nhất trong batch rồi đặt mask.

        Input : features - list[dict] lấy từ DeprecatedApiMLM
        Output: dict {input_ids đã che, attention_mask, labels}
                labels = -100 ở vị trí không bị che
        """
        width = max(len(f["input_ids"]) for f in features)

        def stack(key, pad_value):
            return torch.tensor([f[key] + [pad_value] * (width - len(f[key]))
                                 for f in features])

        input_ids = stack("input_ids", self.pad_id)
        attention_mask = stack("attention_mask", 0)
        target = stack("target_mask", 0).bool()

        # HF lo phần che ngẫu nhiên (80/10/10, bỏ qua token đặc biệt)
        masked, labels = self.mlm.torch_mask_tokens(input_ids.clone())
        # token API luôn bị che, bất kể lượt rút ngẫu nhiên ở trên
        masked[target] = self.mask_id
        labels[target] = input_ids[target]
        # target_mask dừng ở đây; model chỉ nhận đúng 3 key này
        return {"input_ids": masked, "attention_mask": attention_mask,
                "labels": labels}


class MlmTrainer(Trainer):
    """Trainer dùng collator riêng lúc eval, để eval_loss không bị nhiễu.

    Input : như Trainer, thêm eval_collator
    """

    def __init__(self, *a, eval_collator=None, **kw):
        super().__init__(*a, **kw)
        self.eval_collator = eval_collator

    def get_eval_dataloader(self, eval_dataset=None):
        """Dựng eval dataloader bằng eval_collator thay cho data_collator.

        Input : eval_dataset - mặc định lấy self.eval_dataset
        Output: DataLoader dùng cho vòng eval
        """
        train_collator, self.data_collator = self.data_collator, self.eval_collator
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.data_collator = train_collator


def masked_accuracy(eval_pred):
    """Tính tỉ lệ token bị che mà model đoán đúng.

    Input : eval_pred - (preds đã argmax, labels; -100 = không bị che)
    Output: dict {'masked_acc': float}
    """
    preds, labels = eval_pred
    keep = labels != -100
    return {"masked_acc": float((preds[keep] == labels[keep]).mean())}


def main():
    """Nạp dữ liệu, chia 80/20, train với early stopping, lưu bản tốt nhất.

    Input : tham số từ get_args, file JSON ở --data
    Output: model + tokenizer ghi ra --output_dir
    """
    args = get_args()
    set_seed(args.seed)

    with open(args.data, encoding="utf-8") as f:
        samples = json.load(f)
    random.Random(args.seed).shuffle(samples)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)

    n_val = int(len(samples) * args.val_ratio)
    train_set = DeprecatedApiMLM(samples[n_val:], tokenizer, args.max_length)
    eval_set = DeprecatedApiMLM(samples[:n_val], tokenizer, args.max_length)
    print(f"train={len(train_set)} eval={len(eval_set)}")

    trainer = MlmTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            fp16=args.fp16,
            logging_steps=50,
            evaluation_strategy="steps",
            save_strategy="steps",
            # eval_steps phải bằng save_steps khi bật load_best_model_at_end
            eval_steps=args.eval_steps,
            save_steps=args.eval_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            remove_unused_columns=False,  # giữ target_mask cho collator
            report_to=args.report_to,
            seed=args.seed,
        ),
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=MaskingCollator(tokenizer, args.random_mask_prob),
        eval_collator=MaskingCollator(tokenizer, 0.0),
        compute_metrics=masked_accuracy,
        # argmax trước khi tích luỹ, nếu không eval giữ logits (N, 512, 50265)
        preprocess_logits_for_metrics=lambda logits, labels: logits.argmax(-1),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
