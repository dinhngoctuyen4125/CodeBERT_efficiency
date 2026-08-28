# Finetuning CodeBERT on deprecated APIs

Masked-language-model finetuning on `data/codellama/D_forget.json`: the tokens of
the deprecated API call are masked and CodeBERT must recover them from the
surrounding code. Split 80/20 train/eval, early stopping on eval loss.

## 1. Set Up the Environment

```bash
conda create -n codebert python=3.10
conda activate codebert
pip install -r requirements.txt
```

## 2. Finetune

```bash
mkdir -p logs
nohup bash train_codebert.sh > logs/train_codebert.log 2>&1 &
```

## 3. Evaluate on U_dep_test

```bash
nohup bash test_codebert.sh > logs/test_codebert.log 2>&1 &
```

## 4. Plot the loss curve

```bash
python plot_loss.py
```
