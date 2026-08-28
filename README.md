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
sbatch train_codebert.sh
```