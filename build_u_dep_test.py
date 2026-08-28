import argparse
import json


def get_args():
    """Đọc tham số dòng lệnh.

    Input : argv
    Output: Namespace {updated, test, out}
    """
    p = argparse.ArgumentParser()
    p.add_argument("--updated", default="../Data-Collection/codellama/updated_dep.json")
    p.add_argument("--test", default="../Data-Collection/codellama/D_test.json")
    p.add_argument("--out", default="../Data-Collection/codellama/U_dep_test.json")
    return p.parse_args()


def load(path):
    """Nạp một file JSON.

    Input : path - đường dẫn file
    Output: list[dict] các mẫu
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    """Lọc theo "function" rồi ghi ra file JSON."""
    args = get_args()
    updated = load(args.updated)
    test_functions = {s.get("function") for s in load(args.test)}
    test_functions.discard(None)

    kept = [s for s in updated if s.get("function") in test_functions]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    print(f"updated_dep : {len(updated)}")
    print(f"D_test      : {len(test_functions)} function duy nhat")
    print(f"trung khop  : {len(kept)}  ->  {args.out}")


if __name__ == "__main__":
    main()
