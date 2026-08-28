"""Lấy các mẫu trong D_test.json không nằm trong U_dep_test.json.

Đối sánh theo trường "function". Kết quả ghi ra D_test_rest.json, giữ nguyên
mẫu gốc của D_test.json.
"""
import argparse
import json


def get_args():
    """Đọc tham số dòng lệnh.

    Input : argv
    Output: Namespace {test, exclude, out}
    """
    p = argparse.ArgumentParser()
    p.add_argument("--test", default="../Data-Collection/codellama/D_test.json")
    p.add_argument("--exclude", default="../Data-Collection/codellama/U_dep_test.json")
    p.add_argument("--out", default="../Data-Collection/codellama/D_test_rest.json")
    return p.parse_args()


def load(path):
    """Nạp một file JSON.

    Input : path - đường dẫn file
    Output: list[dict] các mẫu
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    """Loại các mẫu trùng "function" rồi ghi phần còn lại ra file JSON."""
    args = get_args()
    test = load(args.test)
    excluded = {s.get("function") for s in load(args.exclude)}
    excluded.discard(None)

    kept = [s for s in test if s.get("function") not in excluded]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    print(f"D_test      : {len(test)}")
    print(f"loại bỏ     : {len(test) - len(kept)}  ({len(excluded)} function duy nhất)")
    print(f"còn lại     : {len(kept)}  ->  {args.out}")


if __name__ == "__main__":
    main()
