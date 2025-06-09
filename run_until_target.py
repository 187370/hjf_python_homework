import json
import os
import re
import shutil
import subprocess
import sys
import time

TARGET_RETURN = 170
RESULT_DIR = "result"
MAX_INDEX = 100


def next_available_index() -> int | None:
    """返回 ``RESULT_DIR`` 目录中在 [0, MAX_INDEX) 范围内未使用的最小索引。"""
    used = set()
    if os.path.isdir(RESULT_DIR):
        for name in os.listdir(RESULT_DIR):
            m = re.search(r"_(\d+)\.(?:png|json)$", name)
            if m:
                used.add(int(m.group(1)))
    for i in range(MAX_INDEX):
        if i not in used:
            return i
    return None


FILES = {
    "enhanced_strategy_detailed_analysis.png": "enhanced_strategy_detailed_analysis_{i}.png",
    "enhanced_strategy_evaluation.json": "enhanced_strategy_evaluation_{i}.json",
    "stock_relationship_network.png": "stock_relationship_network_{i}.png",
    "llm_analysis_details.json": "llm_analysis_details_{i}.json",
}


os.makedirs(RESULT_DIR, exist_ok=True)

attempt = 1

while True:
    idx = next_available_index()
    if idx is None:
        print(f"没有可用的索引（0到{MAX_INDEX}）。停止运行。")
        break

    print(f"尝试第{attempt}次（索引{idx}）：正在运行 enhanced_eval_main.py...")
    proc = subprocess.run([sys.executable, "enhanced_eval_main.py"])
    if proc.returncode != 0:
        print("执行失败，正在重试...")
        attempt += 1
        continue

    # 将输出文件移动到结果目录，并添加索引后缀
    for src, pattern in FILES.items():
        if os.path.exists(src):
            dst = os.path.join(RESULT_DIR, pattern.format(i=idx))
            shutil.move(src, dst)
        else:
            print(f"警告：未找到文件 {src}")

    eval_path = os.path.join(RESULT_DIR, FILES["enhanced_strategy_evaluation.json"].format(i=idx))
    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        total_return = data.get("total_return", 0)
    except Exception as e:
        print(f"读取结果失败：{e}")
        total_return = 0

    print(f"总收益率：{total_return:.2f}%")
    if total_return >= TARGET_RETURN:
        print(f"已达到目标：{total_return:.2f}% >= {TARGET_RETURN}")
        break
    
    attempt += 1
    time.sleep(1)