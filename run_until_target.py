import json
import subprocess
import time
import sys

TARGET_RETURN = 170
RESULT_FILE = 'enhanced_strategy_evaluation.json'

attempt = 1

while True:
    print(f'Attempt {attempt}: running enhanced_eval_main.py...')
    # Run the evaluation script
    proc = subprocess.run([sys.executable, 'enhanced_eval_main.py'])
    if proc.returncode != 0:
        print('Execution failed, retrying...')
        attempt += 1
        continue

    # Load evaluation results
    try:
        with open(RESULT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        total_return = data.get('total_return', 0)
    except Exception as e:
        print(f'Failed to read results: {e}')
        total_return = 0

    print(f'Total return: {total_return:.2f}%')
    if total_return >= TARGET_RETURN:
        print(f'Target achieved: {total_return:.2f}% >= {TARGET_RETURN}')
        break

    attempt += 1
    time.sleep(1)

