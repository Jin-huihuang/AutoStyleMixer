import json
import argparse

parser = argparse.ArgumentParser(description='best_test')

parser.add_argument('--dir', type=str, help='jsonl_dir')
args = parser.parse_args()

# 定义jsonl文件路径
jsonl_file = args.dir

domain = -1
loss = 10000
n = 0
sum = 0
acc = 0
# 逐行读取jsonl文件
with open(jsonl_file, 'r') as f:
    for line in f:
        # 解析JSON数据
        data = json.loads(line)
        if data['args']['real_test_envs'][0] !=domain:
            sum += acc
            domain +=1
            loss = 10000
            acc = 0
            n=0
            continue
        if n<5:
            n+=1
            continue
        if data['loss_cot'] < loss:
            loss = data['loss_cot']
            acc = data['test_inMT']
    sum += acc

# 输出结果
print("Average Accuracy: {:.3f}".format(100*sum/(data['args']['real_test_envs'][0] + 1)))
