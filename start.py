import time
import py3nvml
import subprocess

# 初始化 NVML
py3nvml.py3nvml.nvmlInit()

# 获取显卡数量
device_count = py3nvml.py3nvml.nvmlDeviceGetCount()

# 确保存在至少一个显卡
if device_count < 1:
    print("未找到可用的显卡")
    exit(1)

# 获取显卡0的句柄
handle = py3nvml.py3nvml.nvmlDeviceGetHandleByIndex(0)

# 持续监测显存剩余量
while True:
    info = py3nvml.py3nvml.nvmlDeviceGetMemoryInfo(handle)
    if info.free / 1024**2 > 21000:
        subprocess.run("nohup bash run_OfficeHome.sh &", shell=True)
        break
    # 休眠1秒
    time.sleep(100)
