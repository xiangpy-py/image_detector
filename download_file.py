import kagglehub
import os

# 设置缓存目录
os.environ["KAGGLEHUB_CACHE"] = "database"

# 下载
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

print("数据集路径:", path)
print("内容列表:", os.listdir(path))
