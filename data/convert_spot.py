import json
import os

train_path = "./spot_for_mmvid/train"
os.makedirs(train_path, exist_ok=True)
os.makedirs(os.path.join(train_path, "txt"), exist_ok=True)
os.makedirs(os.path.join(train_path, "video"), exist_ok=True)

with open("./spot/captions/filter_train.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

train_data = {item["img_id"]: item["sentences"] for item in train_data}

with open("./spot_for_mmvid/train/txt/spot_videos.json", "wt", encoding="utf-8") as f:
    json.dump(train_data, f)
