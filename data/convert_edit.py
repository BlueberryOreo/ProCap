import json
import os

train_path = "./edit_for_mmvid/train"
os.makedirs(train_path, exist_ok=True)
os.makedirs(os.path.join(train_path, "txt"), exist_ok=True)
os.makedirs(os.path.join(train_path, "video"), exist_ok=True)

with open("./edit/train.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

train_data = {item["uid"]: item["sents"] for item in train_data}
# val_data = {item["uid"]: item["sents"] for item in val_data}
# test_data = {item["uid"]: item["sents"] for item in test_data}

with open("./edit_for_mmvid/train/txt/edit_videos.json", "wt", encoding="utf-8") as f:
    json.dump(train_data, f)

