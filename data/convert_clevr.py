import json
import os

split_json_file = "./clevr/splits.json"
sc_caption_file = "./clevr/change_captions.json"
nsc_caption_file = "./clevr/no_change_captions.json"

output_root = "./clevr_for_mmvid/"

split_data = json.load(open(split_json_file, "r"))
sc_data = json.load(open(sc_caption_file, "r"))
nsc_data = json.load(open(nsc_caption_file, "r"))

# for split in ["train", "val", "test"]:
for split in ["train"]:
    target_dir = os.path.join(output_root, split, "txt")
    target_video_dir = os.path.join(output_root, split, "video")
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(target_video_dir, exist_ok=True)
    split_keys = split_data[split]
    sc_output = {"{:0>6}.png".format(k): sc_data["CLEVR_default_{:0>6}.png".format(k)] for k in split_keys}
    nsc_output = {"{:0>6}.png".format(k): nsc_data["CLEVR_default_{:0>6}.png".format(k)] for k in split_keys}
    json.dump(sc_output, open(os.path.join(target_dir, "sc_videos.json"), "w"))
    json.dump(nsc_output, open(os.path.join(target_dir, "nsc_videos.json"), "w"))
