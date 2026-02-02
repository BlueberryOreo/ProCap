import json

with open("./edit_word2idx2.json", "r") as f:
    word2idx = json.load(f)

new_word2idx = {
    "[PAD]": 0,
    "[CLS]": 1,
    "[SEP]": 2,
    "[VID]": 3,
    "[BOS]": 4,
    "[EOS]": 5,
    "[UNK]": 6,
    "[MASK]": 7,
}

for k in word2idx:
    if k not in new_word2idx:
        new_word2idx[k] = len(new_word2idx)

with open("./edit_word2idx2.json", "w") as f:
    json.dump(new_word2idx, f, indent=4)
