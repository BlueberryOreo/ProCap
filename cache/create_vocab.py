import os
import json

with open("../data/levir/LevirCCcaptions.json", "r") as f:
    data = json.load(f)

data = data["images"]
word2idx = {
    "[PAD]": 0,
    "[CLS]": 1,
    "[SEP]": 2,
    "[VID]": 3,
    "[BOS]": 4,
    "[EOS]": 5,
    "[UNK]": 6,
    "[MASK]": 7,
}

for item in data:
    for caption in item["sentences"]:
        for word in caption["tokens"]:
            if word not in word2idx:
                word2idx[word] = len(word2idx)

# Save the word2idx dictionary to a file
with open("./levir_word2idx.json", "wt") as f:
    json.dump(word2idx, f, indent=4)
