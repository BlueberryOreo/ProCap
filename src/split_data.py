from src.utils import load_json, save_json
import os


data_dir = '../densevid_eval/clevr_data'

# splits = load_json(os.path.join(data_dir, 'splits.json'))
# change_captions = load_json(os.path.join(data_dir, 'change_captions.json'))
# no_change_captions = load_json(os.path.join(data_dir, 'no_change_captions.json'))
#
#
# for e in ['train', 'test', 'val']:
#     subset = splits[e]
#     data_dict = {}
#     for key, value in change_captions.items():
#         name = key.split('.')[0].split('_')[-1]
#         if int(name) in subset:
#             data_dict[name] = value
#     save_json(data_dict, '%s_change_captions.json' % e)
#
#     data_dict = {}
#     for key, value in no_change_captions.items():
#         name = key.split('.')[0].split('_')[-1]
#         if int(name) in subset:
#             data_dict[name] = value
#     save_json(data_dict, '%s_no_change_captions.json' % e)

vocab_name = '../cache/vocab.json'

vocab = load_json(vocab_name)

new_vocab = {
    "[PAD]": 0,
    "[CLS]": 1,
    "[SEP]": 2,
    "[VID]": 3,
    "[BOS]": 4,
    "[EOS]": 5,
    "[UNK]": 6,
}

vocab_list = list(vocab.keys())[4:]

for v in vocab_list:
    new_vocab[v] = len(new_vocab)

save_json(new_vocab, '../cache/clevr_word2idx.json', save_pretty=True)