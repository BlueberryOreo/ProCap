import copy
import torch
import logging
import math
import nltk
import numpy as np
import os
import random

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm
from PIL import Image
import h5py

from src.utils import load_json, flat_list_of_lists

log_format = "%(asctime)-10s: %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


class RecursiveCaptionDataset(Dataset):
    PAD_TOKEN = "[PAD]"  # padding of the whole sequence, note
    CLS_TOKEN = "[CLS]"  # leading token of the joint sequence
    SEP_TOKEN = "[SEP]"  # a separator for video and text
    VID_TOKEN = "[VID]"  # used as placeholder in the clip+text joint sequence
    BOS_TOKEN = "[BOS]"  # beginning of the sentence
    EOS_TOKEN = "[EOS]"  # ending of the sentence
    UNK_TOKEN = "[UNK]"
    PAD = 0
    CLS = 1
    SEP = 2
    VID = 3
    BOS = 4
    EOS = 5
    UNK = 6
    IGNORE = -1  # used to calculate loss

    """
    recurrent: if True, return recurrent data
    """
    def __init__(self, dset_name, data_dir, video_feature_dir, word2idx_path,
                 max_t_len, max_v_len, mode="train", recurrent=True, untied=False, tokenizer=None):
        self.dset_name = dset_name
        self.word2idx = load_json(word2idx_path)
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.data_dir = data_dir  # containing training data
        self.video_feature_dir = video_feature_dir  # a set of .h5 files
        self.max_seq_len = max_v_len + max_t_len
        self.max_v_len = max_v_len
        self.max_t_len = max_t_len  # sen
        self.tokenizer = tokenizer

        self.mode = mode
        self.recurrent = recurrent
        self.untied = untied
        assert not (self.recurrent and self.untied), "untied and recurrent cannot be True for both"

        # data entries
        self.data = None

        if mode == "train":
            mode = ["train", "val"]
            # mode = ["train"]
        else:
            mode = [mode]
        
        raw_data = []
        for m in mode:
            raw_data.extend(load_json(os.path.join(data_dir, "reformat_%s.json" % m)))
        # raw_data = load_json(os.path.join(data_dir, "reformat_%s.json" % mode))

        change_path = os.path.join(self.video_feature_dir, self.mode)
        self.change_path = change_path

        image_id = {}
        for v in raw_data:
            image1 = v["img0"]
            image2 = v["img1"]
            img_id = "%s_%s" % (image1, image2)
            image_id[len(image_id)] = v

        # if mode == "train":
        #     for d in raw_data:
        #         for sent in d["sentences"]:
        #             item = {"img_id": d["img_id"], "sentences": sent}
        #             image_id[len(image_id)] = item
        # else:
        #     for d in raw_data:
        #         item = {"img_id": d["img_id"], "sentences": d["sentences"]}
        #         image_id[len(image_id)] = item

        self.image_id = image_id
        self.num_sens = None  # number of sentence for each video, set in self._load_data()

    def __len__(self):
        return len(self.image_id)

    def __getitem__(self, index):
        image_item = self.image_id[index]
        image1 = image_item["img0"]
        image2 = image_item["img1"]
        image_id = "%s_%s" % (image1, image2)
        image_sent = image_item["sents"]

        change_feat = self.load_visual_feature(image_id)

        change_ids, change_labels, change_mask = self.load_sentence(image_sent)

        return image_id, change_feat, change_ids, change_labels, change_mask

    def load_visual_feature(self, image_id):
        change_path = os.path.join(self.change_path, '%s.npy' % image_id)
        change_feat = np.load(open(change_path, 'rb')).astype(np.float32)

        raw_change_feat = np.zeros((self.max_v_len + self.max_t_len, change_feat.shape[1])).astype(np.float32)

        raw_change_feat[:self.max_v_len] = change_feat
        return change_feat


    def load_sentence(self, captions):
        sentence = random.sample(captions, 1)[0]
        text_tokens, text_mask = self._tokenize_pad_sentence(sentence)
        
        if self.tokenizer is not None:
            input_ids = text_tokens
        else:
            input_tokens = text_tokens
            input_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in input_tokens]
            
        visual_tokens = [self.VID_TOKEN] * self.max_v_len
        visual_mask = [1] * len(visual_tokens)
        input_labels = [self.IGNORE if m == 0 else tid for tid, m in zip(input_ids, text_mask)][1:] + [self.IGNORE]

        mask = visual_mask + text_mask

        return np.array(input_ids).astype(np.int64), \
               np.array(input_labels).astype(np.int64), \
               np.array(mask).astype(np.float32), \
               sentence


    def convert_example_to_features(self, example):
        """example single snetence
        {"name": str,
         "duration": float,
         "timestamp": [st(float), ed(float)],
         "sentence": str
        } or
        {"name": str,
         "duration": float,
         "timestamps": list([st(float), ed(float)]),
         "sentences": list(str)
        }
        """
        name = example["name"]
        video_name = name[2:] if self.dset_name == "anet" else name
        feat_path_resnet = os.path.join(self.video_feature_dir, "{}_resnet.npy".format(video_name))
        feat_path_bn = os.path.join(self.video_feature_dir, "{}_bn.npy".format(video_name))
        video_feature = np.concatenate([np.load(feat_path_resnet), np.load(feat_path_bn)], axis=1)
        if self.recurrent:  # recurrent
            num_sen = len(example["sentences"])
            single_video_features = []
            single_video_meta = []
            for clip_idx in range(num_sen):
                cur_data, cur_meta = self.clip_sentence_to_feature(example["name"],
                                                                   example["timestamps"][clip_idx],
                                                                   example["sentences"][clip_idx],
                                                                   video_feature)
                single_video_features.append(cur_data)
                single_video_meta.append(cur_meta)
            return single_video_features, single_video_meta
        else:  # single sentence
            clip_dataloader = self.clip_sentence_to_feature_untied \
                if self.untied else self.clip_sentence_to_feature
            cur_data, cur_meta = clip_dataloader(example["name"],
                                                 example["timestamp"],
                                                 example["sentence"],
                                                 video_feature)
            return cur_data, cur_meta

    def clip_sentence_to_feature(self, name, timestamp, sentence, video_feature):
        """ make features for a single clip-sentence pair.
        [CLS], [VID], ..., [VID], [SEP], [BOS], [WORD], ..., [WORD], [EOS]
        Args:
            name: str,
            timestamp: [float, float]
            sentence: str
            video_feature: np array
        """
        frm2sec = self.frame_to_second[name[2:]] if self.dset_name == "anet" else self.frame_to_second[name]

        # video + text tokens
        feat, video_tokens, video_mask = self._load_indexed_video_feature(video_feature, timestamp, frm2sec)
        text_tokens, text_mask = self._tokenize_pad_sentence(sentence)

        input_tokens = video_tokens + text_tokens

        input_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in input_tokens]
        # shifted right, `-1` is ignored when calculating CrossEntropy Loss
        input_labels = \
            [self.IGNORE] * len(video_tokens) + \
            [self.IGNORE if m == 0 else tid for tid, m in zip(input_ids[-len(text_mask):], text_mask)][1:] + \
            [self.IGNORE]
        input_mask = video_mask + text_mask
        token_type_ids = [0] * self.max_v_len + [1] * self.max_t_len

        data = dict(
            name=name,
            input_tokens=input_tokens,
            # model inputs
            input_ids=np.array(input_ids).astype(np.int64),
            input_labels=np.array(input_labels).astype(np.int64),
            input_mask=np.array(input_mask).astype(np.float32),
            token_type_ids=np.array(token_type_ids).astype(np.int64),
            video_feature=feat.astype(np.float32)
        )
        meta = dict(
            # meta
            name=name,
            timestamp=timestamp,
            sentence=sentence,
        )
        return data, meta

    def clip_sentence_to_feature_untied(self, name, timestamp, sentence, raw_video_feature):
        """ make features for a single clip-sentence pair.
        [CLS], [VID], ..., [VID], [SEP], [BOS], [WORD], ..., [WORD], [EOS]
        Args:
            name: str,
            timestamp: [float, float]
            sentence: str
            raw_video_feature: np array, N x D, for the whole video
        """
        frm2sec = self.frame_to_second[name[2:]] if self.dset_name == "anet" else self.frame_to_second[name]

        # video + text tokens
        video_feature, video_mask = self._load_indexed_video_feature_untied(raw_video_feature, timestamp, frm2sec)
        text_tokens, text_mask = self._tokenize_pad_sentence(sentence)

        text_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in text_tokens]
        # shifted right, `-1` is ignored when calculating CrossEntropy Loss
        text_labels = [self.IGNORE if m == 0 else tid for tid, m in zip(text_ids, text_mask)][1:] + [self.IGNORE]

        data = dict(
            name=name,
            text_tokens=text_tokens,
            # model inputs
            text_ids=np.array(text_ids).astype(np.int64),
            text_mask=np.array(text_mask).astype(np.float32),
            text_labels=np.array(text_labels).astype(np.int64),
            video_feature=video_feature.astype(np.float32),
            video_mask=np.array(video_mask).astype(np.float32),
        )
        meta = dict(
            # meta
            name=name,
            timestamp=timestamp,
            sentence=sentence,
        )
        return data, meta

    def _load_indexed_video_feature(self, raw_feat, timestamp, frm2sec):
        """ [CLS], [VID], ..., [VID], [SEP], [PAD], ..., [PAD],
        All non-PAD tokens are valid, will have a mask value of 1.
        Returns:
            feat is padded to length of (self.max_v_len + self.max_t_len,)
            video_tokens: self.max_v_len
            mask: self.max_v_len
        """
        max_v_l = self.max_v_len - 2
        feat_len = len(raw_feat)
        st, ed = self._convert_to_feat_index_st_ed(feat_len, timestamp, frm2sec)
        indexed_feat_len = ed - st + 1

        feat = np.zeros((self.max_v_len + self.max_t_len, raw_feat.shape[1]))  # includes [CLS], [SEP]
        if indexed_feat_len > max_v_l:
            downsamlp_indices = np.linspace(st, ed, max_v_l, endpoint=True).astype(np.int).tolist()
            assert max(downsamlp_indices) < feat_len
            feat[1:max_v_l+1] = raw_feat[downsamlp_indices]  # truncate, sample???

            video_tokens = [self.CLS_TOKEN] + [self.VID_TOKEN] * max_v_l + [self.SEP_TOKEN]
            mask = [1] * (max_v_l + 2)
        else:
            valid_l = ed - st + 1
            feat[1:valid_l+1] = raw_feat[st:ed + 1]
            video_tokens = [self.CLS_TOKEN] + [self.VID_TOKEN] * valid_l + \
                           [self.SEP_TOKEN] + [self.PAD_TOKEN] * (max_v_l - valid_l)
            mask = [1] * (valid_l + 2) + [0] * (max_v_l - valid_l)
        return feat, video_tokens, mask

    def _load_indexed_video_feature_untied(self, raw_feat, timestamp, frm2sec):
        """ Untied version: [VID], ..., [VID], [PAD], ..., [PAD], len == max_v_len
        Returns:
            feat is padded to length of (self.max_v_len,)
            mask: self.max_v_len, with 1 indicates valid bits, 0 indicates padding
        """
        max_v_l = self.max_v_len
        feat_len = len(raw_feat)
        st, ed = self._convert_to_feat_index_st_ed(feat_len, timestamp, frm2sec)
        indexed_feat_len = ed - st + 1

        if indexed_feat_len > max_v_l:
            downsamlp_indices = np.linspace(st, ed, max_v_l, endpoint=True).astype(np.int).tolist()
            assert max(downsamlp_indices) < feat_len
            feat = raw_feat[downsamlp_indices]  # truncate, sample???
            mask = [1] * max_v_l  # no padding
        else:
            feat = np.zeros((max_v_l, raw_feat.shape[1]))  # only video features and padding
            valid_l = ed - st + 1
            feat[:valid_l] = raw_feat[st:ed + 1]
            mask = [1] * valid_l + [0] * (max_v_l - valid_l)
        return feat, mask

    def _tokenize_pad_sentence(self, sentence):
        """[BOS], [WORD1], [WORD2], ..., [WORDN], [EOS], [PAD], ..., [PAD], len == max_t_len
        All non-PAD values are valid, with a mask value of 1
        """
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                sentence + self.tokenizer.eos_token,
                max_length=self.max_t_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            sentence_tokens = tokens['input_ids'].squeeze(0).tolist()
            mask = tokens['attention_mask'].squeeze(0).tolist()
        else:
            max_t_len = self.max_t_len
            sentence_tokens = nltk.tokenize.word_tokenize(sentence.lower())[:max_t_len - 2]
            sentence_tokens = [self.BOS_TOKEN] + sentence_tokens + [self.EOS_TOKEN]

            # pad
            valid_l = len(sentence_tokens)
            mask = [1] * valid_l + [0] * (max_t_len - valid_l)
            sentence_tokens += [self.PAD_TOKEN] * (max_t_len - valid_l)
        return sentence_tokens, mask

    def convert_ids_to_sentence(self, ids, rm_padding=True, return_sentence_only=True):
        """A list of token ids"""
        if self.tokenizer is not None:
            words = self.tokenizer.decode(ids, skip_special_tokens=rm_padding)
            return words
        
        rm_padding = True if return_sentence_only else rm_padding
        if rm_padding:
            raw_words = [self.idx2word[wid] for wid in ids if wid not in [self.PAD, self.IGNORE]]
        else:
            raw_words = [self.idx2word[wid] for wid in ids if wid != self.IGNORE]

        # get only sentences, the tokens between `[BOS]` and the first `[EOS]`
        if return_sentence_only:
            words = []
            for w in raw_words[1:]:  # no [BOS]
                if w != self.EOS_TOKEN:
                    words.append(w)
                else:
                    break
        else:
            words = raw_words
        return " ".join(words)


class VideoCaptionDataset(RecursiveCaptionDataset):
    # video_feature_dir: the directory to the raw images or video frames
    def __init__(self, dset_name, data_dir, video_feature_dir, word2idx_path,
                 max_t_len, max_v_len, mode="train", recurrent=True, untied=False, tokenizer=None, 
                 filtered=False, filter_file_path=None, max_k=2, num_frames=9, transform=None, return_raw_sentence=False):
        self.dset_name = dset_name
        self.word2idx = load_json(word2idx_path)
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.data_dir = data_dir  # containing training data
        self.video_feature_dir = video_feature_dir  # a set of .h5 files
        self.max_seq_len = max_v_len + max_t_len
        self.max_v_len = max_v_len
        self.max_t_len = max_t_len  # sentence
        self.tokenizer = tokenizer

        self.mode = mode
        self.recurrent = recurrent
        self.untied = untied
        assert not (self.recurrent and self.untied), "untied and recurrent cannot be True for both"
        
        if mode == "train":
            # mode = ["train", "val"]
            mode = ["train"]
        else:
            mode = [mode]

        # self.raw_data = load_json(os.path.join(data_dir, "%s.json" % mode))
        self.raw_data = []
        for m in mode:
            self.raw_data.extend(load_json(os.path.join(data_dir, "reformat_%s.json" % m)))

        image_id = {}
        for d in self.raw_data:
            image_id[d["uid"]] = len(image_id)
        
        self.image_id = image_id

        self.filtered = filtered
        self.max_k = max_k
        self.num_frames = num_frames
        if self.filtered:
            # raise NotImplementedError("filtered is not implemented yet")
            if filter_file_path is None:
                raise ValueError("filter_file_path is required when filtered is True")
            
            self.filter_scores = load_json(filter_file_path)

        # self.data = CacheDataMapper(video_feature_dir, self.transform, cache_size=20)
        self.data = h5py.File(os.path.join(video_feature_dir, "edit_videos.h5"), 'r')
        self.transform = transform
        self.return_raw_sentence = return_raw_sentence
    
    def get_filter_indices(self, vid, video_len=9):
        """ Find the changing frames

        Args:
            vid: str, video id, 'xxx.png'
        Returns
            - indices: torch.tensor, filter indices to preserve
        """
        bef_scores = torch.tensor(self.filter_scores[vid]["bef_inter_sim"]).to(torch.float32).mean(dim=0)
        aft_scores = torch.tensor(self.filter_scores[vid]["inter_aft_sim"]).to(torch.float32).mean(dim=0)
        
        scores = (bef_scores - aft_scores) ** 2

        indices = torch.argsort(scores[1: -1], descending=True)[self.max_k: ] + 1
        indices = torch.cat([torch.tensor([0]), torch.sort(indices)[0], torch.tensor([video_len - 1])], dim=0)
        
        return indices
    
    def __getitem__(self, index):
        # _, change_feat = self.data[index]
        # change_feat = change_feat.permute(1, 0, 2, 3) # [C, T, H, W]
        # image_id = self.mapper["idx_to_vid"][str(index)]
        # image_sent = self.raw_data[self.image_id[image_id]]["sents"]
        cap_item = self.raw_data[index]
        image_id = cap_item["uid"]
        # image_sent = random.choice(cap_item["sents"])
        change_feat = torch.from_numpy(self.data[image_id][:]).type(torch.float32) / 255

        if self.filtered:
            filtered_indices = self.get_filter_indices(image_id, self.num_frames)
            change_feat = change_feat[filtered_indices]
        
        if self.transform:
            change_feat = self.transform(change_feat)
            
        change_feat = change_feat.permute(1, 0, 2, 3) # [C, T, H, W]
        
        change_ids, change_labels, change_mask, change_sentence = self.load_sentence(cap_item["sents"])

        if self.return_raw_sentence:
            return image_id, change_feat, change_ids, change_labels, change_mask, change_sentence
        return image_id, change_feat, change_ids, change_labels, change_mask
    
    # def load_visual_feature(self, image_id):
    #     # >>> Load two images
    #     img1 = Image.open(os.path.join(self.video_feature_dir, image_id[0])).convert('RGB')
    #     img2 = Image.open(os.path.join(self.video_feature_dir, image_id[1])).convert('RGB')

    #     img1 = self.transform(img1).unsqueeze(0)
    #     img2 = self.transform(img2).unsqueeze(0)
    #     # <<<
        
    #     input_features = torch.cat([img1, img2], dim=0)
    #     feature = torch.zeros([self.max_v_len, img1.shape[1], img1.shape[2], img1.shape[3]])
    #     feature[:input_features.shape[0]] = input_features
    #     feature = feature.permute(1, 0, 2, 3) # [C, T, H, W]
    #     return feature


def prepare_batch_inputs(batch, device, non_blocking=False):
    batch_inputs = dict()
    bsz = len(batch["name"])
    for k, v in batch.items():
        assert bsz == len(v), (bsz, k, v)
        if isinstance(v, torch.Tensor):
            batch_inputs[k] = v.to(device, non_blocking=non_blocking)
        else:  # all non-tensor values
            batch_inputs[k] = v
    return batch_inputs


def step_collate(padded_batch_step):
    """The same step (clip-sentence pair) from each example"""
    c_batch = dict()
    for key in padded_batch_step[0]:
        value = padded_batch_step[0][key]
        if isinstance(value, list):
            c_batch[key] = [d[key] for d in padded_batch_step]
        else:
            c_batch[key] = default_collate([d[key] for d in padded_batch_step])
    return c_batch


def caption_collate(batch):
    """get rid of unexpected list transpose in default_collate
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py#L66

    HOW to batch clip-sentence pair?
    1) directly copy the last sentence, but do not count them in when back-prop OR
    2) put all -1 to their text token label, treat
    """
    # collect meta
    raw_batch_meta = [e[1] for e in batch]
    batch_meta = []
    for e in raw_batch_meta:
        cur_meta = dict(
            name=None,
            timestamp=[],
            gt_sentence=[]
        )
        for d in e:
            cur_meta["name"] = d["name"]
            cur_meta["timestamp"].append(d["timestamp"])
            cur_meta["gt_sentence"].append(d["sentence"])
        batch_meta.append(cur_meta)

    batch = [e[0] for e in batch]
    # Step1: pad each example to max_n_sen
    max_n_sen = max([len(e) for e in batch])
    raw_step_sizes = []

    padded_batch = []
    padding_clip_sen_data = copy.deepcopy(batch[0][0])  # doesn"t matter which one is used
    padding_clip_sen_data["input_labels"][:] = RecursiveCaptionDataset.IGNORE
    for ele in batch:
        cur_n_sen = len(ele)
        if cur_n_sen < max_n_sen:
            ele = ele + [padding_clip_sen_data] * (max_n_sen - cur_n_sen)
        raw_step_sizes.append(cur_n_sen)
        padded_batch.append(ele)

    # Step2: batching each steps individually in the batches
    collated_step_batch = []
    for step_idx in range(max_n_sen):
        collated_step = step_collate([e[step_idx] for e in padded_batch])
        collated_step_batch.append(collated_step)
    return collated_step_batch, raw_step_sizes, batch_meta


def single_sentence_collate(batch):
    """get rid of unexpected list transpose in default_collate
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py#L66
    """
    # collect meta
    batch_meta = [{"name": e[1]["name"],
                   "timestamp": e[1]["timestamp"],
                   "gt_sentence": e[1]["sentence"]
                   } for e in batch]  # change key
    padded_batch = step_collate([e[0] for e in batch])
    return padded_batch, None, batch_meta
