from pathlib import Path
import random
from random import randint, choice
import os
import pickle
from natsort import natsorted
import numpy as np
from tqdm import tqdm
import PIL
from PIL import Image
import decord
import h5py
import json

decord.bridge.set_bridge("torch")

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as TF

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
VID_EXTENSIONS = ['.mp4', '.avi', '.h5']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def is_video_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in VID_EXTENSIONS)


def to_tensor(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    try:
        image_tensor = TF.to_tensor(image)
    except:
        return None
    return image_tensor


def read_frames_imagestack(video_path, frame_idxs=None):
    imgs = Image.open(video_path).convert('RGB')  # size (W, H)
    imgs = np.array(imgs)  # shape (H, W, C)
    horizontal = imgs.shape[1] > imgs.shape[0]
    shorter, longer = min(imgs.shape[0],
                          imgs.shape[1]), max(imgs.shape[0], imgs.shape[1])
    vlen = longer // shorter
    frames = np.stack(np.split(imgs, vlen, axis=1 if horizontal else 0))
    if frame_idxs:
        frames = frames[frame_idxs, ...]
    frames = torch.from_numpy(frames).permute(
        0, 3, 1, 2).float() / 255  # tensor of shape (T, C, H, W), range (0, 1)
    return frames


def sharpen(scores, mu=0.5):
    sharpened = scores.pow(1.0 / mu)
    return sharpened / sharpened.sum(dim=-1, keepdim=True)


class TextImageDataset(Dataset):
    def __init__(
        self,
        folder,
        text_len=256,
        image_size=128,
        truncate_captions=False,
        resize_ratio=0.75,
        tokenizer=None,
        shuffle=False,
        cache=None,
        image_only=False,
        deterministic=False,
    ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        self.shuffle = shuffle
        self.image_only = image_only
        path = Path(folder)

        cache = path.parent / (path.name +
                               '_local.db') if cache is None else Path(cache)
        if cache is not None and cache.exists():
            with open(cache, 'rb') as f:
                self.keys, self.text_files, self.image_files = pickle.load(f)
        else:
            text_files = [*path.glob('**/*.txt')]
            image_files = [
                *path.glob('**/*.png'), *path.glob('**/*.jpg'),
                *path.glob('**/*.jpeg'), *path.glob('**/*.bmp')
            ]

            text_files = {
                text_file.stem: text_file
                for text_file in text_files
            }
            image_files = {
                image_file.stem: image_file
                for image_file in image_files
            }

            keys = (image_files.keys() & text_files.keys())

            self.keys = list(keys)
            self.text_files = {
                k: v
                for k, v in text_files.items() if k in keys
            }
            self.image_files = {
                k: v
                for k, v in image_files.items() if k in keys
            }
            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump((self.keys, self.text_files, self.image_files),
                                f)

        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        if deterministic:
            self.image_transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB')
                         if img.mode != 'RGB' else img),
                T.Resize(image_size),
                T.CenterCrop(image_size),
                T.ToTensor()
            ])
        else:
            self.image_transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB')
                         if img.mode != 'RGB' else img),
                T.RandomResizedCrop(image_size,
                                    scale=(self.resize_ratio, 1.),
                                    ratio=(1., 1.)),
                T.ToTensor()
            ])

    def __len__(self):
        return len(self.keys)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        key = self.keys[ind]

        text_file = self.text_files[key]
        image_file = self.image_files[key]

        descriptions = text_file.read_text().split('\n')
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            description = choice(descriptions)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        tokenized_text = self.tokenizer.tokenize(
            description, self.text_len, truncate_text=self.truncate_captions
        ).squeeze(0) if self.tokenizer is not None else description
        try:
            image_tensor = self.image_transform(PIL.Image.open(image_file))
        except (PIL.UnidentifiedImageError,
                OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        if self.image_only:
            return image_tensor, 0

        # Success
        return tokenized_text, image_tensor


class TextVideoDataset(Dataset):
    def __init__(
        self,
        folder,
        text_len=256,
        image_size=128,
        truncate_captions=False,
        resize_ratio=0.75,
        tokenizer=None,
        shuffle=False,
        mode='video',
        frame_step=2,
        frame_num=8,
        deterministic=False,
        cache=None,
        return_vc=False,
        video_only=False,
        keys=None,
        return_neg=False,
        drop_sentence=False,
        tokenizer2=None,
        rep_num=1,
        skip_min_len_check=False,
        return_label=False,
        filtered=False,
        filter_file_path=None,
        max_k=None,
    ):
        super().__init__()
        self.mode = mode
        self.shuffle = shuffle

        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        self.tokenizer2 = tokenizer2
        self.rep_num = rep_num
        self.image_size = image_size
        self.return_label = return_label

        path = Path(folder)

        # video
        min_len = 2 # 8
        nframe_num = 2
        self.frame_num = frame_num
        self.frame_step = frame_step
        self.nframe_num = nframe_num
        frame_step_max = self.frame_step
        if skip_min_len_check:
            self.min_len = max(
                min_len, (self.frame_num - 1) * int(self.frame_step * 1.5) + 1)
        else:
            self.min_len = max(min_len,
                               (self.frame_num - 1) * frame_step_max + 1)
        
        self.min_len = 2 # set min_len to 2
        
        self.unbind = False
        self.return_vc = return_vc
        self.return_neg = return_neg
        self.video_only = video_only
        self.drop_sentence = drop_sentence

        dataroot = str(path)
        self.root = dataroot
        video_root = os.path.join(dataroot, 'video')
        text_root = os.path.join(dataroot, 'txt')
        cache = path.parent / (path.name +
                               '_local.pkl') if cache is None else Path(cache)
        if cache is not None and cache.exists():
            with open(cache, 'rb') as f:
                cache_data = pickle.load(f)
            assert (isinstance(cache_data, dict))
            self.keys = cache_data['keys']
            self.texts, self.videos, self.lengths = cache_data[
                'texts'], cache_data['videos'], cache_data['lengths']
        else:
            text_files = os.listdir(text_root)
            text_dict = dict()
            video_dict = dict()
            length_dict = dict()
            keys_list = list()
            for i, video in enumerate(
                    tqdm(os.listdir(video_root), desc="Counting videos")):
                key = video  # no stem
                text = key + '.txt'
                if os.path.isdir(os.path.join(video_root,
                                              key)) and text in text_files:
                    frames = natsorted(
                        os.listdir(os.path.join(video_root, key)))
                else:
                    continue
                frame_list = []
                for j, frame_name in enumerate(frames):
                    if is_image_file(os.path.join(video_root, key,
                                                  frame_name)):
                        frame_list.append(
                            os.path.join('video', key, frame_name))
                if len(frame_list) > 0:
                    # add entry
                    keys_list.append(key)
                    text_dict[key] = os.path.join('txt', text)
                    video_dict[key] = frame_list
                    length_dict[key] = len(frame_list)
                # clear
                frame_list = frames = None
            self.keys = keys_list
            self.texts, self.videos, self.lengths = text_dict, video_dict, length_dict
            assert len(self.keys) > 0
            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump(
                        {
                            'root': dataroot,
                            'keys': self.keys,
                            'texts': self.texts,
                            'videos': self.videos,
                            'lengths': self.lengths,
                        }, f)

        if return_neg:
            attr_cache = path.parent / (path.name + '_attr_dict.pkl')
            if attr_cache.exists():
                with open(attr_cache, 'rb') as f:
                    self.attr_dict = pickle.load(f)
            else:
                attr_dict = {'text': {}}
                for k in tqdm(self.keys):
                    descriptions = Path(os.path.join(
                        self.root, self.texts[k])).read_text().split('\n')
                    description = descriptions[0]
                    text = description.lower().replace(',', '')
                    if text in attr_dict['text']:
                        attr_dict['text'][text].append(k)
                    else:
                        attr_dict['text'][text] = [k]
                with open(attr_cache, 'wb') as f:
                    pickle.dump(attr_dict, f)
                self.attr_dict = attr_dict

        # Filter out videos that are too short
        keys_keep = [k for k in self.keys if self.lengths[k] >= self.min_len]
        if keys is not None:
            keys_keep = list(set(keys_keep) & set(keys))
        self.texts = {k: self.texts[k] for k in keys_keep}
        self.videos = {k: self.videos[k] for k in keys_keep}
        self.lengths = {k: self.lengths[k] for k in keys_keep}
        self.keys = sorted(keys_keep)

        if return_neg:
            attr_dict = {}
            for attr_type in self.attr_dict:
                attr_dict[attr_type] = {}
                for attr in self.attr_dict[attr_type].keys():
                    attr_dict[attr_type][attr] = list(
                        set(self.attr_dict[attr_type][attr]) & set(keys_keep))
            self.attr_dict = attr_dict

        if self.mode == 'video':
            self._dataset_length = len(self.keys)
        elif self.mode == '1frame':
            self._dataset_length = len(self.keys)
        else:
            raise NotImplementedError
        
        # print(len(self.keys), "videos in total", len(self.videos), len(self))

        # image transform
        self.deterministic = deterministic
        if deterministic:
            self.image_transform = T.Compose([
                T.Resize(image_size),
                T.CenterCrop(image_size),
            ])
        else:
            self.image_transform = T.Compose([
                # transforms.ToTensor(),  # this should be done in __getitem__
                # T.RandomHorizontalFlip(),
                T.Resize(image_size),
                # T.CenterCrop(image_size),
                T.RandomResizedCrop(image_size,
                                    scale=(self.resize_ratio, 1.),
                                    ratio=(1., 1.)),
            ])
        
        
        self.filtered = filtered
        if self.filtered:
            if filter_file_path is None:
                raise ValueError("filter_file_path is required when filtered is True")
            if not os.path.exists(filter_file_path):
                raise ValueError("filter_file_path does not exist")
            
            self.filter_scores = json.load(open(filter_file_path, "r", encoding="utf-8"))
        self.max_k = max_k

    def __len__(self):
        return self._dataset_length

    def _get_label(self, key):
        label_file = Path(
            os.path.join(self.root, self.texts[key].replace('txt/', 'label/')))
        label = label_file.read_text().rstrip()
        return int(label)
    
    def get_filter_indices(self, vid, video_len=9, cat: str=None):
        """ Find the changing frames

        Args:
            vid: str, video id, 'xxx.png'
            cat: str, category for CLEVR dataset, 'sc_videos' or 'nsc_videos'
        Returns
            - indices: torch.tensor, filter indices to preserve
        """
        if cat is not None:
            if cat.startswith("sc"):
                bef_scores = torch.tensor(self.filter_scores[vid]["sc_sim_matrixes"]["bef_inter_sim"]).to(torch.float32).mean(dim=0)
                aft_scores = torch.tensor(self.filter_scores[vid]["sc_sim_matrixes"]["inter_aft_sim"]).to(torch.float32).mean(dim=0)
            elif cat.startswith("nsc"):
                bef_scores = torch.tensor(self.filter_scores[vid]["nsc_sim_matrixes"]["bef_inter_sim"]).to(torch.float32).mean(dim=0)
                aft_scores = torch.tensor(self.filter_scores[vid]["nsc_sim_matrixes"]["inter_aft_sim"]).to(torch.float32).mean(dim=0)
            else:
                bef_scores = torch.tensor(self.filter_scores[vid]["bef_inter_sim"]).to(torch.float32).mean(dim=0)
                aft_scores = torch.tensor(self.filter_scores[vid]["inter_aft_sim"]).to(torch.float32).mean(dim=0)
        else:
            raise NotImplementedError("cat is not implemented yet")
        
        scores = (bef_scores - aft_scores) ** 2

        # >>> filter method 2
        # start = 1
        # delta_scores = torch.abs(scores[start: -start - 1] - scores[start+1: -start])
        # indices = torch.argsort(delta_scores, descending=True)[:self.max_k] + start
        # <<<

        # >>> filter method 1
        # indices = torch.argsort(scores[1: -1], descending=False)[:self.max_k] + 1
        # <<<

        # >>> random filter method 1
        scores = 1 - torch.softmax(scores[1: -1], dim=0)
        # scores = 1 - sharpen(scores[1: -1])
        indices = torch.multinomial(scores, num_samples=self.max_k, replacement=False) + 1
        # <<<

        # >>> random filter method 2
        # start = 1
        # delta_scores = torch.abs(scores[start: -start - 1] - scores[start+1: -start])
        # indices = torch.multinomial(delta_scores, num_samples=self.max_k, replacement=False) + start
        # <<<

        # >>> Random filter
        # indices = torch.randperm(video_len - 2)[:self.max_k] + 1
        # <<<

        indices = torch.sort(indices)[0]

        # if indices[0] == 0:
        #     indices[0] = 1

        indices = torch.cat([torch.tensor([0]), indices, torch.tensor([video_len - 1])], dim=0)
        
        return indices

    def _get_video(self, index, frame_step=None):
        if frame_step is None:
            frame_step = self.frame_step
        key = self.keys[index]
        video_len = self.lengths[key]
        start_idx = 0 if self.deterministic else random.randint(
            0, video_len - (self.frame_num - 1) * frame_step - 1)  # inclusive
        frames = []
        if self.filtered:
            frame_idx = self.get_filter_indices(key, video_len, cat="cat")
        else:
            if self.rep_num == 1:
                frame_idx = range(start_idx,
                                start_idx + self.frame_num * frame_step,
                                frame_step)
            else:
                m_step = int(
                    (video_len - (self.frame_num - 1) * frame_step) / self.rep_num)
                frame_idx = []
                for m in range(self.rep_num):
                    start_idx = m_step * m
                    frame_idx += list(
                        range(start_idx, start_idx + self.frame_num * frame_step,
                            frame_step))
        for i in frame_idx:
            img = Image.open(os.path.join(self.root, self.videos[key][i]))
            img = T.Resize((self.image_size, self.image_size))(img)
            frames.append(to_tensor(img))  # to_tensor done here
        frames = torch.stack(frames, 0)
        
        frames = self.image_transform(frames)
        if True:
            idx = 0 if self.deterministic else random.randint(0, video_len - 1)
            visual = Image.open(os.path.join(self.root, self.videos[key][idx]))
            visual = self.image_transform(to_tensor(visual))
            return frames, key, visual
        return frames, key

    def _get_1frame(self, index):
        # randomly pick one frame in each video, this is different from _get_nframe when n==1
        key = self.keys[index]
        video_len = self.lengths[key]
        keep_ratio = 0.75
        delta_r = int(video_len * (1 - keep_ratio) / 2)
        delta_l = int(video_len * (1 - keep_ratio)) - delta_r
        frame_idx = random.randint(delta_l, video_len - delta_r - 1)
        frame = Image.open(os.path.join(self.root,
                                        self.videos[key][frame_idx]))
        frame = self.image_transform(to_tensor(frame))
        if True:
            idx = random.randint(delta_l, video_len - delta_r - 1)
            visual = Image.open(os.path.join(self.root, self.videos[key][idx]))
            visual = self.image_transform(to_tensor(visual))
            return frame, key, visual
        return frame, key

    def _get_image(self, index):
        # copied from MoCoGAN, consider all frames as a image dataset
        if index == 0:
            video_id = 0
            frame_id = 0
        else:
            video_id = np.searchsorted(self.cumsum, index) - 1
            frame_id = index - self.cumsum[video_id] - 1
        key = self.keys[video_id]
        frame = Image.open(os.path.join(self.root, self.videos[key][frame_id]))
        frame = self.image_transform(
            to_tensor(frame))  # no ToTensor in transform
        return frame, key

    def _get_nframe(self, index):
        # consider all consecutive n-frames as one dataset
        if index == 0:
            video_id = 0
            frame_id = 0
        else:
            video_id = np.searchsorted(self.cumsumn, index) - 1
            frame_id = index - self.cumsumn[video_id] - 1
        key = self.keys[video_id]
        frames = []
        for i in range(self.nframe_num):
            frame = Image.open(
                os.path.join(self.root, self.videos[key][frame_id + i]))
            frames.append(to_tensor(frame))
        frames = torch.stack(frames, 0)
        frames = self.image_transform(frames)
        return frames, key

    def __len__(self):
        return self._dataset_length

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        visual = 0
        if self.mode == 'video':
            image_tensor, key, visual = self._get_video(ind)
        elif self.mode == '1frame':
            image_tensor, key, visual = self._get_1frame(ind)
        elif self.mode == 'image':
            image_tensor, key = self._get_image(ind)
        elif self.mode == 'nframe':
            image_tensor, key = self._get_nframe(ind)

        if self.video_only:
            description = 'dummy text'
            tokenized_text = self.tokenizer.tokenize(
                description,
                self.text_len,
                truncate_text=self.truncate_captions,
            ).squeeze(0) if self.tokenizer is not None else description
            if self.return_label:
                label = self._get_label(key)
                return tokenized_text, image_tensor, label
            return tokenized_text, image_tensor, visual

        text_file = Path(os.path.join(self.root, self.texts[key]))
        descriptions = text_file.read_text().split('\n')
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            if self.deterministic:
                description = descriptions[0]
            else:
                description = choice(descriptions)
            if self.drop_sentence:
                description_ = description.split('. ')
                if self.deterministic:
                    description = description_[0]
                    if 'and' in description:
                        description = description.split(', ')[0] + '.'
                else:
                    # num_drop = random.randint(0, min(len(description_)-1, 3))
                    num_drop = random.randint(0, len(description_) - 1)
                    for _ in range(num_drop):
                        description_.remove(random.choice(description_))
                    description = '. '.join(description_)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        tokenized_text = self.tokenizer.tokenize(
            description,
            self.text_len,
            truncate_text=self.truncate_captions,
        ).squeeze(0) if self.tokenizer is not None else description

        if self.return_neg:
            text = descriptions[0].lower().replace(',', '')
            text_ = choice(
                list(set(self.attr_dict['text'].keys()) - set([text])))
            key_ = choice(self.attr_dict['text'][text_])
            text_file = Path(os.path.join(self.root, self.texts[key_]))
            descriptions = text_file.read_text().split('\n')
            descriptions = list(filter(lambda t: len(t) > 0, descriptions))
            description_ = choice(descriptions)
            tokenized_text_ = self.tokenizer.tokenize(
                description_,
                self.text_len,
                truncate_text=self.truncate_captions,
            ).squeeze(0) if self.tokenizer is not None else description_
            visual_ = 0
            return tokenized_text, image_tensor, visual, visual_, tokenized_text_

        return tokenized_text, image_tensor, visual


def sample_frames(num_frames, vlen, sample='rand', fix_start=None):
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen,
                            num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError

    return frame_idxs


def read_frames_decord(video_path, num_frames, sample='rand', fix_start=None):
    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    frame_idxs = sample_frames(num_frames,
                               vlen,
                               sample=sample,
                               fix_start=fix_start)
    frames = video_reader.get_batch(frame_idxs)
    frames = frames.float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs


class TextMP4Dataset(Dataset):
    def __init__(
        self,
        folder,
        text_len=256,
        image_size=128,
        truncate_captions=False,
        resize_ratio=0.75,
        tokenizer=None,
        shuffle=False,
        mode='video',
        frame_step=2,
        frame_num=8,
        deterministic=False,
        image_only=False,
        cache=None,
        return_vc=False,
        return_text=False,
        return_label=False,
        keys=None,
        video_only=False,
    ):
        super().__init__()
        self.mode = mode
        self.shuffle = shuffle

        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer

        path = Path(folder)

        # video
        min_len = 8
        nframe_num = 2
        self.frame_num = frame_num
        self.frame_step = frame_step
        self.nframe_num = nframe_num
        self.min_len = max(min_len, (self.frame_num - 1) * self.frame_step + 1)
        self.unbind = False
        self.return_vc = return_vc
        self.return_text = return_text
        self.return_label = return_label
        self.image_only = image_only
        self.video_only = video_only

        dataroot = str(path)
        self.root = dataroot
        video_root = os.path.join(dataroot, 'video')
        text_root = os.path.join(dataroot, 'txt')
        self.has_label = (Path(dataroot) / 'label').exists()
        self.has_visual = (Path(dataroot) / 'visual').exists()

        # Build or load cache
        video_files = os.listdir(video_root)
        cache = path.parent / (path.name +
                               '_local.pkl') if cache is None else Path(cache)
        if cache is not None and cache.exists():
            with open(cache, 'rb') as f:
                cache_data = pickle.load(f)
            assert (isinstance(cache_data, dict))
            self.keys = cache_data['keys']
            self.texts, self.videos, self.lengths = cache_data[
                'texts'], cache_data['videos'], cache_data['lengths']
        else:
            text_files = os.listdir(text_root)
            text_dict = dict()
            video_dict = dict()
            length_dict = dict()
            keys_list = list()
            for i, video in enumerate(tqdm(video_files,
                                           desc="Counting videos")):
                videoid = Path(video).stem
                text = videoid + '.txt'
                if is_video_file(video) and text in text_files:
                    # get video info
                    video_path = os.path.join(self.root, 'video', video)
                    try:
                        video_reader = decord.VideoReader(video_path,
                                                          num_threads=1)
                        vlen = len(video_reader)
                        # add entry
                        keys_list.append(videoid)
                        text_dict[videoid] = os.path.join('txt', text)
                        video_dict[videoid] = os.path.join('video', video)
                        length_dict[videoid] = vlen
                    except:
                        continue
                else:
                    continue
            self.keys = keys_list
            self.texts, self.videos, self.lengths = text_dict, video_dict, length_dict
            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump(
                        {
                            'root': dataroot,
                            'keys': self.keys,
                            'texts': self.texts,
                            'videos': self.videos,
                            'lengths': self.lengths,
                        }, f)

        # Filter out videos that are too short
        keys_keep = [k for k in self.keys if self.lengths[k] >= self.min_len]
        if keys is not None:
            keys_keep = list(set(keys_keep) & set(keys))
        self.texts = {k: self.texts[k] for k in keys_keep}
        self.videos = {k: self.videos[k] for k in keys_keep}
        self.lengths = {k: self.lengths[k] for k in keys_keep}
        self.keys = keys_keep

        if self.mode == 'video':
            self._dataset_length = len(self.keys)
        elif self.mode == '1frame':
            self._dataset_length = len(self.keys)
        else:
            raise NotImplementedError

        # image transform
        if deterministic:
            self.image_transform = T.Compose([
                T.Resize(image_size),
                T.CenterCrop(image_size),
            ])
        else:
            self.image_transform = T.Compose([
                # transforms.ToTensor(),  # this should be done in __getitem__
                # T.RandomHorizontalFlip(),
                T.Resize(image_size),
                # T.CenterCrop(image_size),
                T.RandomResizedCrop(image_size,
                                    scale=(self.resize_ratio, 1.),
                                    ratio=(1., 1.)),
            ])

    def _get_label(self, key):
        label_file = Path(
            os.path.join(self.root, self.texts[key].replace('txt/', 'label/')))
        label = label_file.read_text().rstrip()
        return int(label)

    def _get_video(self, index):
        key = self.keys[index]
        video_len = self.lengths[key]
        start_idx = random.randint(0, video_len -
                                   (self.frame_num - 1) * self.frame_step -
                                   1)  # inclusive
        video_path = os.path.join(self.root, self.videos[key])
        video_reader = decord.VideoReader(video_path, num_threads=1)
        frame_idxs = range(start_idx,
                           start_idx + self.frame_num * self.frame_step,
                           self.frame_step)
        frames = video_reader.get_batch(frame_idxs)
        frames = frames.float() / 255  # to [0, 1]
        frames = frames.permute(0, 3, 1, 2)
        frames = self.image_transform(frames)
        if True:
            idx = random.randint(0, video_len - 1)
            visual = video_reader.get_batch([idx])
            visual = visual.permute(0, 3, 1, 2).squeeze().float() / 255
            visual = self.image_transform(visual)
            return frames, key, visual
        return frames, key

    def _get_1frame(self, index):
        # randomly pick one frame in each video, this is different from _get_nframe when n==1
        key = self.keys[index]
        video_len = self.lengths[key]
        keep_ratio = 0.75
        delta_r = int(video_len * (1 - keep_ratio) / 2)
        delta_l = int(video_len * (1 - keep_ratio)) - delta_r
        frame_idx = random.randint(delta_l, video_len - delta_r - 1)
        video_path = os.path.join(self.root, self.videos[key])
        video_reader = decord.VideoReader(video_path, num_threads=1)
        frame = video_reader.get_batch([frame_idx])
        frame = frame.permute(0, 3, 1, 2).squeeze().float() / 255
        frame = self.image_transform(frame)
        if True:
            idx = random.randint(delta_l, video_len - delta_r - 1)
            visual = video_reader.get_batch([idx])
            visual = visual.permute(0, 3, 1, 2).squeeze().float() / 255
            visual = self.image_transform(visual)
            return frame, key, visual
        return frame, key

    def __len__(self):
        return self._dataset_length

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        visual = 0
        if self.mode == 'video':
            image_tensor, key, visual = self._get_video(ind)
        elif self.mode == '1frame':
            image_tensor, key, visual = self._get_1frame(ind)
        elif self.mode == 'image':
            image_tensor, key = self._get_image(ind)
        elif self.mode == 'nframe':
            image_tensor, key = self._get_nframe(ind)

        # if self.image_only:
        #     return image_tensor, 0
        if self.video_only:
            description = 'dummy text'
            tokenized_text = self.tokenizer.tokenize(
                description,
                self.text_len,
                truncate_text=self.truncate_captions,
            ).squeeze(0) if self.tokenizer is not None else description
            return tokenized_text, image_tensor, visual

        text_file = Path(os.path.join(self.root, self.texts[key]))
        descriptions = text_file.read_text().split('\n')
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            description = choice(descriptions)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        tokenized_text = self.tokenizer.tokenize(
            description,
            self.text_len,
            truncate_text=self.truncate_captions,
        ).squeeze(0) if self.tokenizer is not None else description

        if self.return_label:
            label = self._get_label(key)
            return tokenized_text, image_tensor, label

        # Success
        # if self.return_vc:
        #     return tokenized_text, image_tensor, visual
        # if self.return_text:
        #     return tokenized_text, image_tensor, description
        # return tokenized_text, image_tensor

        return tokenized_text, image_tensor, visual


class TextImageStackDataset(Dataset):
    def __init__(
        self,
        folder,
        text_len=256,
        image_size=128,
        truncate_captions=False,
        resize_ratio=0.75,
        tokenizer=None,
        shuffle=False,
        mode='video',
        frame_step=2,
        frame_num=8,
        deterministic=False,
        image_only=False,
        cache=None,
        return_vc=False,
        return_text=False,
        return_label=False,
        keys=None,
        no_cache=False,
    ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        self.mode = mode
        self.shuffle = shuffle

        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer

        path = Path(folder)

        # video
        min_len = 8
        nframe_num = 2
        self.frame_num = frame_num
        self.frame_step = frame_step
        self.nframe_num = nframe_num
        self.min_len = max(min_len, (self.frame_num - 1) * self.frame_step + 1)
        self.unbind = False
        self.return_vc = return_vc
        self.return_text = return_text
        self.return_label = return_label
        self.image_only = image_only

        dataroot = str(path)
        self.root = dataroot
        video_root = os.path.join(dataroot, 'video')
        text_root = os.path.join(dataroot, 'txt')
        self.has_label = (Path(dataroot) / 'label').exists()
        self.has_visual = (Path(dataroot) / 'visual').exists()

        # Build or load cache
        video_files = os.listdir(video_root)
        cache = path.parent / (path.name +
                               '_local.pkl') if cache is None else Path(cache)
        if cache is not None and cache.exists():
            with open(cache, 'rb') as f:
                cache_data = pickle.load(f)
            assert (isinstance(cache_data, dict))
            self.keys = cache_data['keys']
            self.texts, self.videos, self.lengths = cache_data[
                'texts'], cache_data['videos'], cache_data['lengths']
        else:
            text_files = os.listdir(text_root)
            text_dict = dict()
            video_dict = dict()
            length_dict = dict()
            keys_list = list()
            for i, video in enumerate(tqdm(video_files,
                                           desc="Counting videos")):
                videoid = Path(video).stem
                text = videoid + '.txt'
                if is_image_file(video) and text in text_files:
                    # get video info
                    video_path = os.path.join(self.root, 'video', video)
                    try:
                        imgs = Image.open(video_path).convert('RGB')
                        imgs = np.array(imgs)  # [H, W, C]
                        # horizontal = imgs.shape[1] > imgs.shape[0]
                        shorter, longer = min(imgs.shape[0],
                                              imgs.shape[1]), max(
                                                  imgs.shape[0], imgs.shape[1])
                        vlen = longer // shorter
                        # frames = np.split(imgs, vlen, axis=1 if horizontal else 0)

                        # add entry
                        keys_list.append(videoid)
                        text_dict[videoid] = os.path.join('txt', text)
                        video_dict[videoid] = os.path.join('video', video)
                        length_dict[videoid] = vlen
                    except:
                        continue
                else:
                    continue
            self.keys = keys_list
            self.texts, self.videos, self.lengths = text_dict, video_dict, length_dict
            if not no_cache and cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump(
                        {
                            'root': dataroot,
                            'keys': self.keys,
                            'texts': self.texts,
                            'videos': self.videos,
                            'lengths': self.lengths,
                        }, f)

        # Filter out videos that are too short
        keys_keep = [k for k in self.keys if self.lengths[k] >= self.min_len]
        if keys is not None:
            keys_keep = list(set(keys_keep) & set(keys))
        self.texts = {k: self.texts[k] for k in keys_keep}
        self.videos = {k: self.videos[k] for k in keys_keep}
        self.lengths = {k: self.lengths[k] for k in keys_keep}
        self.keys = keys_keep

        if self.mode == 'video':
            self._dataset_length = len(self.keys)
        elif self.mode == '1frame':
            self._dataset_length = len(self.keys)
        else:
            raise NotImplementedError

        # image transform
        if deterministic:
            self.image_transform = T.Compose([
                T.Resize(image_size),
                T.CenterCrop(image_size),
            ])
        else:
            self.image_transform = T.Compose([
                # transforms.ToTensor(),  # this should be done in __getitem__
                # T.RandomHorizontalFlip(),
                T.Resize(image_size),
                # T.CenterCrop(image_size),
                T.RandomResizedCrop(image_size,
                                    scale=(self.resize_ratio, 1.),
                                    ratio=(1., 1.)),
            ])

    def _get_video(self, index):
        key = self.keys[index]
        video_len = self.lengths[key]
        start_idx = random.randint(0, video_len -
                                   (self.frame_num - 1) * self.frame_step -
                                   1)  # inclusive
        frame_idxs = range(start_idx,
                           start_idx + self.frame_num * self.frame_step,
                           self.frame_step)
        video_path = os.path.join(self.root, self.videos[key])
        frames = read_frames_imagestack(video_path, frame_idxs)
        image_tensor = self.image_transform(frames)
        if not self.return_vc:
            return image_tensor, key, None
        if self.has_visual:
            idx = random.randint(0, video_len - 1)
            visual_path = os.path.join(self.root, 'visual',
                                       Path(self.videos[key]).name)
            visuals = read_frames_imagestack(visual_path, [idx])
            visual_tensor = self.image_transform(visuals[0])
        else:
            idx = random.randint(0, video_len - 1)
            visual = frames[idx]
            visual_tensor = self.image_transform(visual)
        return image_tensor, key, visual_tensor

    def _get_1frame(self, index):
        # randomly pick one frame in each video, this is different from _get_nframe when n==1
        key = self.keys[index]
        video_len = self.lengths[key]
        keep_ratio = 0.75
        delta_r = int(video_len * (1 - keep_ratio) / 2)
        delta_l = int(video_len * (1 - keep_ratio)) - delta_r
        frame_idx = random.randint(delta_l, video_len - delta_r - 1)
        video_path = os.path.join(self.root, self.videos[key])
        frames = read_frames_imagestack(video_path, None)
        frame = frames[frame_idx]
        image_tensor = self.image_transform(frame)
        if not self.return_vc:
            return image_tensor, key, None
        if self.has_visual:
            idx = random.randint(0, video_len - 1)
            visual_path = os.path.join(self.root, 'visual',
                                       Path(self.videos[key]).name)
            visuals = read_frames_imagestack(visual_path, [idx])
            visual_tensor = self.image_transform(visuals[0])
        else:
            idx = random.randint(0, video_len - 1)
            visual = frames[idx]
            visual_tensor = self.image_transform(visual)
        return image_tensor, key, visual_tensor

    def __len__(self):
        return self._dataset_length

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def _get_label(self, key):
        label_path = os.path.join(self.root,
                                  self.texts[key]).replace('txt/', 'label/')
        label = [int(s) for s in Path(label_path).read_text().split(',')]
        return np.array(label)

    def __getitem__(self, ind):
        visual_tensor = 0
        if self.mode == 'video':
            image_tensor, key, visual_tensor = self._get_video(ind)
        elif self.mode == '1frame':
            image_tensor, key, visual_tensor = self._get_1frame(ind)
        elif self.mode == 'image':
            image_tensor, key = self._get_image(ind)
        elif self.mode == 'nframe':
            image_tensor, key = self._get_nframe(ind)

        if self.image_only:
            return image_tensor, 0

        text_file = Path(os.path.join(self.root, self.texts[key]))
        descriptions = text_file.read_text().split('\n')
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            description = choice(descriptions)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        tokenized_text = self.tokenizer.tokenize(
            description,
            self.text_len,
            truncate_text=self.truncate_captions,
        ).squeeze(0) if self.tokenizer is not None else description

        # Success
        if self.return_label:
            label = self._get_label(key)
            return tokenized_text, image_tensor, label
        if self.return_vc:
            return tokenized_text, image_tensor, visual_tensor
        if self.return_text:
            return tokenized_text, image_tensor, description
        return tokenized_text, image_tensor


class TextH5Dataset(Dataset):
    def __init__(
        self,
        folder,
        text_len=256,
        image_size=128,
        truncate_captions=False,
        resize_ratio=0.75,
        tokenizer=None,
        shuffle=False,
        mode='video',
        frame_step=2,
        frame_num=8,
        deterministic=False,
        cache=None,
        return_vc=False,
        video_only=False,
        keys=None,
        return_neg=False,
        drop_sentence=False,
        tokenizer2=None,
        rep_num=1,
        skip_min_len_check=False,
        return_label=False,
        filtered=False,
        filter_file_path=None,
        max_k=None,
        sample_strategy='multi-modal',
        return_detail=False,
    ):
        super().__init__()
        self.mode = mode
        self.shuffle = shuffle

        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        self.tokenizer2 = tokenizer2
        self.rep_num = rep_num
        self.image_size = image_size
        self.return_label = return_label
        self.return_detail = return_detail

        path = Path(folder)

        # video
        min_len = 8
        nframe_num = 2
        self.frame_num = frame_num
        self.frame_step = frame_step
        self.nframe_num = nframe_num
        frame_step_max = self.frame_step
        if skip_min_len_check:
            self.min_len = max(
                min_len, (self.frame_num - 1) * int(self.frame_step * 1.5) + 1)
        else:
            self.min_len = max(min_len,
                               (self.frame_num - 1) * frame_step_max + 1)
        self.unbind = False
        self.return_vc = return_vc
        self.return_neg = return_neg
        self.video_only = video_only
        self.drop_sentence = drop_sentence

        dataroot = str(path)
        self.root = dataroot
        video_root = os.path.join(dataroot, 'video') # dataroot/video -> video path
        text_root = os.path.join(dataroot, 'txt') # dataroot/txt -> text path
        cache = path.parent / (path.name +
                               '_local.pkl') if cache is None else Path(cache)
        """
            video dataset format:
            - video
                - video1
                    - frame1.jpg
                    - frame2.jpg
                    - ...
                - video2
                    - frame1.jpg
                    - frame2.jpg
                    - ...
                - ...
            - txt
                - video1.txt
                - video2.txt
                - ...
            
            keys_list: [video1, video2, ...]
            text_dict: {video1: 'video1.txt', video2: 'video2.txt', ...}
            video_dict: {video1: ['frame1.jpg', 'frame2.jpg', ...], video2: ['frame1.jpg', 'frame2.jpg', ...], ...}
            length_dict: {video1: <frame_num>, video2: <frame_num>, ...}

            Need to convert to:
                videos: h5py files
                texts: json files
            
            Target h5py video dataset format:
            - video
                - videos1.h5
                - videos2.h5
                - ...
            - txt
                - videos1.json
                - videos2.json
                - ...
            
            keys_list: ["videos1/key1", "videos1/key2", ...] + ["videos2/key1", "videos2/key2", ...] + ...
            texts: {videos1: dict(videos1.json), videos2: dict(videos2.json), ...}
            videos: {videos1: path/to/videos1.h5, videos2: path/to/videos2.h5, ...}
            lengths: {videos1: <frame_num>, videos2: <frame_num>, ...}
        """
        captions_files = os.listdir(text_root)
        video_files = os.listdir(video_root)
        text_dict = dict()
        video_dict = dict()
        for captions_file in captions_files:
            with open(os.path.join(text_root, captions_file), "r", encoding="utf-8") as f:
                k_prefix = captions_file.split(".")[0]
                text_dict[k_prefix] = json.load(f)
        self.texts = text_dict
        
        for video_file in video_files:
            video_path = os.path.join(video_root, video_file)
            # video = h5py.File(video_path, "r")
            k_prefix = video_file.split(".")[0]
            video_dict[k_prefix] = video_path
        self.videos = video_dict

        if cache is not None and cache.exists():
            with open(cache, 'rb') as f:
                cache_data = pickle.load(f)
            assert (isinstance(cache_data, dict))
            self.keys, self.lengths = cache_data['keys'], cache_data['lengths']
        else:
            length_dict = dict()
            keys_list = []

            # print("Counting videos")
            # for k_prefix in video_dict.keys():
            #     video = video_dict[k_prefix]
            #     for k in tqdm(list(video.keys())):
            #         keys_list.append(f"{k_prefix}/{k}")
            #         length_dict[f"{k_prefix}/{k}"] = len(video[k][:])
            print("Counting videos")
            with h5py.File(video_dict[list(video_dict.keys())[0]], "r") as video:
                video_len = len(video[list(video.keys())[0]][:])
            print(f"Video length: {video_len}")
            for k_prefix in text_dict.keys():
                for k in tqdm(text_dict[k_prefix].keys()):
                    keys_list.append(f"{k_prefix}/{k}")
                    length_dict[f"{k_prefix}/{k}"] = video_len
            
            self.keys, self.lengths = keys_list, length_dict
            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump(
                        {
                            'root': dataroot,
                            'keys': self.keys,
                            'lengths': self.lengths,
                        }, f)

        if return_neg:
            raise NotImplementedError("return_neg is not implemented yet")
            # attr_cache = path.parent / (path.name + '_attr_dict.pkl')
            # if attr_cache.exists():
            #     with open(attr_cache, 'rb') as f:
            #         self.attr_dict = pickle.load(f)
            # else:
            #     attr_dict = {'text': {}}
            #     for k in tqdm(self.keys):
            #         descriptions = Path(os.path.join(
            #             self.root, self.texts[k])).read_text().split('\n')
            #         description = descriptions[0]
            #         text = description.lower().replace(',', '')
            #         if text in attr_dict['text']:
            #             attr_dict['text'][text].append(k)
            #         else:
            #             attr_dict['text'][text] = [k]
            #     with open(attr_cache, 'wb') as f:
            #         pickle.dump(attr_dict, f)
            #     self.attr_dict = attr_dict

        if return_neg:
            raise NotImplementedError("return_neg is not implemented yet")
            # attr_dict = {}
            # for attr_type in self.attr_dict:
            #     attr_dict[attr_type] = {}
            #     for attr in self.attr_dict[attr_type].keys():
            #         attr_dict[attr_type][attr] = list(
            #             set(self.attr_dict[attr_type][attr]) & set(keys_keep))
            # self.attr_dict = attr_dict

        if self.mode == 'video':
            self._dataset_length = len(self.keys)
        elif self.mode == '1frame':
            self._dataset_length = len(self.keys)
        else:
            raise NotImplementedError

        # image transform
        self.deterministic = deterministic
        if deterministic:
            self.image_transform = T.Compose([
                T.Resize(image_size),
                T.CenterCrop(image_size),
            ])
        else:
            self.image_transform = T.Compose([
                # transforms.ToTensor(),  # this should be done in __getitem__
                # T.RandomHorizontalFlip(),
                T.Resize(image_size),
                # T.CenterCrop(image_size),
                T.RandomResizedCrop(image_size,
                                    scale=(self.resize_ratio, 1.),
                                    ratio=(1., 1.)),
            ])
        

        self.filtered = filtered
        assert sample_strategy in ['multi-modal', 'visual', 'mixed', 'random'], "sample_strategy must be one of ['multi-modal', 'visual', 'mixed', 'random']"
        self.sample_strategy = sample_strategy
        print("** Using sample strategy:", self.sample_strategy)
        
        if self.filtered and self.sample_strategy != 'random':
            if filter_file_path is None:
                raise ValueError("filter_file_path is required when filtered is True")
            if not os.path.exists(filter_file_path):
                raise ValueError("filter_file_path does not exist")
            
            self.filter_scores = json.load(open(filter_file_path, "r", encoding="utf-8"))
            
            self.min_val = [float("inf"), float("inf")] # [sc, (nsc)]
            self.max_val = [float("-inf"), float("-inf")] # [sc, (nsc)]
            self.dino_min_val = [float("inf"), float("inf")] # [sc, (nsc)]
            self.dino_max_val = [float("-inf"), float("-inf")] # [sc, (nsc)]
            for vid in self.filter_scores.keys():
                if "bef_inter_sim" in self.filter_scores[vid]:
                    bef_scores = torch.tensor(self.filter_scores[vid]["bef_inter_sim"]).to(torch.float32).mean(dim=0).unsqueeze(0)
                    aft_scores = torch.tensor(self.filter_scores[vid]["inter_aft_sim"]).to(torch.float32).mean(dim=0).unsqueeze(0)
                else:
                    sc_bef_scores = torch.tensor(self.filter_scores[vid]["sc_sim_matrixes"]["bef_inter_sim"]).to(torch.float32).mean(dim=0)
                    sc_aft_scores = torch.tensor(self.filter_scores[vid]["sc_sim_matrixes"]["inter_aft_sim"]).to(torch.float32).mean(dim=0)
                    nsc_bef_scores = torch.tensor(self.filter_scores[vid]["nsc_sim_matrixes"]["bef_inter_sim"]).to(torch.float32).mean(dim=0)
                    nsc_aft_scores = torch.tensor(self.filter_scores[vid]["nsc_sim_matrixes"]["inter_aft_sim"]).to(torch.float32).mean(dim=0)
                    bef_scores = torch.stack([sc_bef_scores, nsc_bef_scores], dim=0)
                    aft_scores = torch.stack([sc_aft_scores, nsc_aft_scores], dim=0)

                scores = (bef_scores - aft_scores) ** 2
                for i in range(scores.shape[0]):
                    self.min_val[i] = min(self.min_val[i], scores[i].min().item())
                    self.max_val[i] = max(self.max_val[i], scores[i].max().item())
                
                # if "bef_inter_dino_score" in self.filter_scores[vid] and "inter_aft_dino_score" in self.filter_scores[vid]:
                #     dino_bef_scores = torch.tensor(self.filter_scores[vid]["bef_inter_dino_score"]).to(torch.float32).mean(dim=0)
                #     dino_aft_scores = torch.tensor(self.filter_scores[vid]["inter_aft_dino_score"]).to(torch.float32).mean(dim=0)
                #     dino_scores = (dino_bef_scores - dino_aft_scores) ** 2
                #     self.dino_min_val = min(self.dino_min_val, dino_scores.min().item())
                #     self.dino_max_val = max(self.dino_max_val, dino_scores.max().item())
                if self.filter_scores[vid].get("bef_inter_dino_score", None) and self.filter_scores[vid].get("inter_aft_dino_score", None):
                    dino_bef_scores = torch.tensor(self.filter_scores[vid]["bef_inter_dino_score"]).to(torch.float32).mean(dim=0).unsqueeze(0)
                    dino_aft_scores = torch.tensor(self.filter_scores[vid]["inter_aft_dino_score"]).to(torch.float32).mean(dim=0).unsqueeze(0)
                elif self.filter_scores[vid].get("sc_sim_matrixes", {}).get("bef_inter_dino_score", None) and self.filter_scores[vid].get("sc_sim_matrixes", {}).get("inter_aft_dino_score", None):
                    sc_dino_bef_scores = torch.tensor(self.filter_scores[vid]["sc_sim_matrixes"]["bef_inter_dino_score"]).to(torch.float32).mean(dim=0)
                    sc_dino_aft_scores = torch.tensor(self.filter_scores[vid]["sc_sim_matrixes"]["inter_aft_dino_score"]).to(torch.float32).mean(dim=0)
                    nsc_dino_bef_scores = torch.tensor(self.filter_scores[vid]["nsc_sim_matrixes"]["bef_inter_dino_score"]).to(torch.float32).mean(dim=0)
                    nsc_dino_aft_scores = torch.tensor(self.filter_scores[vid]["nsc_sim_matrixes"]["inter_aft_dino_score"]).to(torch.float32).mean(dim=0)
                    dino_bef_scores = torch.stack([sc_dino_bef_scores, nsc_dino_bef_scores], dim=0)
                    dino_aft_scores = torch.stack([sc_dino_aft_scores, nsc_dino_aft_scores], dim=0)
                else:
                    dino_bef_scores = torch.tensor([], dtype=torch.float32)
                    dino_aft_scores = torch.tensor([], dtype=torch.float32)
                
                for i in range(dino_bef_scores.shape[0]):
                    dino_scores = (dino_bef_scores[i] - dino_aft_scores[i]) ** 2
                    self.dino_min_val[i] = min(self.dino_min_val[i], dino_scores.min().item())
                    self.dino_max_val[i] = max(self.dino_max_val[i], dino_scores.max().item())
        else:
            self.filter_scores = None
            self.min_val = None
            self.max_val = None
            self.dino_min_val = None
            self.dino_max_val = None
            
        self.max_k = max_k
    
    def get_filter_indices(self, vid, video_len=9, cat: str=None):
        """ Find the changing frames

        Args:
            vid: str, video id, 'xxx.png'
            cat: str, category for CLEVR dataset, 'sc_videos' or 'nsc_videos'
        Returns
            - indices: torch.tensor, filter indices to preserve
        """
        if self.sample_strategy != "random":
            if cat is not None:
                if cat.startswith("sc"):
                    min_val, max_val = self.min_val[0], self.max_val[0]
                    dino_min_val, dino_max_val = self.dino_min_val[0], self.dino_max_val[0]
                    filter_scores = self.filter_scores[vid]["sc_sim_matrixes"]
                    # bef_scores = torch.tensor(self.filter_scores[vid]["sc_sim_matrixes"]["bef_inter_sim"]).to(torch.float32).mean(dim=0)
                    # aft_scores = torch.tensor(self.filter_scores[vid]["sc_sim_matrixes"]["inter_aft_sim"]).to(torch.float32).mean(dim=0)
                elif cat.startswith("nsc"):
                    min_val, max_val = self.min_val[1], self.max_val[1]
                    dino_min_val, dino_max_val = self.dino_min_val[1], self.dino_max_val[1]
                    filter_scores = self.filter_scores[vid]["nsc_sim_matrixes"]
                    # bef_scores = torch.tensor(self.filter_scores[vid]["nsc_sim_matrixes"]["bef_inter_sim"]).to(torch.float32).mean(dim=0)
                    # aft_scores = torch.tensor(self.filter_scores[vid]["nsc_sim_matrixes"]["inter_aft_sim"]).to(torch.float32).mean(dim=0)
                else:
                    filter_scores = self.filter_scores[vid]
                    min_val, max_val = self.min_val[0], self.max_val[0]
                    dino_min_val, dino_max_val = self.dino_min_val[0], self.dino_max_val[0]
                    # bef_scores = torch.tensor(self.filter_scores[vid]["bef_inter_sim"]).to(torch.float32).mean(dim=0)
                    # aft_scores = torch.tensor(self.filter_scores[vid]["inter_aft_sim"]).to(torch.float32).mean(dim=0)
                    # bef_dino_scores = torch.tensor(self.filter_scores[vid].get("bef_inter_dino_score", [])).to(torch.float32)
                    # aft_dino_scores = torch.tensor(self.filter_scores[vid].get("inter_aft_dino_score", [])).to(torch.float32)

                bef_scores = torch.tensor(filter_scores["bef_inter_sim"]).to(torch.float32).mean(dim=0)
                aft_scores = torch.tensor(filter_scores["inter_aft_sim"]).to(torch.float32).mean(dim=0)
                if filter_scores.get("bef_inter_dino_score", None) and filter_scores.get("inter_aft_dino_score", None):
                    bef_dino_scores = torch.tensor(filter_scores.get("bef_inter_dino_score")).to(torch.float32)
                    aft_dino_scores = torch.tensor(filter_scores.get("inter_aft_dino_score")).to(torch.float32)
            else:
                raise NotImplementedError("cat is not implemented yet")
            
            scores = (bef_scores - aft_scores) ** 2
        else:
            scores = 0
        
        if self.max_k > 0:

            # >>> filter method 2
            # start = 1
            # delta_scores = torch.abs(scores[start: -start - 1] - scores[start+1: -start])
            # indices = torch.argsort(delta_scores, descending=True)[:self.max_k] + start
            # <<<

            # >>> filter method 1
            # indices = torch.argsort(scores[1: -1], descending=False)[:self.max_k] + 1
            # <<<

            # >>> random filter method 1
            if self.sample_strategy == 'multi-modal':
                scores = 1 - torch.softmax(scores[1: -1], dim=0)
                # scores = 1 - sharpen(scores[1: -1])
                indices = torch.multinomial(scores, num_samples=self.max_k, replacement=False) + 1
            # <<<

            # >>> random filter method 2
            # start = 1
            # delta_scores = sharpen(torch.abs(scores[start: -start - 1] - scores[start+1: -start]))
            # indices = torch.multinomial(delta_scores, num_samples=self.max_k, replacement=False) + start
            # <<<
            
            # >>> random filter method 3 (multi-modal + visual)
            if self.sample_strategy == 'mixed':
                assert bef_dino_scores.shape[0] > 0, "bef_dino_scores must not be empty"
                assert aft_dino_scores.shape[0] > 0, "aft_dino_scores must not be empty"
                assert bef_dino_scores.shape == aft_dino_scores.shape, "bef_dino_scores and aft_dino_scores must have the same shape"
                
                scores = (scores - min_val) / (max_val - min_val)
                dino_scores = (bef_dino_scores - aft_dino_scores) ** 2
                dino_scores = (dino_scores - dino_min_val) / (dino_max_val - dino_min_val)
                scores = 0.7 * scores + 0.3 * dino_scores
                scores = 1 - torch.softmax(scores[1: -1], dim=0)
                indices = torch.multinomial(scores, num_samples=self.max_k, replacement=False) + 1
            # <<<
            
            # >>> random filter method 4 (visual only)
            if self.sample_strategy == 'visual':
                assert bef_dino_scores.shape[0] > 0, "bef_dino_scores must not be empty"
                assert aft_dino_scores.shape[0] > 0, "aft_dino_scores must not be empty"
                assert bef_dino_scores.shape == aft_dino_scores.shape, "bef_dino_scores and aft_dino_scores must have the same shape"
                dino_scores = (bef_dino_scores - aft_dino_scores) ** 2
                # dino_scores = (dino_scores - dino_min_val) / (dino_max_val - dino_min_val)
                dino_scores = 1 - torch.softmax(dino_scores[1: -1], dim=0)
                indices = torch.multinomial(dino_scores, num_samples=self.max_k, replacement=False) + 1
            # <<<

            # >>> Random filter
            if self.sample_strategy == 'random':
                indices = torch.randperm(video_len - 2)[:self.max_k] + 1
            # <<<

            indices = torch.sort(indices)[0]
        else:
            indices = torch.tensor([], dtype=torch.int32)

        # if indices[0] == 0:
        #     indices[0] = 1

        indices = torch.cat([torch.tensor([0]), indices, torch.tensor([video_len - 1])], dim=0)
        
        return indices
    
    def _get_video_from_h5(self, k_prefix, k_suffix):
        video_path = self.videos[k_prefix]
        with h5py.File(video_path, "r") as video:
            video = video[k_suffix][:].transpose(0, 2, 3, 1)
        return video

    def _get_label(self, key):
        raise NotImplementedError("get_label is not implemented yet")
        # label_file = Path(
        #     os.path.join(self.root, self.texts[key].replace('txt/', 'label/')))
        # label = label_file.read_text().rstrip()
        # return int(label)

    def _get_video(self, index, frame_step=None):
        if frame_step is None:
            frame_step = self.frame_step
        key = self.keys[index]
        k_prefix, k_suffix = key.split("/")
        video_len = self.lengths[key]
        frames = []
        if self.filtered:
            frame_idx = self.get_filter_indices(k_suffix, video_len, cat=k_prefix)
        else:
            start_idx = 0 if self.deterministic else random.randint(
                0, video_len - (self.frame_num - 1) * frame_step - 1)  # inclusive
            if self.rep_num == 1:
                frame_idx = range(start_idx,
                                start_idx + self.frame_num * frame_step,
                                frame_step)
            else:
                m_step = int(
                    (video_len - (self.frame_num - 1) * frame_step) / self.rep_num)
                frame_idx = []
                for m in range(self.rep_num):
                    start_idx = m_step * m
                    frame_idx += list(
                        range(start_idx, start_idx + self.frame_num * frame_step,
                            frame_step))
        
        video = self._get_video_from_h5(k_prefix, k_suffix)  # get all frames, type: np.ndarray, shape: [T, C, H, W]
        # print(type(video), video.shape)
        # print(frame_idx)
        for i in frame_idx:
            img = Image.fromarray(video[i])
            img = T.Resize((self.image_size, self.image_size))(img)
            frames.append(to_tensor(img))  # to_tensor done here
        frames = torch.stack(frames, 0)
        frames = self.image_transform(frames)
        if True:
            idx = 0 if self.deterministic else random.randint(0, video_len - 1)
            visual = Image.fromarray(video[idx])
            visual = self.image_transform(to_tensor(visual))
            return frames, key, visual
        return frames, key

    def _get_1frame(self, index):
        # randomly pick one frame in each video, this is different from _get_nframe when n==1
        key = self.keys[index]
        k_prefix, k_suffix = key.split("/")
        video_len = self.lengths[key]
        keep_ratio = 0.75
        delta_r = int(video_len * (1 - keep_ratio) / 2)
        delta_l = int(video_len * (1 - keep_ratio)) - delta_r
        frame_idx = random.randint(delta_l, video_len - delta_r - 1)
        video = self._get_video_from_h5(k_prefix, k_suffix)
        frame = Image.fromarray(video[frame_idx])
        frame = self.image_transform(to_tensor(frame))
        if True:
            idx = random.randint(delta_l, video_len - delta_r - 1)
            visual = Image.fromarray(video[idx])
            visual = self.image_transform(to_tensor(visual))
            return frame, key, visual
        return frame, key

    def _get_image(self, index):
        # copied from MoCoGAN, consider all frames as a image dataset
        if index == 0:
            video_id = 0
            frame_id = 0
        else:
            video_id = np.searchsorted(self.cumsum, index) - 1
            frame_id = index - self.cumsum[video_id] - 1
        key = self.keys[video_id]
        k_prefix, k_suffix = key.split("/")
        frame = Image.fromarray(self._get_video_from_h5(k_prefix, k_suffix)[frame_id])
        frame = self.image_transform(
            to_tensor(frame))  # no ToTensor in transform
        return frame, key

    def _get_nframe(self, index):
        # consider all consecutive n-frames as one dataset
        if index == 0:
            video_id = 0
            frame_id = 0
        else:
            video_id = np.searchsorted(self.cumsumn, index) - 1
            frame_id = index - self.cumsumn[video_id] - 1
        key = self.keys[video_id]
        k_prefix, k_suffix = key.split("/")
        frames = []
        for i in range(self.nframe_num):
            frame = Image.fromarray(self._get_video_from_h5(k_prefix, k_suffix)[frame_id + i])
            frames.append(to_tensor(frame))
        frames = torch.stack(frames, 0)
        frames = self.image_transform(frames)
        return frames, key

    def __len__(self):
        return self._dataset_length

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        visual = 0
        if self.mode == 'video':
            image_tensor, key, visual = self._get_video(ind)
        elif self.mode == '1frame':
            image_tensor, key, visual = self._get_1frame(ind)
        elif self.mode == 'image':
            image_tensor, key = self._get_image(ind)
        elif self.mode == 'nframe':
            image_tensor, key = self._get_nframe(ind)

        if self.video_only:
            description = 'dummy text'
            tokenized_text = self.tokenizer.tokenize(
                description,
                self.text_len,
                truncate_text=self.truncate_captions,
            ).squeeze(0) if self.tokenizer is not None else description
            if self.return_label:
                label = self._get_label(key)
                return tokenized_text, image_tensor, label
            return tokenized_text, image_tensor, visual

        k_prefix, k_suffix = key.split("/")
        descriptions = self.texts[k_prefix][k_suffix] # list of descriptions
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            if self.deterministic:
                description = descriptions[0]
            else:
                description = choice(descriptions)
            if self.drop_sentence:
                description_ = description.split('. ')
                if self.deterministic:
                    description = description_[0]
                    if 'and' in description:
                        description = description.split(', ')[0] + '.'
                else:
                    # num_drop = random.randint(0, min(len(description_)-1, 3))
                    num_drop = random.randint(0, len(description_) - 1)
                    for _ in range(num_drop):
                        description_.remove(random.choice(description_))
                    description = '. '.join(description_)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load sample {key}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        tokenized_text = self.tokenizer.tokenize(
            description,
            self.text_len,
            truncate_text=self.truncate_captions,
        ).squeeze(0) if self.tokenizer is not None else description

        if self.return_neg:
            raise NotImplementedError("return_neg is not implemented yet")
            # text = descriptions[0].lower().replace(',', '')
            # text_ = choice(
            #     list(set(self.attr_dict['text'].keys()) - set([text])))
            # key_ = choice(self.attr_dict['text'][text_])
            # text_file = Path(os.path.join(self.root, self.texts[key_]))
            # descriptions = text_file.read_text().split('\n')
            # descriptions = list(filter(lambda t: len(t) > 0, descriptions))
            # description_ = choice(descriptions)
            # tokenized_text_ = self.tokenizer.tokenize(
            #     description_,
            #     self.text_len,
            #     truncate_text=self.truncate_captions,
            # ).squeeze(0) if self.tokenizer is not None else description_
            # visual_ = 0
            # return tokenized_text, image_tensor, visual, visual_, tokenized_text_

        if self.return_detail:
            return tokenized_text, image_tensor, visual, description, key
        
        return tokenized_text, image_tensor, visual
    
