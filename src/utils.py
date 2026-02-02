import json
import numpy as np
# import pyarrow as pa
# import pyarrow.parquet as pq
from typing import Tuple
import torch
import os
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
import copy

try:
    from src.taming.models.vqgan import VQModel, GumbelVQ
except ImportError:
    pass

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        return json.JSONEncoder.default(self, obj)


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, cls=MyEncoder, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


def save_parsed_args_to_json(parsed_args, file_path, pretty=True):
    args_dict = vars(parsed_args)
    save_json(args_dict, file_path, save_pretty=pretty)


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def set_lr(optimizer, decay_factor):
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * decay_factor


def flat_list_of_lists(l):
    """flatten a list of lists [[1,2], [3,4]] to [1,2,3,4]"""
    return [item for sublist in l for item in sublist]


def count_parameters(model, verbose=True):
    """Count number of parameters in PyTorch model,
    References: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7.

    from utils.utils import count_parameters
    count_parameters(model)
    import sys
    sys.exit(1)
    """
    n_all = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print("Parameter Count: all {:,d}; trainable {:,d}".format(n_all, n_trainable))
    return n_all, n_trainable


def sum_parameters(model, verbose=True):
    """Count number of parameters in PyTorch model,
    References: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7.

    from utils.utils import count_parameters
    count_parameters(model)
    import sys
    sys.exit(1)
    """
    p_sum = sum(p.sum().item() for p in model.parameters())
    if verbose:
        print("Parameter sum {}".format(p_sum))
    return p_sum


def merge_dicts(list_dicts):
    merged_dict = list_dicts[0].copy()
    for i in range(1, len(list_dicts)):
        merged_dict.update(list_dicts[i])
    return merged_dict


def merge_json_files(paths, merged_path):
    merged_dict = merge_dicts([load_json(e) for e in paths])
    save_json(merged_dict, merged_path)


def compute_recall(sim_matrix, k_values=[1, 5, 10]):
    """
    计算 Recall@K
    :param sim_matrix: BxB 的相似矩阵
    :param k_values: 需要计算 Recall 的 k 值列表
    :return: 一个字典，其中键为 k 值，值为 Recall@K
    """
    # Batch size
    B = sim_matrix.shape[0]

    # 初始化结果字典
    recall_at_k = {k: 0 for k in k_values}

    for i in range(B):
        # 对第 i 行进行排序，得到按相似度降序排列的索引
        sorted_indices = np.argsort(-sim_matrix[i])

        # 找到正确匹配的索引
        rank = np.where(sorted_indices == i)[0][0] + 1  # +1 因为索引从 0 开始

        # 更新 Recall@K 统计
        for k in k_values:
            if rank <= k:
                recall_at_k[k] += 1

    # 计算 Recall 的百分比
    recall_at_k = {f"R@{k}": v / B for k, v in recall_at_k.items()}
    return recall_at_k


def compute_ranks(sim_matrix):
    """
    计算 Median Rank 和 Mean Rank
    :param sim_matrix: BxB 的相似矩阵
    :return: median_rank 和 mean_rank
    """
    B = sim_matrix.shape[0]
    ranks = []

    for i in range(B):
        # 对第 i 行进行排序，得到按相似度降序排列的索引
        sorted_indices = np.argsort(-sim_matrix[i])

        # 找到正确匹配的索引
        rank = np.where(sorted_indices == i)[0][0] + 1  # +1 因为索引从 0 开始
        ranks.append(rank)

    # 计算 Median Rank 和 Mean Rank
    median_rank = np.median(ranks)
    mean_rank = np.mean(ranks)

    return median_rank, mean_rank


def evaluate_retrieval(sim_matrix, k_values=[1, 5, 10]):
    """
    评估检索任务的 Recall@K、Median Rank 和 Mean Rank
    :param sim_matrix: BxB 的相似矩阵
    :param k_values: 需要计算 Recall 的 k 值列表
    :return: 评估结果字典，包括 Recall@K、Median Rank 和 Mean Rank
    """
    # 计算 Recall@K
    recall_at_k = compute_recall(sim_matrix, k_values)

    # 计算 Median Rank 和 Mean Rank
    median_rank, mean_rank = compute_ranks(sim_matrix)

    # 综合结果
    results = {
        "recall_at_k": recall_at_k,
        "median_rank": median_rank,
        "mean_rank": mean_rank
    }
    return results


def load_vqgan(config, ckpt_path=None, is_gumbel=False, freeze=True):
    if is_gumbel:
        model = GumbelVQ(**config['params'])
    else:
        model = VQModel(**config['params'])
    if ckpt_path is not None:
        print(f"Lodaing pretrained model from: {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    
    if freeze:
        print("Freezing the parameters in vqgan")
        for name, param in model.named_parameters():
            param.requires_grad = False
    return model.eval()


def load_vqgan_encoder(config, ckpt_path=None, is_gumbel=False, freeze=True):
    if is_gumbel:
        model = GumbelVQ(**config['params'])
    else:
        model = VQModel(**config['params'])
    
    encoder = model.encoder
    if ckpt_path is not None:
        print(f"Lodaing pretrained model from: {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = encoder.load_state_dict(sd, strict=False)
    pass


def get_vae_model(which_vae,
                  vae_params=None,
                  vae_path=None,
                  image_size=None,
                  args=None):
    # if vae_params is given (not None), RESUMING from custom DiscreteVAE(**vae_params)
    # weight loading is handled in dalle model
    if args is not None and args.dalle_path:
        vae_path = None
    if which_vae == 'vqgan1024':
        try:
            from mmvid_pytorch.vae import VQGanVAE1024
        except ImportError:
            from src.mmvid_pytorch.vae import VQGanVAE1024
        vae = VQGanVAE1024(vae_path=vae_path, image_size=image_size)
        vae_params = None
    else:
        # NOTE: See dalle_pytorch if you want to use OpenAI's VAE or custom VAE
        raise NotImplementedError
    return vae, vae_params


def score_generation(anno_file, result_file):
    coco = COCO(anno_file)
    coco_res = coco.loadRes(result_file)

    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.params['image_id'] = coco_res.getImgIds()

    coco_eval.evaluate()
    return copy.deepcopy(coco_eval.eval)

def score_generation_by_type(anno_file, result_file, type_file):
    coco = COCO(anno_file)
    coco_res = coco.loadRes(result_file)
    coco_eval = COCOEvalCap(coco, coco_res)

    type_dict = json.load(open(type_file, 'r'))
    results = {}
    for type, image_ids in type_dict.items():
        image_ids = list(map(lambda x: x.split(".")[0] + "c", image_ids))
        filtered = set(coco_res.getImgIds()).intersection(set(image_ids))
        coco_eval.params['image_id'] = list(filtered)
        coco_eval.evaluate()
        results[type] = copy.deepcopy(coco_eval.eval)

    return results