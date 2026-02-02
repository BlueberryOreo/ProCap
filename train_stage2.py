"""
This script handles the training process.
"""

import argparse
import math
import time

import random
import numpy as np
import os
import json
import subprocess
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize
from src.rtransformer.optimization import BertAdam, EMA
from src.translator import Translator
from src.translate import run_translate
from src.evaluate import evaluate
from src.utils import save_parsed_args_to_json, save_json, load_json, \
    count_parameters, merge_dicts
from src.mmvid_idc import MMVID_IDC
from peft import LoraConfig, get_peft_model

from ruamel.yaml import YAML

from easydict import EasyDict as EDict
from tensorboardX import SummaryWriter
import logging

from accelerate import Accelerator, DistributedDataParallelKwargs
logger = logging.getLogger(__name__)

# ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
# accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
accelerator = Accelerator()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="./config/config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save_model", type=str, default="model")
    parser.add_argument("--save_mode", type=str, default="best", choices=["all", "best"])
    parser.add_argument("--res_root_dir", type=str, default="./results")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--pretrained_model", type=str, default=None)
    parser.add_argument("--use_fi_frames", action="store_true")
    parser.add_argument("--eval", action="store_true", help="Run evaluation only, no training")
    parser.add_argument("--lr_finetune", type=float, default=2e-5, help="learning rate for finetuning mmvid")

    opt = parser.parse_args()

    opt.cuda = not opt.no_cuda

    config = YAML().load(open(opt.config, "r"))
    for k, v in config.items():
        if opt.__dict__.get(k, None):
            continue
        setattr(opt, k, v if not isinstance(v, dict) else EDict(v))
    
        # make paths
    opt.res_dir = os.path.join(
        opt.res_root_dir, "_".join(
            [opt.dset_name, 
             time.strftime("%Y_%m_%d_%H_%M"),
             "seed{}".format(opt.seed),
             "ema{}".format(opt.ema_decay),
            #  "stage{}".format(opt.training_stage),
             "mmvid"
            ]))
    if opt.debug:
        opt.res_dir = opt.res_dir + "_debug"
    if opt.eval:
        opt.res_dir = opt.res_dir + "_eval"

    if os.path.exists(opt.res_dir) and os.listdir(opt.res_dir):
        raise ValueError("File exists {}".format(opt.res_dir))
    elif not os.path.exists(opt.res_dir):
        os.makedirs(opt.res_dir, exist_ok=True)
    
    opt.log = os.path.join(opt.res_dir, opt.save_model)
    opt.save_model = os.path.join(opt.res_dir, opt.save_model)
    opt.pin_memory = not opt.no_pin_memory

    return opt


def cal_performance(pred, gold):
    pred = pred.max(2)[1].contiguous().view(-1)
    gold = gold.contiguous().view(-1)
    valid_label_mask = gold.ne(VCDataset.IGNORE)
    pred_correct_mask = pred.eq(gold)
    n_correct = pred_correct_mask.masked_select(valid_label_mask).sum().item()
    return n_correct


def eval_language_metrics(checkpoint, eval_data_loader, opt, model=None, eval_mode="val"):
    """eval_mode can only be set to `val` here, as setting to `test` is cheating
    0, run inference
    1, Get METEOR, BLEU1-4, CIDEr scores
    2, Get vocab size, sentence length
    """
    translator = Translator(opt, checkpoint, model=model)
    json_res = run_translate(eval_data_loader, translator, opt=opt, is_main_process=accelerator.is_main_process)
    res_filepath = os.path.abspath(opt.save_model + "_tmp_greedy_pred_{}_{}.json".format(eval_mode, accelerator.state.local_process_index))
    save_json(json_res, res_filepath, save_pretty=True)
    
    tps = sum(json_res["tps"][1: -1]) / len(json_res["tps"][1: -1])

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger.info(f"Average TPS: {tps:.2f}, Average SPC: {sum(json_res['spc'][1:-1]) / len(json_res['spc'][1:-1]):.2f}")
        res_filepath = os.path.abspath(opt.save_model + "_tmp_greedy_pred_{}.json".format(eval_mode))
        logger.info(f"Merging results to {res_filepath}.")
        all_res = {
            "version": "VERSION 1.0",
            "results": {},
        }
        for pid in range(accelerator.state.num_processes):
            tmp_res_filepath = res_filepath.replace(".json", f"_{pid}.json")
            all_res["results"].update(load_json(tmp_res_filepath)["results"])
            logger.info(f"Merged {tmp_res_filepath}.")
            os.remove(tmp_res_filepath)
        
        save_json(all_res, res_filepath, save_pretty=True)
        
        reference_file_path = os.path.join(opt.data_dir, "total_change_captions_reformat.json")
        all_metrics = evaluate(opt, all_res["results"], reference_file_path)

        # save results
        logger.info("Finished eval {}.".format(eval_mode))

        # metric_filepaths = [lang_filepath, stat_filepath, rep_filepath]
        # all_metrics = merge_dicts([load_json(e) for e in metric_filepaths])

        all_metrics_filepath = res_filepath.replace(".json", "_all_metrics.json")
        save_json(all_metrics, all_metrics_filepath, save_pretty=True)
        all_metrics = all_metrics["total_results"]
    else:
        all_metrics = None
        all_metrics_filepath = None
    return all_metrics, [res_filepath, all_metrics_filepath]


def train_epoch(model, training_data, optimizer, ema, device, opt, writer, epoch):
    torch.cuda.empty_cache()
    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        config = model.module.config
    else:
        config = model.config

    torch.autograd.set_detect_anomaly(True)
    for batch_idx, batch in tqdm(enumerate(training_data), mininterval=2,
                                 desc="  Training =>", total=len(training_data), disable=not accelerator.is_main_process):
        niter = epoch * len(training_data) + batch_idx
        if writer is not None:
            writer.add_scalar("Train/LearningRate", float(optimizer.param_groups[0]["lr"]), niter)

        if opt.dset_name == "clevr":
            image_id, change_feat, change_ids, change_labels, change_mask, \
                no_change_feat, no_change_ids, no_change_labels, no_change_mask = batch

            change_feat = change_feat.to(device)
            change_ids = change_ids.to(device)
            change_labels = change_labels.to(device)
            change_mask: torch.Tensor = change_mask.to(device)
            no_change_feat = no_change_feat.to(device)
            no_change_ids = no_change_ids.to(device)
            no_change_labels = no_change_labels.to(device)
            no_change_mask: torch.Tensor = no_change_mask.to(device)

            change_video_mask = change_mask[:, :config.max_v_len]
            no_change_video_mask = no_change_mask[:, :config.max_v_len]

            # change_text_mask = change_mask[:, config.max_v_len:]
            # no_change_text_mask = no_change_mask[:, config.max_v_len:]
            if getattr(opt, "training_stage", None) and opt.training_stage == 1 and opt.mask_prob > 0:
                change_text_mask = change_mask[:, config.max_v_len:]
                no_change_text_mask = no_change_mask[:, config.max_v_len:]
                change_rnd_mask = torch.rand_like(change_text_mask)
                no_change_rnd_mask = torch.rand_like(no_change_text_mask)
                change_rnd_mask = change_rnd_mask < opt.mask_prob
                no_change_rnd_mask = no_change_rnd_mask < opt.mask_prob
                change_text_mask = change_text_mask.bool() & change_rnd_mask
                no_change_text_mask = no_change_text_mask.bool() & no_change_rnd_mask
            else:
                change_text_mask = None
                no_change_text_mask = None

            # forward & backward
            optimizer.zero_grad()
            change_loss, change_pred_scores = model(change_feat, change_video_mask, change_ids,
                                                    change_text_mask, change_labels)
            no_change_loss, no_change_pred_scores = model(no_change_feat, no_change_video_mask, no_change_ids,
                                                            no_change_text_mask, no_change_labels, mtm=False)

            loss = change_loss + no_change_loss

            # make it consistent with other configs
            pred_scores_list = [change_pred_scores]
            input_labels_list = [change_labels]

            no_pred_scores_list = [no_change_pred_scores]
            no_input_labels_list = [no_change_labels]

        else:
            image_id, change_feat, change_ids, change_labels, change_mask = batch

            change_feat = change_feat.to(device)
            change_ids = change_ids.to(device)
            change_labels = change_labels.to(device)
            change_mask = change_mask.to(device)

            change_video_mask = change_mask[:, :config.max_v_len]

            change_text_mask = change_mask[:, config.max_v_len:]
            
            if getattr(opt, "training_stage", None) and opt.training_stage == 1 and opt.mask_prob > 0:
                change_text_mask = change_mask[:, config.max_v_len:]
                change_rnd_mask = torch.rand_like(change_text_mask)
                change_rnd_mask = change_rnd_mask < opt.mask_prob
                change_text_mask = change_text_mask.bool() & change_rnd_mask
            else:
                change_text_mask = None

            # forward & backward
            optimizer.zero_grad()
            change_loss, change_pred_scores = model(change_feat, change_video_mask, change_ids,
                                        change_text_mask, change_labels)

            loss = change_loss

            # make it consistent with other configs
            pred_scores_list = [change_pred_scores]
            input_labels_list = [change_labels]

        accelerator.wait_for_everyone()
        
        # loss.backward()
        accelerator.backward(loss)
        if opt.grad_clip != -1:  # enable, -1 == disable
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()

        # update model parameters with ema
        if ema is not None:
            ema(model, niter)

        # keep logs
        n_correct = 0
        n_word = 0

        if opt.dset_name == "clevr":
            for pred, gold in zip(pred_scores_list, input_labels_list):
                n_correct += cal_performance(pred, gold)
                valid_label_mask = gold.ne(VCDataset.IGNORE)
                n_word += valid_label_mask.sum().item()

            for pred, gold in zip(no_pred_scores_list, no_input_labels_list):
                n_correct += cal_performance(pred, gold)
                valid_label_mask = gold.ne(VCDataset.IGNORE)
                n_word += valid_label_mask.sum().item()

        else:
            for pred, gold in zip(pred_scores_list, input_labels_list):
                n_correct += cal_performance(pred, gold)
                valid_label_mask = gold.ne(VCDataset.IGNORE)
                n_word += valid_label_mask.sum().item()

        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()

        if opt.debug:
            break
    torch.autograd.set_detect_anomaly(False)

    loss_per_word = 1.0 * total_loss / n_word_total
    accuracy = 1.0 * n_word_correct / n_word_total
    return loss_per_word, accuracy


def eval(model, checkpoint, validation_data, opt):
    val_greedy_output, filepaths = eval_language_metrics(
    checkpoint, validation_data, opt, eval_mode="val", model=model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:

        logger.info("[Val] METEOR {m:.2f} Bleu@4 {b:.2f} ROUGE_L {r:.2f} CIDEr {c:.2f}"
                    .format(m=val_greedy_output["METEOR"]*100,
                            b=val_greedy_output["Bleu_4"]*100,
                            r=val_greedy_output["ROUGE_L"]*100,
                            c=val_greedy_output["CIDEr"]*100,
                            # r=val_greedy_output["re4"]*100))
                    ))
        # writer.add_scalar("Val/Re4", val_greedy_output["re4"]*100, niter)


def train(model: MMVID_IDC, training_data, validation_data, device, opt):

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    mmvid_params = list(filter(lambda x: x[0].startswith("intermediate_generator"), param_optimizer))
    other_params = list(filter(lambda x: not x[0].startswith("intermediate_generator"), param_optimizer))
    mmvid_no_decay = [p for n, p in mmvid_params if any(nd in n for nd in no_decay)]
    mmvid_decay = [p for n, p in mmvid_params if not any(nd in n for nd in no_decay)]
    other_no_decay = [p for n, p in other_params if any(nd in n for nd in no_decay)]
    other_decay = [p for n, p in other_params if not any(nd in n for nd in no_decay)]
    optimizer_grouped_parameters = [
        {"params": mmvid_decay, "weight_decay": opt.weight_decay, "lr": opt.lr_finetune},
        {"params": mmvid_no_decay, "weight_decay": 0.0, "lr": opt.lr_finetune},
        {"params": other_decay, "weight_decay": opt.weight_decay},
        {"params": other_no_decay, "weight_decay": 0.0},
    ]

    num_train_optimization_steps = len(training_data) * opt.n_epoch
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=opt.lr,
                         warmup=opt.lr_warmup_proportion,
                         t_total=num_train_optimization_steps,
                         schedule="warmup_linear")
    
    start_epoch = 0
    if opt.resume is not None:
        checkpoint = torch.load(opt.resume, map_location="cpu")
        # model.load_state_dict(checkpoint["model"])
        missing_keys = []
        for name, parameter in model.named_parameters():
            if name in checkpoint["model"]:
                parameter.data = checkpoint["model"][name].data
            else:
                missing_keys.append(name)
        if missing_keys:
            accelerator.print(f"Missing keys in checkpoint: {missing_keys}")
        else:
            accelerator.print("All model parameters loaded successfully from checkpoint.")
            
        if checkpoint.get("optimizer", None) is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        if start_epoch >= opt.n_epoch:
            logger.info("Starting epoch is larger than total epochs, exiting")
            return
        if accelerator.is_main_process:
            logger.info("Loaded checkpoint from {}, starting epoch: {}".format(opt.resume, start_epoch))
    
    elif getattr(opt, "pretrained_model", None) is not None:
        checkpoint = torch.load(opt.pretrained_model, map_location="cpu")["weights"]
        accelerator.print("Loading MMVID pretrained model from {}".format(opt.pretrained_model))
        missing_keys = []
        for name, param in model.intermediate_generator.named_parameters():
            requires_grad = param.requires_grad
            if name in checkpoint:
                param.data = checkpoint[name].data
                param.requires_grad = requires_grad
            else:
                missing_keys.append(name)
        if missing_keys:
            accelerator.print(f"Missing keys: {missing_keys}")
        else:
            accelerator.print("All keys loaded successfully")
    else:
        accelerator.print("[Warning] No pretrained model or resume checkpoint provided, starting from scratch.")
    
    model, training_data, validation_data, optimizer = accelerator.prepare(model, training_data, validation_data, optimizer)
    
    if opt.eval:
        assert checkpoint.get("model", None) is not None, "Checkpoint must contain 'model' key for evaluation. Current keys: {}".format(checkpoint.keys())
        eval(model, checkpoint, validation_data, opt)
        return
    
    if opt.ema_decay != -1:
        ema = EMA(opt.ema_decay)
        for name, p in model.named_parameters():
            # print(name, p.requires_grad)
            if p.requires_grad:
                ema.register(name, p.data)
    else:
        ema = None

    # for name, param in model.module.named_parameters():
    #     accelerator.print(f"{name}: {param.requires_grad}")
    # return

    if accelerator.is_main_process:
        writer = SummaryWriter(opt.res_dir)
    else:
        writer = None
    log_train_file = None
    log_valid_file = None

    if opt.log and accelerator.is_main_process:
        log_train_file = opt.log + ".train.log"
        log_valid_file = opt.log + ".valid.log"

        logger.info("Training performance will be written to file: {} and {}".format(
            log_train_file, log_valid_file))

        with open(log_train_file, "w") as log_tf, open(log_valid_file, "w") as log_vf:
            log_tf.write("epoch,loss,ppl,accuracy\n")
            # log_vf.write("epoch,loss,ppl,accuracy,METEOR,BLEU@4,CIDEr,re4\n")
            log_vf.write("epoch,METEOR,BLEU@4,CIDEr,ROUGE_L,re4\n")

    prev_best_score = 0.
    es_cnt = 0

    for epoch_i in range(start_epoch, opt.n_epoch):
        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            logger.info("[Epoch {}]".format(epoch_i))

        # schedule sampling prob update, TODO not implemented yet

        start = time.time()
        if ema is not None and epoch_i != 0:  # use normal parameters for training, not EMA model
            ema.resume(model)
        train_loss, train_acc = train_epoch(
            model, training_data, optimizer, ema, device, opt, writer, epoch_i)
        
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            logger.info("[Training]  ppl: {ppl: 8.5f}, accuracy: {acc:3.3f} %, elapse {elapse:3.3f} min"
                        .format(ppl=math.exp(min(train_loss, 100)), acc=100*train_acc, elapse=(time.time()-start)/60.))
            niter = (epoch_i + 1) * len(training_data)  # number of bart
            writer.add_scalar("Train/Acc", train_acc, niter)
            writer.add_scalar("Train/Loss", train_loss, niter)

        start = time.time()

        # Note here GT words are used to predicted next words, the same as training case!
        if ema is not None:
            ema.assign(model)  # EMA model

        # Note here we use greedy generated words to predicted next words, the true inference situation.
        checkpoint = {
            "model": accelerator.unwrap_model(model).state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict(),  # EMA model
            "model_cfg": model.module.config if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.config,
            "opt": opt,
            "epoch": epoch_i,
            "optimizer": optimizer.state_dict(),}
        
        if getattr(opt, "training_stage", None) and opt.training_stage == 1:
            if accelerator.is_main_process:
                model_name = opt.save_model + "_stg1_latest.chkpt"
                torch.save(checkpoint, model_name)
            
            continue

        val_greedy_output, filepaths = eval_language_metrics(
            checkpoint, validation_data, opt, eval_mode="val", model=model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model)
        
        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process:
            cider = val_greedy_output["CIDEr"]
            bleu4 = val_greedy_output["Bleu_4"]
            meteor = val_greedy_output["METEOR"]
            # r4 = val_greedy_output["re4"]
            r4 = 0
        
            logger.info("[Val] METEOR {m:.2f} Bleu@4 {b:.2f} CIDEr {c:.2f} re4 {r:.2f}"
                        .format(m=val_greedy_output["METEOR"]*100,
                                b=val_greedy_output["Bleu_4"]*100,
                                c=val_greedy_output["CIDEr"]*100,
                                r=0
                                # r=val_greedy_output["re4"]*100))
                        ))
            writer.add_scalar("Val/METEOR", val_greedy_output["METEOR"]*100, niter)
            writer.add_scalar("Val/Bleu_4", val_greedy_output["Bleu_4"]*100, niter)
            writer.add_scalar("Val/CIDEr", val_greedy_output["CIDEr"]*100, niter)
            # writer.add_scalar("Val/Re4", val_greedy_output["re4"]*100, niter)
            
            last_name = opt.save_model + "_last.chkpt"
            torch.save(checkpoint, last_name)
            logger.info("Saved last checkpoint to {}".format(last_name))

            if opt.save_mode == "all":
                model_name = opt.save_model + "_e{e}_b{b}_m{m}_c{c}_r{r}.chkpt".format(
                    e=epoch_i, b=round(bleu4*100, 2), m=round(meteor*100, 2),
                    c=round(cider*100, 2), r=round(r4*100, 2))
                torch.save(checkpoint, model_name)
            elif opt.save_mode == "best":
                model_name = opt.save_model + ".chkpt"
                if val_greedy_output[getattr(opt, "metric_reference", "CIDEr")] > prev_best_score:
                    es_cnt = 0
                    prev_best_score = val_greedy_output[getattr(opt, "metric_reference", "CIDEr")]
                    torch.save(checkpoint, model_name)
                    new_filepaths = [e.replace("tmp", "best") for e in filepaths]
                    for src, tgt in zip(filepaths, new_filepaths):
                        os.renames(src, tgt)
                    logger.info("The checkpoint file has been updated.")
                else:
                    es_cnt += 1
                    if es_cnt > opt.max_es_cnt:  # early stop
                        logger.info("Early stop at {} with {} {}".format(epoch_i, getattr(opt, "metric_reference", "CIDEr"), prev_best_score))
                        break
            cfg_name = opt.save_model + ".cfg.json"
            save_parsed_args_to_json(opt, cfg_name)

            if log_train_file and log_valid_file:
                with open(log_train_file, "a") as log_tf, open(log_valid_file, "a") as log_vf:
                    log_tf.write("{epoch},{loss: 8.5f},{ppl: 8.5f},{acc:3.3f}\n".format(
                        epoch=epoch_i, loss=train_loss, ppl=math.exp(min(train_loss, 100)), acc=100*train_acc))
                    log_vf.write("{epoch},{m:.2f},{b:.2f},{c:.2f},{rouge:.2f},{r:.2f}\n".format(
                        epoch=epoch_i,
                        m=val_greedy_output["METEOR"]*100,
                        b=val_greedy_output["Bleu_4"]*100,
                        c=val_greedy_output["CIDEr"]*100,
                        rouge=val_greedy_output["ROUGE_L"]*100,
                        r=0
                        # r=val_greedy_output["re4"]*100))
                    ))

        if opt.debug:
            break
    
    if accelerator.is_main_process:
        writer.close()


def main(opt):
    # random seed
    random.seed(opt.seed)
    os.environ['PYTHONHASHSEED'] = str(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    # Load Data
    transform = Compose([Resize((opt.image_size, opt.image_size), antialias=True)])
    train_dataset = VCDataset(
        dset_name=opt.dset_name,
        data_dir=opt.data_dir, video_feature_dir=opt.video_feature_dir, 
        word2idx_path=opt.word2idx_path, max_t_len=opt.decoder_param.max_t_len,
        max_v_len=opt.decoder_param.max_v_len, mode="train",
        recurrent=opt.recurrent, untied=opt.untied or opt.mtrans, num_frames=opt.num_frames,
        filtered=opt.filtered, filter_file_path=opt.filter_file_path, max_k=opt.max_k, transform=transform)
    val_dataset = VCDataset(
        dset_name=opt.dset_name,
        data_dir=opt.data_dir, video_feature_dir=opt.video_feature_dir, 
        word2idx_path=opt.word2idx_path, max_t_len=opt.decoder_param.max_t_len,
        max_v_len=opt.decoder_param.max_v_len, mode="test",
        recurrent=opt.recurrent, untied=opt.untied or opt.mtrans, num_frames=opt.num_frames,
        filtered=opt.filtered, filter_file_path=opt.filter_file_path, max_k=opt.max_k, transform=transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.num_workers, pin_memory=opt.pin_memory)
    val_loader = DataLoader(val_dataset,
                            batch_size=opt.val_batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=opt.pin_memory)

    opt.decoder_param.vocab_size = len(train_dataset.word2idx)
    opt.decoder_param.mask_token_id = train_dataset.word2idx["[MASK]"]
    accelerator.print(json.dumps(vars(opt), indent=4, sort_keys=True))

    # device = torch.device("cuda" if opt.cuda else "cpu")
    device = accelerator.device
    model_config = opt
    model = MMVID_IDC(model_config, accelerator=accelerator)

    if model_config.dalle_param.use_lora:
        target_modules = [name for name, module in model.named_modules() 
                          if name.startswith("intermediate_generator.transformer.") and
                            "out_proj" not in name and
                            isinstance(module, (nn.Linear, nn.MultiheadAttention))
                          ]
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    count_parameters(model)
    if hasattr(model, "embeddings") and hasattr(model.embeddings, "word_embeddings"):
        count_parameters(model.embeddings.word_embeddings)
    
    train(model, train_loader, val_loader, device, opt)


if __name__ == "__main__":
    opt = parse_args()

    if opt.dset_name == "edit":
        # from src.rtransformer.edit_dataset import RecursiveCaptionDataset as RCDataset
        from src.rtransformer.edit_dataset import VideoCaptionDataset as VCDataset
    elif opt.dset_name == "spot":
        # from src.rtransformer.spot_dataset import RecursiveCaptionDataset as RCDataset
        from src.rtransformer.spot_dataset import VideoCaptionDataset as VCDataset
    elif opt.dset_name == "clevr":
        # from src.rtransformer.clevr_dataset import RecursiveCaptionDataset as RCDataset
        from src.rtransformer.clevr_dataset import VideoCaptionDataset as VCDataset
    else:
        raise NotImplementedError("Dataset {} not implemented.".format(opt.dset_name))

    main(opt)