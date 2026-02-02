# DEBUG
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from easydict import EasyDict as edict

from src.rtransformer.masked_transformer import MTransformerWithoutVisionEncoder as MTransformer
from src.utils import get_vae_model


class MMVID_IDC(nn.Module):
    text_token_length: int
    text_length: int
    target_frames: int
    
    intermediate_generator: nn.Module
    mlp: nn.Module
    
    def __init__(self, configs, **kwargs):
        super(MMVID_IDC, self).__init__()

        self.text_token_length = configs.dalle_param.bert.text_seq_len
        self.text_length = configs.decoder_param.max_t_len
        self.target_frames = configs.decoder_param.max_v_len
        # self.mp_config = configs.dalle_param.mp_config

        # Get model params
        dalle_params = configs.dalle_param.bert
        vae_params = configs.dalle_param.vae
        text_decoder_param = configs.decoder_param

        self.config = text_decoder_param

        # Get intermediate frame generator
        vae, vae_params = get_vae_model(
            vae_params.which_vae,
            vae_path=vae_params.vae_path,
            image_size=vae_params.image_size,
        )
        self.is_beit = dalle_params.beit
        from src.mmvid_pytorch.dalle_beit import BERT
        self.intermediate_generator = BERT(vae=vae, **dalle_params)

        self.end_to_end = getattr(configs, "end_to_end", False)
        self.use_fi_frames = getattr(configs, "use_fi_frames", False)
        self.use_reconstruct = getattr(configs, "use_reconstruct", False)
        self.num_frames = configs.num_frames
        
        if self.use_fi_frames:
            self.intermediate_generator.image_mask.requires_grad = False

        if self.end_to_end:
            print("End-to-end training mode. ")
        else:
            for name, param in self.intermediate_generator.named_parameters():
                if configs.dalle_param.freeze or self.use_reconstruct:
                    param.requires_grad = False
                else:
                    if name in configs.dalle_param.skip_params:
                        param.requires_grad = False
            # param.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Linear(configs.dalle_param.bert.dim, configs.dalle_param.bert.dim),
            nn.ReLU(),
            nn.Linear(configs.dalle_param.bert.dim, configs.decoder_param.hidden_size),
            nn.LayerNorm(configs.decoder_param.hidden_size),
        )

        # Get text decoder
        self.text_decoder = MTransformer(text_decoder_param)

    def get_intermediate_features(self, bef_img: torch.Tensor, aft_img: torch.Tensor, intermediate: torch.Tensor = None):
        """
            bef_img: torch.tensor, (B, C, H, W)
            aft_img: torch.tensor, (B, C, H, W)
            intermediate: torch.tensor, (B, T, C, H, W) or None ablation in using interpolated frames to train
        """
        B, C, H, W = bef_img.shape

        if self.end_to_end:
            original_frames = torch.cat([bef_img.unsqueeze(1), aft_img.unsqueeze(1)], dim=1)
            embed_in = self.intermediate_generator.get_image_embedding(original_frames)
            return self.intermediate_generator.transformer_forward(self.intermediate_generator.target_pos_emb(embed_in))

        tokenized_text = torch.zeros(B, self.text_token_length, device=bef_img.device).long() # b, t

        img_sequence = torch.cat([bef_img.unsqueeze(1), aft_img.unsqueeze(1)], dim=1)
        # >>> Not used in BEiT
        if not self.is_beit:
            img_tokens = self.intermediate_generator.get_image_tokens(img_sequence, False)
            img_tokens = rearrange(img_tokens, "(b t) n -> b t n", b=B)
            video_tokens = torch.zeros(B, self.target_frames, img_tokens.shape[-1], device=bef_img.device).long() \
                + self.intermediate_generator.image_token_lut["[MASK]"]
            video_tokens[:, 0, ...] = img_tokens[:, 0, ...]
            video_tokens[:, -1, ...] = img_tokens[:, -1, ...]
            video_tokens = rearrange(video_tokens, "b t n -> b (t n)")
            print("video_tokens.shape:", video_tokens.shape)
        # <<<
        else:
            N = (self.intermediate_generator.vae.image_size // (2**self.intermediate_generator.vae.num_layers)) ** 2
            video_tokens = torch.zeros((B, self.target_frames * N))

        if intermediate is not None:
            original_frames = torch.cat([bef_img.unsqueeze(1), intermediate, aft_img.unsqueeze(1)], dim=1)
        else:
            original_frames = torch.cat([bef_img.unsqueeze(1), aft_img.unsqueeze(1)], dim=1)

        intermediate_features = self.intermediate_generator.predict_intermediate_frames(
            text=tokenized_text,
            visual=bef_img,
            preserve=video_tokens,
            origin_frames=original_frames,
            masking=not self.use_fi_frames
        )
        
        if self.use_reconstruct:
            img_seq = rearrange(intermediate_features,
                            'b (t n) -> (b t) n',
                            n=self.intermediate_generator.image_seq_len)
            images = self.intermediate_generator.vae.decode(img_seq)
            images = rearrange(images,
                            '(b t) c h w -> b t c h w',
                            t=self.intermediate_generator.num_targets)
            intermediate_features = self.intermediate_generator.get_image_embedding(images)
        
        return intermediate_features # B, (TxN), D

    def encode(self, video_clip, video_mask, return_dict=False):
        """
            video_clip: torch.tensor, (B, C, T, H, W), C=3 \n
        """
        assert len(video_clip.shape) == 5, "video_clip should be 5D tensor"
        assert video_clip.shape[1] == 3, "video_clip should have 3 channels"
        video_clip = rearrange(video_clip, 'b c t h w -> b t c h w')
        bef_img, aft_img = video_clip[:, 0, ...], video_clip[:, -1, ...]
        if self.use_fi_frames:
            selected_frames = [i / (self.target_frames - 1) * (self.num_frames - 1) for i in range(1, self.target_frames - 1)]
            selected_frames = torch.round(torch.tensor(selected_frames)).int().to(video_clip.device)
            original_frames = video_clip[:, selected_frames, ...]
            intermediate_features = self.get_intermediate_features(bef_img, aft_img, original_frames)
        else:
            intermediate_features = self.get_intermediate_features(bef_img, aft_img)
        intermediate_features = self.mlp(intermediate_features)
        return intermediate_features  # encoder_outputs

    def decode(self, text_input_ids, text_masks, text_input_labels, encoder_outputs, video_mask=None):
        caption_loss, prediction_scores, text_cls = self.text_decoder(encoder_outputs, text_input_ids, text_masks, text_input_labels)
        return caption_loss, prediction_scores, text_cls

    def forward(self, video_clip, video_mask, text_input_ids, text_masks, text_input_labels, return_dict=False, mtm=True):
        """
            video_clip: torch.tensor, (B, C, T, H, W), C=3 \n
            text_input_ids: (N, Lt) \n
            text_masks: (N, Lt)  boolean tensor, True means the token to be replaced with [MASK]. Or None \n
            text_input_labels: (N, Lt)  with `-1` on ignored positions  (in some sense duplicate with text_masks)
        """
        # Get intermediate features
        intermediate_features = self.encode(video_clip, video_mask, return_dict=return_dict)

        # Generate caption
        caption_loss, prediction_scores, text_cls = self.decode(text_input_ids, text_masks, text_input_labels, intermediate_features, video_mask)

        return caption_loss, prediction_scores
