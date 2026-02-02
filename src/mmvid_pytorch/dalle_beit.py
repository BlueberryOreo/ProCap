import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.transforms as T
from axial_positional_embedding import AxialPositionalEmbedding
from einops import rearrange

from src.mmvid_pytorch.modules import AxialPositionalEmbeddingList
# from utils.utils import DivideMax

class DivideMax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        maxes = x.amax(dim=self.dim, keepdim=True)
        return x / maxes

FAKE_POOL_SIZE = 64


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def always(val):
    def inner(*args, **kwargs):
        return val

    return inner


def is_empty(t):
    return t.nelement() == 0


def masked_mean(t, mask, dim=1):
    t = t.masked_fill(~mask[:, :, None], 0.)
    return t.sum(dim=1) / mask.sum(dim=1)[..., None]


def set_requires_grad(model, value):
    if model is not None:
        for param in model.parameters():
            param.requires_grad = value


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


# sampling helpers


def top_k(logits, thres=0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


# Augmentation helpers

from itertools import permutations

PERM_LIST = None


def randperm(n, ordered=False):
    global PERM_LIST
    # ordered: include ordered permutation?
    if ordered:
        return torch.randperm(n)
    else:
        if n < 6:
            if PERM_LIST is None:
                PERM_LIST = list(permutations(range(n)))[1:]
            return random.choice(PERM_LIST)
        perm_ord = torch.tensor(range(n))
        while True:
            perm = torch.randperm(n)
            if (perm != perm_ord).any():
                return perm


def swap(tensor, dim=0):
    if tensor.shape[dim] % 2 == 0:
        tensor_swapped = torch.cat(torch.chunk(tensor, 2, dim=dim)[::-1],
                                   dim=dim)
    else:
        idx_perm = randperm(tensor.shape[dim], False)
        if dim == 0:
            tensor_swapped = tensor[idx_perm, ...]
        elif dim == 1:
            tensor_swapped = tensor[:, idx_perm, ...]
        else:
            raise RuntimeError
    return tensor_swapped


def swap_one_frame_along_batch(tokens, t=1, shuffle=False):
    tokens_shuffled = tokens.detach().clone()
    b, n, c = tokens.shape
    tokens_shuffled = tokens_shuffled.reshape(b, t, n // t, c)
    idx = np.random.randint(0, t, b)
    if shuffle:
        perm_idx = randperm(t)
        frames_shuffled = tokens_shuffled[range(b), idx, ...][perm_idx, ...]
    else:
        frames_shuffled = swap(tokens_shuffled[range(b), idx, ...], 0)
    tokens_shuffled[range(b), idx, ...] = frames_shuffled
    tokens_shuffled = tokens_shuffled.reshape(b, n, c)
    return tokens_shuffled


def warp_video_with_color(video):
    # video (n, t, 3, h, w)
    out = []
    for n in range(video.shape[0]):
        x = video[n]  # x (c, h, w)
        c_shift = torch.rand(1) - 0.5
        c_shift = c_shift.to(x.device)
        m = torch.zeros_like(x)
        num = random.randint(0, 3)
        if num == 0:
            m.data += c_shift
        elif num == 1:
            m[:, 0].data += c_shift
        elif num == 2:
            m[:, 1].data += c_shift
        else:
            m[:, 2].data += c_shift
        out.append(torch.clamp(x + m, 0, 1))
    return torch.stack(out)


def warp_with_color(x):
    # x (c, h, w)
    c_shift = torch.rand(1) - 0.5
    c_shift = c_shift.to(x.device)
    m = torch.zeros_like(x)
    num = random.randint(0, 3)
    if num == 0:
        m.data += c_shift
    elif num == 1:
        m[0].data += c_shift
    elif num == 2:
        m[1].data += c_shift
    else:
        m[2].data += c_shift
    out = torch.clamp(x + m, 0, 1)
    return out.unsqueeze(0)  # out (1, 3, h, w)


def warp_with_affine(x, angle=180, trans=0.1, scale=0.05):
    angle = np.pi * angle / 180.

    pa = torch.FloatTensor(4)
    th = torch.FloatTensor(2, 3)

    pa[0].uniform_(-angle, angle)
    pa[1].uniform_(-trans, trans)
    pa[2].uniform_(-trans, trans)
    pa[3].uniform_(1. - scale, 1. + scale)

    th[0][0] = pa[3] * torch.cos(pa[0])
    th[0][1] = pa[3] * torch.sin(-pa[0])
    th[0][2] = pa[1]
    th[1][0] = pa[3] * torch.sin(pa[0])
    th[1][1] = pa[3] * torch.cos(pa[0])
    th[1][2] = pa[2]

    x = x.unsqueeze(0)
    th = th.unsqueeze(0)
    grid = F.affine_grid(th, x.size()).to(x.device)
    out = F.grid_sample(x, grid, padding_mode="reflection")
    return out  # out (1, 3, h, w)


def warp(x, vid_strategy_prob=[0.25, 0.25, 0.25, 0.25]):
    # x (b, t, c, h, w)
    b, t, c, h, w = x.shape
    out = []
    for i in range(b):
        strategy = np.random.choice(range(4), p=vid_strategy_prob)
        if strategy == 0:
            # swap frame from another seq
            i_ = np.random.choice(list(set(range(b)) - {i}))
            y = x[i].detach().clone()
            j1 = random.randint(0, t - 1)
            j2 = random.randint(0, t - 1)
            y[j1, ...] = x[i_, j2, ...]
            out.append(y)
        elif strategy == 1:
            # shuffle frames
            perm_idx = randperm(t)
            y = x[i, perm_idx, ...].detach().clone()
            out.append(y)
        elif strategy == 2:
            # color
            j1 = random.randint(0, t - 1)
            y = x[i].detach().clone()
            y[j1, ...] = warp_with_color(y[j1]).squeeze(0)
            out.append(y)
        elif strategy == 3:
            # affine
            j1 = random.randint(0, t - 1)
            y = x[i].detach().clone()
            y[j1, ...] = warp_with_affine(y[j1], 30, 0.1, 0.1).squeeze(0)
            out.append(y)
        else:
            raise NotImplementedError
    out = torch.stack(out, 0)
    return out


# discrete vae class


class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(chan, chan, 3,
                                           padding=1), nn.ReLU(),
                                 nn.Conv2d(chan, chan, 3, padding=1),
                                 nn.ReLU(), nn.Conv2d(chan, chan, 1))

    def forward(self, x):
        return self.net(x) + x


# main DALL-E class


class BERT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        vae,
        cvae=None,
        num_text_tokens=10000,
        text_seq_len=256,
        stable=False,
        text_feature_dim=0,
        fixed_language_model=None,
        which_transformer='none',
        num_visuals=1,
        num_targets=1,
        use_separate_visual_emb=False,
        insert_sep=False,
        text_emb_bottleneck=False,
        use_lora=False,
        rel=True,
        vid=True,
        **kwargs,
    ):
        super().__init__()
        """
        Special Tokens:
        [REL]  if text-video are relevant
        [VID]  if video is continuous (shuffle frames)
        [MASK] masking
        [SEP]  separation (reserved)
        """
        image_size = vae.image_size
        num_image_tokens = vae.num_tokens
        image_fmap_size = (vae.image_size // (2**vae.num_layers))
        image_seq_len = image_fmap_size**2
        self.dim = dim

        self.num_visuals = num_visuals
        self.num_targets = num_targets

        self.rel = rel
        self.vid = vid

        self.random_erasing = T.RandomErasing(p=1,
                                              scale=(0.2, 0.8),
                                              ratio=(0.5, 2),
                                              value=0)


        if fixed_language_model is None:
            # reserve unique padding tokens for each position (text seq len)
            if text_seq_len > 0:
                num_text_tokens = num_text_tokens + text_seq_len
                self.text_emb = nn.Embedding(num_text_tokens + 1, dim)
                self.text_pos_emb = nn.Embedding(text_seq_len, dim)
                self.text_feature_mapping = lambda x: x
            else:
                print("** Info: Text sequence length is 0, no text embedding. **")
        else:
            assert text_feature_dim > 0
            text_seq_len = 1  # NOTE: if use fixed language model, text_seq_len is 1
            num_text_tokens = 1
            self.text_emb, self.text_pos_emb = None, None
            if text_emb_bottleneck is not None:
                nf = int(text_emb_bottleneck)
                self.text_feature_mapping = nn.Sequential(
                    nn.LayerNorm(text_feature_dim),
                    nn.Linear(text_feature_dim, nf),
                    nn.LayerNorm(nf),
                    nn.Linear(nf, dim),
                    nn.LayerNorm(dim),
                )
            else:
                self.text_feature_mapping = nn.Linear(text_feature_dim, dim)
        
        self.text_token_lut = {
            "[MASK]": num_text_tokens
        }

        # TODO: for masking+separate visual
        self.image_emb = nn.Embedding(num_image_tokens + 2, dim)
        self.conv_to_embedding = nn.Sequential(
            nn.LayerNorm(vae.model.quantize.e_dim),
            nn.Linear(vae.model.quantize.e_dim, dim),
        )
        self.target_pos_emb = AxialPositionalEmbedding(
            dim, axial_shape=(num_targets, image_fmap_size, image_fmap_size))

        if cvae is not None:
            use_separate_visual_emb = True
        if num_visuals > 0:
            if use_separate_visual_emb:
                # TODO: for masking+separate visual
                self.visual_emb = nn.Embedding(num_image_tokens + 2, dim)
            else:
                self.visual_emb = None

            self.visual_pos_emb = AxialPositionalEmbeddingList(
                dim,
                num_visuals,
                axial_shape=(image_fmap_size, image_fmap_size))

        self.image_token_lut = {
            '[MASK]': num_image_tokens,
            '[SEP]': num_image_tokens + 1,
        }
        # self.image_mask = self.image_emb.weight[self.image_token_lut['[MASK]']]
        self.image_mask = nn.Parameter(torch.zeros(dim))
        self.attention_forward = nn.Sequential()
        print("** Info: Temporal Consistency Loss is not used in this model. **")

        # for offsetting logits index and calculating cross entropy loss
        self.num_text_tokens = num_text_tokens
        self.num_image_tokens = num_image_tokens
        self.text_seq_len = text_seq_len
        self.image_seq_len = image_seq_len
        self.image_fmap_size = image_fmap_size
        self.image_size = image_size
        self.visual_seq_len = num_visuals * image_seq_len + (num_visuals *
                                                             insert_sep)
        self.target_seq_len = num_targets * image_seq_len
        self.insert_sep = insert_sep

        self.special_token_lut = {
            '[REL]': 0,
            # ---------- before and after control sequence ----------
            '[ST1]': 1,
            '[VID]': 2,
            '[ST3]': 3,
            '[ST4]': 4,
        }  # NOTE: [ST{1,3,4}] are reserved for future use
        self.num_special_tokens = len(self.special_token_lut)
        if self.rel:
            self.before_control_tok = [0]  # REL
        else:
            self.before_control_tok = []  # no REL
            print("** Info: REL token is not used in this model. **")
        
        if self.vid:
            self.after_control_tok = [1, 2]  # ST1, VID
        else:
            self.after_control_tok = [1]  # ST1 only
            print("** Info: VID token is not used in this model. **")

        self.before_control_seq_len = len(self.before_control_tok)
        self.after_control_seq_len = len(self.after_control_tok)
        self.special_emb = nn.Embedding(self.num_special_tokens, dim)
        self.special_pos_emb = nn.Embedding(self.num_special_tokens, dim)
        self.rel_tok_index = 0
        self.st1_tok_index = self.before_control_seq_len + self.text_seq_len + self.visual_seq_len
        self.vid_tok_index = self.before_control_seq_len + self.text_seq_len + self.visual_seq_len + 1
        self.txt_tok_index = self.before_control_seq_len

        seq_len = self.before_control_seq_len + \
            self.text_seq_len + \
            self.visual_seq_len + \
            self.after_control_seq_len + \
            self.target_seq_len
        control_seq_len = seq_len - self.target_seq_len
        self.total_seq_len = seq_len

        self.vae = vae
        self.cvae = cvae
        set_requires_grad(self.vae, False)  # freeze VAE from being trained
        set_requires_grad(self.cvae, False)  # freeze cVAE from being trained

        self.fixed_language_model = fixed_language_model
        self.which_transformer = which_transformer
        mask_prev_index = [self.st1_tok_index, self.vid_tok_index]
        assert which_transformer != 'default'
        if which_transformer.startswith('openai_clip'):
            from src.mmvid_pytorch.transformers.clip_model import OpenAICLIPTransformer
            self.transformer = OpenAICLIPTransformer(
                seq_len,
                which_transformer,
                model_path=kwargs['openai_clip_path'],
                causal=True,
                mask_type='mask_prev',
                mask_kwargs={'index': mask_prev_index},
                vision_layers=kwargs.get('vision_layers', 12)
            )
            print("Using OpenAI CLIP Transformer")
        else:  # NOTE: You can port the Transformer from dalle_pytorch if you want to train from scratch
            raise NotImplementedError

        self.stable = stable

        if stable:
            self.norm_by_max = DivideMax(dim=-1)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.num_image_tokens),
        )
        self.to_logits_rel = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
        )
        self.to_logits_vid = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
        )

        self.current_step = 0
        # erase visual
        self.visual_eraser = T.RandomErasing(p=0.95,
                                             scale=(0.55, 0.85),
                                             ratio=(0.5, 2),
                                             value=self.num_image_tokens)

    @torch.no_grad()
    @eval_decorator
    def generate_images(
        self,
        text,
        *,
        visual=None,
        mask=None,
        img=None,
        argmax=False,
        dynamic=True,
        debug=False,
        erase_visual=False,
        mask_predict_steps=10,
        preserve=None,
        origin_frames=None,
        t_overlap=1,
        pc_mode=None,
        vc_mode=None,
        face_mode=None,
        mp_config=None,
        long_mode='infill',
    ):
        vae = self.vae

        control_emb = self(
            text,
            visual=visual,
            erase_visual=erase_visual,
            erase_visual_half=
            True,  # NOTE: always erase half during generation if erase visual
            vc_mode=vc_mode,
            face_mode=face_mode,
            return_loss=False)
        img_seq, pnag_samples = self.mask_predict(
            control_emb,
            argmax=argmax,
            dynamic=dynamic,
            debug=debug,
            steps=mask_predict_steps,
            preserve=preserve,
            t_overlap=t_overlap,
            pc_mode=pc_mode,
            mp_config=mp_config,
            long_mode=long_mode,
            origin_frames=origin_frames,
        )
        img_seq = rearrange(img_seq,
                            'b (t n) -> (b t) n',
                            n=self.image_seq_len)
        images = vae.decode(img_seq)
        images = rearrange(images,
                           '(b t) c h w -> b t c h w',
                           t=self.num_targets)

        return images, pnag_samples, img_seq
    
    def predict_intermediate_frames(
        self,
        text,
        visual,
        preserve,
        origin_frames: torch.Tensor = None,
        masking=True,
        **kwargs,
    ):   
        if origin_frames is not None:
            B, T, C, H, W = origin_frames.shape
            
        control_emb = self(
            text,
            visual=visual,
            erase_visual=False,
            erase_visual_half=
            True,  # NOTE: always erase half during generation if erase visual
            vc_mode=None,
            face_mode=None,
            return_loss=False)

        control_seq_len, device = control_emb.shape[1], control_emb.device
        batch_size = text.shape[0]
        fully_masked_tok = self.image_token_lut['[MASK]'] + torch.zeros(
            batch_size, self.target_seq_len, device=device).long()

        preserve_mask1 = torch.zeros(batch_size,
                                     self.target_seq_len,
                                     device=device).long()
        preserve_ = self.image_token_lut['[MASK]'] + torch.zeros(
            control_emb.shape[0], self.target_seq_len, device=device).long()
        
        preserve_mask1 = rearrange(preserve_mask1,
                                    'b (t n) -> b t n',
                                    t=self.num_targets)
        preserve_mask1[:, 0, :] = 1
        preserve_mask1[:, -1, :] = 1
        preserve = rearrange(preserve,
                                'b (t n) -> b t n',
                                t=self.num_targets)
        preserve_ = rearrange(preserve_,
                                'b (t n) -> b t n',
                                t=self.num_targets)
        preserve_[:, 0, :] = preserve[:, 0, :]
        preserve_[:, -1, :] = preserve[:, -1, :]
        preserve_ = rearrange(preserve_, 'b t n -> b (t n)')
        preserve_mask1 = rearrange(preserve_mask1, 'b t n -> b (t n)')
        preserve = preserve_
        preserve_mask1 = preserve_mask1 == 1

        # fully_masked_emb = self.image_emb(fully_masked_tok)
        # target_pos_emb = self.target_pos_emb(fully_masked_emb)
        # mask_emb = self.image_emb.weight[self.image_token_lut['[MASK]']]

        # tok_in = torch.where(preserve_mask1, preserve, fully_masked_tok)
        emb_in = self.get_image_embedding(origin_frames, reshape=False) # b, t, n, d
        if masking:
            # emb_in = torch.where(preserve_mask1.unsqueeze(-1), emb_in, self.image_mask)
            mask_embeddings = self.image_mask.clone().repeat(
                batch_size, self.num_targets - 2, self.image_seq_len, 1) # b, target - 2, n, d
            # print(mask_embeddings.shape, emb_in.shape)
            emb_in = torch.cat([emb_in[:, 0, ...].unsqueeze(1), mask_embeddings, emb_in[:, -1, ...].unsqueeze(1)],
                               dim=1)  # b, target, n, d
            # print(emb_in.shape)
            emb_in = rearrange(emb_in, 'b t n d -> b (t n) d')  # b, target x n, d
        else:
            emb_in = rearrange(emb_in, 'b t n d -> b (t n) d')
        # emb_in[~preserve_mask1] = self.image_mask
        # print(emb_in.shape)
        target_pos_emb = self.target_pos_emb(emb_in)
        # emb_in = self.image_emb(tok_in)
        tokens = torch.cat((control_emb, emb_in + target_pos_emb), dim=1)
        out = self.transformer_forward(tokens)[:, control_seq_len:, :]
        return out

    def transformer_forward(self, tokens, attn_mask: torch.Tensor = None):
        # tokens are embeddings
        """
            Input tokens: [B, Tx(hxw) + control_seq_len, D] <==> [B, N, D]
        """
        out = self.transformer(tokens, attn_mask=attn_mask)
        if self.stable:
            out = self.norm_by_max(out)
        return out

    def decode_images(self, img_seq):
        img_seq = rearrange(img_seq,
                            'b (t n) -> (b t) n',
                            n=self.image_seq_len)
        images = self.vae.decode(img_seq)
        return images

    def decode_masks(self, mask):
        mask = rearrange(mask,
                         'b (t h w) -> (b t) 1 h w',
                         h=self.image_fmap_size,
                         w=self.image_fmap_size)
        patch_size = self.image_size // self.image_fmap_size
        mask_ = torch.repeat_interleave(
            torch.repeat_interleave(mask, patch_size, 2), patch_size, 3)
        mask = F.pad(mask_, (0, 0, 0, 0, 0, 2))  # red
        return mask

    @torch.no_grad()
    def mask_predict(
        self,
        control_emb,
        dynamic=True,
        debug=False,
        steps=10,
        preserve=None,
        t_overlap=1,
        mp_config=None,
        long_mode='infill',
        origin_frames=None,
        **kwargs,
    ):
        # print(preserve.shape)
        def sample_multinomial(logits, temperature=1.):
            logits = logits + temperature * sample_gumbel(logits)
            probs = F.softmax(logits, dim=2)
            tok = torch.multinomial(rearrange(probs, 'b n c -> (b n) c'), 1)
            tok = rearrange(tok, '(b n) 1 -> b n 1', b=probs.shape[0])
            Y = torch.gather(probs, 2, tok)
            Y, tok = Y.squeeze(2), tok.squeeze(2)
            return Y, tok

        def sample_gumbel(logit, eps=1e-20):
            U = torch.rand_like(logit)
            return -torch.log(-torch.log(U + eps) + eps)

        control_seq_len, device = control_emb.shape[1], control_emb.device
        batch_size = 1
        if long_mode == 'long':
            if preserve is None:
                t_overlap = 0
            N = self.target_seq_len - self.image_seq_len * t_overlap
        elif long_mode == 'interp' or long_mode == 'interp2' or long_mode == 'interp_real':
            N = self.target_seq_len // 2
        else:
            N = self.target_seq_len
        # fully_masked_tok = self.image_token_lut['[MASK]'] + torch.zeros(
        #     batch_size, self.target_seq_len, device=device).long()

        preserve_mask1 = torch.zeros(batch_size,
                                     self.target_seq_len,
                                     device=device).long()
        preserve_ = self.image_token_lut['[MASK]'] + torch.zeros(
            control_emb.shape[0], self.target_seq_len, device=device).long()
        if preserve is not None:
            if long_mode == 'long':
                preserve_mask1[:, :self.image_seq_len * t_overlap] = 1
                preserve = rearrange(preserve,
                                     '(b t) n -> b (t n)',
                                     t=self.num_targets)
                preserve_[:, :self.image_seq_len *
                          t_overlap] = preserve[:, -self.image_seq_len *
                                                t_overlap:]
            elif long_mode == 'interp' or long_mode == 'interp2' or long_mode == 'interp_real':
                preserve_mask1 = rearrange(preserve_mask1,
                                           'b (t n) -> b t n',
                                           t=self.num_targets)
                preserve_mask1[:, ::2, :] = 1
                preserve = rearrange(preserve,
                                     'b (t n) -> b t n',
                                     t=self.num_targets)
                preserve_ = rearrange(preserve_,
                                      'b (t n) -> b t n',
                                      t=self.num_targets)
                preserve_[:, ::2, :] = preserve[:, :self.num_targets // 2, :]
                preserve_ = rearrange(preserve_, 'b t n -> b (t n)')
                preserve_mask1 = rearrange(preserve_mask1, 'b t n -> b (t n)')
            elif long_mode == "infill":
                preserve_mask1 = rearrange(preserve_mask1,
                                           'b (t n) -> b t n',
                                           t=self.num_targets)
                preserve_mask1[:, 0, :] = 1
                preserve_mask1[:, -1, :] = 1
                preserve = rearrange(preserve,
                                     'b (t n) -> b t n',
                                     t=self.num_targets)
                preserve_ = rearrange(preserve_,
                                      'b (t n) -> b t n',
                                      t=self.num_targets)
                preserve_[:, 0, :] = preserve[:, 0, :]
                preserve_[:, -1, :] = preserve[:, -1, :]
                preserve_ = rearrange(preserve_, 'b t n -> b (t n)')
                preserve_mask1 = rearrange(preserve_mask1, 'b t n -> b (t n)')
        no_preserve = preserve is None
        preserve = preserve_
        preserve_mask1 = preserve_mask1 == 1

        # fully_masked_emb = self.image_emb(fully_masked_tok)
        # target_pos_emb = self.target_pos_emb(fully_masked_emb)
        # mask_emb = self.image_emb.weight[self.image_token_lut['[MASK]']]
        fully_mask_emb = self.image_mask.clone().repeat(batch_size, self.target_seq_len, 1)
        target_pos_emb = self.target_pos_emb(fully_mask_emb).to(device)
        mask_emb = self.image_mask

        # NOTE: steps can overwrite T in mp_config if positive
        Tmax = mp_config['T'] if steps <= 0 else steps
        B = mp_config['B']

        sample_toks = []
        T1_n = mp_config['T1_n']
        T2_n = mp_config['T2_n']
        T3_n = mp_config['T3_n']
        N1_n = mp_config['N1_n']
        N2_n = mp_config['N2_n']
        N3_n = max(1, int(N * mp_config['N3_n']))
        N4_n = max(1, int(N * mp_config['N4_n']))
        n = list(N * np.linspace(N1_n, N2_n, T1_n)) + list(
            N3_n * np.ones(T2_n)) + list(N4_n * np.ones(T3_n))

        T1_t = mp_config['T1_t']
        T2_t = mp_config['T2_t']
        T3_t = mp_config['T3_t']
        N1_t = mp_config['N1_t']
        N2_t = mp_config['N2_t']
        N3_t = mp_config['N3_t']
        N4_t = mp_config['N4_t']
        temp = list(np.linspace(N1_t, N2_t, T1_t)) + list(
            N3_t * np.ones(T2_t)) + list(N4_t * np.ones(T3_t))

        n = list(map(int, n))

        image_samples = []
        origin_frames_emb = self.get_image_embedding(origin_frames)

        preserve_mask1 = preserve_mask1.squeeze().repeat(origin_frames_emb.shape[0], 1)
        origin_frames_emb[~preserve_mask1] = mask_emb

        # tokens = torch.cat((control_emb, origin_frames_emb + target_pos_emb), dim=1)
        # out = self.transformer_forward(tokens)[:, control_seq_len:, :]
        # logits = self.to_logits(out)
        # Y, I_new = sample_multinomial(logits, temp[0])
        # I_tok = torch.where(preserve_mask1, preserve, I_new)
        # images = self.decode_images(I_new)
        # images = rearrange(images, "(b t) c h w -> b t c h w", b=origin_frames_emb.shape[0])
        # image_samples = [img for img in images]
        # sample_toks = [tok for tok in I_tok]

        for i in range(control_emb.shape[0]):
            control_emb_ = control_emb[i:i + 1, ...]

            # tok_in = fully_masked_tok
            # if not no_preserve:
            #     tok_in[0, ...] = torch.where(preserve_mask1, preserve[i, ...],
            #                                  fully_masked_tok[0, ...])

            # emb_in = self.image_emb(tok_in)
            if not no_preserve:
                emb_in = origin_frames_emb[i: i + 1, ...]
            else:
                emb_in = fully_mask_emb
            # target_pos_emb = self.target_pos_emb(emb_in).to(device)
            # emb_in + target_pos_emb

            tokens = torch.cat((control_emb_, emb_in + target_pos_emb), dim=1)
            out = self.transformer_forward(tokens)[:, control_seq_len:, :]
            logits = self.to_logits(out)  # b n c
            Y, I_new = sample_multinomial(logits, temp[0])

            I_tok = torch.where(preserve_mask1, preserve[i:i + 1, ...], I_new)

            # if debug:
            #     print('PNAG:')
            image_samples.append(self.decode_images(I_tok))

            sample_toks.append(I_tok)

        sample_toks = torch.cat(sample_toks, 0)
        return sample_toks, image_samples

    def get_image_tokens(self,
                         image,
                         reshape=True,
                         insert_sep=False,
                         which_vae='vae'):
        if which_vae == 'cvae' and self.cvae is not None:
            vae = self.cvae
        else:
            vae = self.vae
        if isinstance(image, list):
            assert len(image[0].shape
                       ) == 4, 'image should be list of 4d image tensors'
            image = torch.stack(image, dim=1)
        if len(image.shape) == 4:
            image = image.unsqueeze(1)
        is_raw_image = len(image.shape) == 5  # [B, T, C, H, W]
        if is_raw_image:
            b, t, c, h, w = image.shape
            image_size = vae.image_size
            assert (c, h, w) == (
                3, image_size, image_size
            ), f'invalid image of dimensions {image.shape} passed in during training'
            image = rearrange(image, 'b t c h w -> (b t) c h w')
            image = vae.get_codebook_indices(image)  # ((b t) n)
            if reshape:
                if insert_sep:
                    image = rearrange(image, '(b t) n -> b t n', t=t)
                    image = torch.cat(
                        (image, torch.empty(
                            b, t, 1, device=image.device).long().fill_(
                                self.image_token_lut['[SEP]'])),
                        dim=2)
                    image = rearrange(image, 'b t n -> b (t n)')
                else:
                    image = rearrange(image, '(b t) n -> b (t n)', t=t)
        return image
    
    def get_image_embedding(self, image, reshape=True, which_vae='vae', **kwargs):
        if which_vae == 'cvae' and self.cvae is not None:
            vae = self.cvae
        else:
            vae = self.vae
        if isinstance(image, list):
            assert len(image[0].shape
                       ) == 4, 'image should be list of 4d image tensors'
            image = torch.stack(image, dim=1)
        if len(image.shape) == 4:
            image = image.unsqueeze(1)
        is_raw_image = len(image.shape) == 5  # [B, T, C, H, W]
        if is_raw_image:
            b, t, c, h, w = image.shape
            image_size = vae.image_size
            assert (c, h, w) == (
                3, image_size, image_size
            ), f'invalid image of dimensions {image.shape} passed in during training'
            image = rearrange(image, 'b t c h w -> (b t) c h w')
            image = vae.model.encoder(image)
            image = rearrange(image, '(b t) d h w -> b t (h w) d', t=t)
            image = self.conv_to_embedding(image) # 256 -> dim
            if reshape:
                image = rearrange(image, 'b t n d -> b (t n) d')
        return image

    @torch.no_grad()
    def recon_images(self, images, which_vae='vae'):
        if which_vae == 'cvae' and self.cvae is not None:
            vae = self.cvae
        else:
            vae = self.vae
        img_seq = self.get_image_tokens(images,
                                        reshape=False,
                                        which_vae=which_vae)
        images = vae.decode(img_seq)
        return images

    @torch.no_grad()
    def get_codebook_emb(self, images, which_vae='vae'):
        b, t, c, h, w = images.shape
        if which_vae == 'cvae' and self.cvae is not None:
            vae = self.cvae
        else:
            vae = self.vae
        img_seq = self.get_image_tokens(images,
                                        reshape=False,
                                        which_vae=which_vae)
        img_code = rearrange(img_seq, '(b t) n -> b t n', t=t)
        img_embd = self.image_emb(img_code)
        return img_code, img_embd

    def random_erase_codebook(self, image, eraser, erase_half=False):
        image = rearrange(image,
                          'b (t h w) -> b t h w',
                          h=self.image_fmap_size,
                          w=self.image_fmap_size)
        if erase_half:
            image_ = image
            image_[:, :, self.image_fmap_size //
                   2:, :] = self.image_token_lut['[MASK]']
        else:
            image_ = torch.stack([eraser(c) for c in image], dim=0)
        image = rearrange(image_,
                          'b t h w -> b (t h w)',
                          h=self.image_fmap_size,
                          w=self.image_fmap_size)
        return image

    def erase_codebook_face(self, image, vc_mode, face_mode=None):
        image = rearrange(image,
                          'b (t h w) -> b t h w',
                          h=self.image_fmap_size,
                          w=self.image_fmap_size)
        if vc_mode == 'face_8x8':
            image_ = self.image_token_lut['[MASK]'] + torch.zeros_like(
                image).long()
            if face_mode is None:
                face_mode = 'eyes_nose' if random.random() < 0.5 else 'mouth'
            if face_mode == 'eyes_nose':  # eyes and nose
                image_[:, :, 2:5, 1:7] = image[:, :, 2:5, 1:7]
            else:  # mouth
                image_[:, :, 5:7, 2:6] = image[:, :, 5:7, 2:6]
            image = image_
        elif vc_mode == 'face2_8x8':  # appearance and motion
            image_ = self.image_token_lut['[MASK]'] + torch.zeros_like(
                image).long()
            image_[:, 0, ...] = image[:, 0, ...]
            image_[:, 1:, 2:6, 2:6] = image[:, 1:, 2:6, 2:6]
            image = image_
        elif vc_mode == 'face3_8x8':
            image_ = self.image_token_lut['[MASK]'] + torch.zeros_like(
                image).long()
            image_[:, 0, ...] = image[:, 0, ...]
            image_[:, :, 2:6, 2:6] = image[:, :, 2:6, 2:6]
            image = image_
        elif vc_mode == 'mask_8x8' or vc_mode == 'mask2_8x8':
            if face_mode is None:
                which_strategy = np.random.choice([1, 2, 3],
                                                  p=[0.5, 0.25, 0.25])
            else:
                which_strategy = 3
            if which_strategy == 1:
                image_ = image
            elif which_strategy == 2:
                image_ = self.image_token_lut['[MASK]'] + torch.zeros_like(
                    image).long()
                image_[:, :, 2:6, 2:6] = image[:, :, 2:6, 2:6]
            elif which_strategy == 3:
                image_ = self.image_token_lut['[MASK]'] + torch.zeros_like(
                    image).long()
                image_[:, :, 1:7, 1:7] = image[:, :, 1:7, 1:7]
            image = image_
        elif vc_mode == 'shape_4x4':
            image[:, :, 1:3, 1:3] = self.image_token_lut['[MASK]']
        else:
            raise NotImplementedError
        image = rearrange(image,
                          'b t h w -> b (t h w)',
                          h=self.image_fmap_size,
                          w=self.image_fmap_size)
        return image

    def get_special_token(self, tok_list, batch_size=1, device='cuda'):
        tok = torch.tensor(tok_list, dtype=torch.long, device=device)
        return tok.repeat(batch_size, 1)

    def swap_one_frame_along_batch(self, tokens, t=1):
        tokens_shuffled = tokens.detach().clone()
        b, n, c = tokens.shape
        tokens_shuffled = tokens_shuffled.reshape(b, t, n // t, -1)
        idx = np.random.randint(0, t, b)
        frames_shuffled = torch.cat(torch.chunk(tokens_shuffled[range(b), idx,
                                                                ...],
                                                2,
                                                dim=0)[::-1],
                                    dim=0)
        tokens_shuffled[range(b), idx, ...] = frames_shuffled
        tokens_shuffled = tokens_shuffled.reshape(b, n, c)
        return tokens_shuffled

    # ======================= forward ==========================
    def forward(
        self,
        text,
        visual=None,
        target=None,
        mask=None,
        return_loss=False,
        rel=False,
        vid=False,
        erase_visual=False,
        erase_visual_half=False,
        msm_strategy_prob=[0.7, 0.1, 0.1, 0.1],
        msm_bernoulli_prob=[0.2, 0.5],
        rel_no_fully_masked=False,
        vid_strategy_prob=[0.25, 0.25, 0.25, 0.25],
        negvc=False,
        visual_neg=None,
        text_neg=None,
        pc_prob=0,
        vc_mode=None,
        face_mode=None,
        visual_aug_mode=None,
        **kwargs,
    ):
        # visual and target are lists or 5d tensors (B, T, C, H, W)
        device = text[0].device
        if self.fixed_language_model is None:
            text_shape = text.shape
        else:  # NOTE: use embedding which takes a single token (from say RoBERTa)
            text_shape = [text.shape[0], 1]
        batch_size = text_shape[0]

        # NOTE: Prepend [REL]
        if rel:
            before_tok = self.get_special_token(self.before_control_tok,
                                                batch_size, device)
            before_emb = self.special_emb(before_tok)
            before_emb += self.special_pos_emb(before_tok)
            control_emb = before_emb
            control_seq_len = before_emb.shape[1]
            if negvc:
                control_neg_emb = before_emb
        else:
            control_seq_len = 0
            control_neg_emb = None

        # NOTE: make sure padding in text tokens get unique padding token id
        
        if self.text_seq_len > 0:
            if self.fixed_language_model is None:
                assert text.shape[
                    -1] == self.text_seq_len, f'the length {text.shape[-1]} of the text tokens you passed in does not have the correct length ({self.text_seq_len})'
                text_range = torch.arange(self.text_seq_len, device=device) + (
                    self.num_text_tokens - self.text_seq_len)
                text = torch.where(text == 0, text_range, text)
                # if return_loss:
                #     text_mask = torch.rand(text.shape, device=device) < 0.85 # 0.15 masking
                # else:
                #     text_mask = torch.ones(text.shape, device=device) == 1
                # text_masked = torch.where(text_mask, text, self.text_token_lut["[MASK]"])
                # text_emb = self.text_emb(text_masked)
                text_emb = self.text_emb(text)
                text_emb += self.text_pos_emb(
                    torch.arange(text_shape[1], device=device))
            else:
                text_emb = self.text_feature_mapping(text)
                text_emb = text_emb.unsqueeze(1)
        else:
            text_emb = torch.empty(batch_size, 0, self.dim,
                                   device=device)

        if rel:
            control_emb = torch.cat((control_emb, text_emb), dim=1)
        else:
            control_emb = text_emb
            
        control_seq_len += text_emb.shape[1]
        if negvc:
            # NOTE: current text_neg does not guarantee to be neg
            text_neg = torch.where(text_neg == 0, text_range, text_neg)
            text_neg_emb = self.text_emb(text_neg)
            text_neg_emb += self.text_pos_emb(
                torch.arange(text_shape[1], device=device))
            control_neg_emb = torch.cat((control_neg_emb, text_neg_emb), dim=1)

        visual_emb = None
        if self.num_visuals > 0:
            if exists(visual) and len(visual):
                if visual_aug_mode == 'motion_color' and random.random() < 0.9:
                    visual_ = visual.detach().clone()
                    visual_[:, 1:, ...] = warp_video_with_color(visual[:, 1:,
                                                                       ...])
                    visual = visual_
                visual = self.get_image_tokens(visual,
                                               insert_sep=self.insert_sep,
                                               which_vae='cvae')
                if erase_visual:
                    visual = self.random_erase_codebook(
                        visual, self.visual_eraser, erase_visual_half)
                if vc_mode is not None:
                    visual = self.erase_codebook_face(visual, vc_mode,
                                                      face_mode)
            else:
                visual = torch.empty(batch_size,
                                     self.visual_seq_len,
                                     device=device).long().fill_(
                                         self.image_token_lut['[MASK]'])
            visual_emb = self.visual_emb(
                visual) if self.visual_emb else self.image_emb(visual)

            visual_pos_emb = self.visual_pos_emb(visual_emb)
            visual_emb += visual_pos_emb
            control_emb = torch.cat((control_emb, visual_emb), dim=1)
            control_seq_len += visual.shape[1]

        # NOTE: Append [VID]
        after_tok = self.get_special_token(self.after_control_tok, batch_size,
                                           device)
        after_emb = self.special_emb(after_tok)
        after_emb += self.special_pos_emb(after_tok)
        control_emb = torch.cat((control_emb, after_emb), dim=1)
        control_seq_len += after_emb.shape[1]
        if negvc:
            control_neg_emb = torch.cat((control_neg_emb, after_emb), dim=1)

        if not return_loss:
            return control_emb

        target_orig = None
        target_tokens = None
        if exists(target) and len(target):
            # target_orig = target.detach().clone()
            # target = self.get_image_tokens(target)
            target_tokens = target.detach().clone()
            target_orig = target_tokens.clone()
            target_tokens = self.get_image_tokens(target_tokens)
            target = self.get_image_embedding(target)

        # NOTE: Masked Sequence Modeling
        #   Masking strategies:
        #     (1) randomly mask a number of tokens;
        #     (2) mask all tokens;
        #     (3) mask within boxed areas;
        #     (4) mask outside boxed areas;

        mask1_ = []
        not_fully_masked = torch.ones(batch_size, device=device)
        for i in range(batch_size):
            which_strategy = np.random.choice([1, 2, 3, 4],
                                              p=msm_strategy_prob)
            if which_strategy == 1:
                p = np.random.uniform(
                    *msm_bernoulli_prob)  # prob of keep GT tok
                mask1 = torch.bernoulli(
                    torch.ones(self.target_seq_len, device=device) *
                    p)  # keep GT if mask is 1
            elif which_strategy == 2:
                not_fully_masked[i] = 0
                mask1 = torch.zeros(self.target_seq_len, device=device)
            elif which_strategy == 3:
                mask1 = self.random_erasing(
                    torch.ones(self.num_targets,
                               1,
                               self.image_fmap_size,
                               self.image_fmap_size,
                               device=device)).reshape(-1)
            elif which_strategy == 4:
                mask1 = 1 - self.random_erasing(
                    torch.ones(self.num_targets,
                               1,
                               self.image_fmap_size,
                               self.image_fmap_size,
                               device=device)).reshape(-1)
            else:
                raise NotImplementedError
            if pc_prob > 0 and random.random() < pc_prob:
                t_overlap = random.randint(1, self.num_targets // 2)
                for tt in random.sample(range(self.num_targets), t_overlap):
                    mask1[self.image_seq_len * tt:self.image_seq_len *
                          (tt + 1)] = 1
            mask1_.append(mask1)

        mask1 = torch.stack(mask1_, 0) == 1
        # print("mask1.shape =", mask1.shape, "target.shape =", target.shape)
        # target_masked = torch.where(mask1, target,
        #                             self.image_token_lut['[MASK]'])
        # target_emb_masked = self.image_emb(target_masked)
        # image_mask = self.image_mask.repeat(target.shape[0], target.shape[1], 1)
        # target_emb_masked = torch.where(mask1, target, image_mask)
        target_emb_masked = target.clone()
        image_mask = self.image_mask.clone()
        target_emb_masked[~mask1] = image_mask
        target_pos_emb = self.target_pos_emb(target_emb_masked)

        tokens_msm = torch.cat(
            (control_emb, target_emb_masked + target_pos_emb), dim=1)
        out = self.transformer_forward(tokens_msm)
        out_msm = out[:, control_seq_len:, :]  # b n d
        # out_txt = out[:, text_seq_range[0]: text_seq_range[1], :]
        logits_msm = self.to_logits(out_msm)
        # logits_mlm = self.to_logits_text(out_txt)
        loss_msm = F.cross_entropy(logits_msm[~mask1], target_tokens[~mask1])
        # loss_mlm = F.cross_entropy(logits_mlm[~text_mask], text[~text_mask])
        # loss_msm = loss_msm + loss_mlm
        
        # NOTE: Relevance Estimation Task
        if rel:
            assert text_shape[
                0] >= 2 and text_shape[0] % 2 == 0, f"Invalid text_shape[0]: {text_shape[0]}"  # for REL swapping
            if negvc:
                tokens_neg_rel = torch.cat(
                    (control_neg_emb, target_emb_masked + target_pos_emb),
                    dim=1)
                out_neg_rel = self.transformer_forward(tokens_neg_rel)
                logits_pos_rel = self.to_logits_rel(
                    out[:, self.rel_tok_index, :]).squeeze()
                logits_neg_rel = self.to_logits_rel(
                    out_neg_rel[:, self.rel_tok_index, :]).squeeze()
            else:
                control_emb_swap = swap(control_emb, 0)
                tokens_neg_rel = torch.cat(
                    (control_emb_swap, target_emb_masked + target_pos_emb),
                    dim=1)
                out_neg_rel = self.transformer_forward(tokens_neg_rel)
                logits_pos_rel = self.to_logits_rel(
                    out[:, self.rel_tok_index, :]).squeeze()
                logits_neg_rel = self.to_logits_rel(
                    out_neg_rel[:, self.rel_tok_index, :]).squeeze()
            weight_pos = 1
            if rel_no_fully_masked:
                loss_rel_pos = F.binary_cross_entropy_with_logits(
                    logits_pos_rel,
                    torch.ones(batch_size, device=device),
                    reduction='none') * weight_pos
                loss_rel_neg = F.binary_cross_entropy_with_logits(
                    logits_neg_rel,
                    torch.zeros(batch_size, device=device),
                    reduction='none')
                loss_rel = (loss_rel_pos * not_fully_masked +
                            loss_rel_neg * not_fully_masked).sum() / max(
                                1., not_fully_masked.sum())
            else:
                loss_rel = (F.binary_cross_entropy_with_logits(
                    logits_pos_rel, torch.ones(batch_size, device=device)) *
                            weight_pos + F.binary_cross_entropy_with_logits(
                                logits_neg_rel,
                                torch.zeros(batch_size, device=device)))
        else:
            loss_rel = torch.tensor(0.0, device=device)

        # NOTE: Continuity Estimation Task
        if vid and self.num_targets > 1:
            weight_pos = 1
            weight_neg = 1
            # get warped frames
            target_warp = warp(target_orig, vid_strategy_prob)
            # target_warp = self.get_image_tokens(target_warp)
            target_warp = self.get_image_embedding(target_warp)
            # target_warp_masked = torch.where(mask1, target_warp,
            #                                  self.image_token_lut['[MASK]'])
            target_emb_warp_masked = target_warp.clone()
            target_emb_warp_masked[~mask1] = image_mask
            # target_emb_warp_masked = self.image_emb(target_warp_masked)
            tokens_neg_vid = torch.cat(
                (control_emb, target_emb_warp_masked + target_pos_emb), dim=1)
            out_neg_vid = self.transformer_forward(tokens_neg_vid)
            out_pos = out
            logits_pos_vid = self.to_logits_vid(out_pos[:,
                                                        self.vid_tok_index, :])
            logits_neg_vid = self.to_logits_vid(
                out_neg_vid[:, self.vid_tok_index, :])
            if rel_no_fully_masked:
                loss_vid = (F.binary_cross_entropy_with_logits(
                    logits_pos_vid,
                    torch.ones(batch_size, 1, device=device),
                    reduction='none').sum() / max(1., not_fully_masked.sum()) *
                            weight_pos + F.binary_cross_entropy_with_logits(
                                logits_neg_vid,
                                torch.zeros(batch_size, 1, device=device),
                                reduction='none').sum() /
                            max(1., not_fully_masked.sum()) * weight_neg)
            else:
                loss_vid = (F.binary_cross_entropy_with_logits(
                    logits_pos_vid, torch.ones(batch_size, 1, device=device)) *
                            weight_pos + F.binary_cross_entropy_with_logits(
                                logits_neg_vid,
                                torch.zeros(batch_size, 1, device=device)) *
                            weight_neg)
        else:
            loss_vid = torch.tensor(0.0, device=device)

        return loss_msm, loss_rel, loss_vid


if __name__ == "__main__":
    from vae import VQGanVAE1024
    from argparse import Namespace
    args = Namespace(amp=False, ar=False, attr_mode='object', batch_size=2, beta_msm=7.0, beta_rel=0.5, beta_vid=0.5, bpe_path=None, 
                     clip_grad_norm=1.0, cvae_path=None, dalle_path=None, dataset='h5_text', dataset_cache=None, dataset_keys=None, 
                     debug=False, deterministic=True, dim=768, dist_backend='nccl', dist_url='tcp://localhost:10002', distributed=True, 
                     drop_sentence=False, dropout_vc=0.4, filter_file_path='/data/sjy/VFI4IDC_test/CLIP4IDC/output/clevr_similarity_scores.json', 
                     filtered=True, fixed_language_model=None, fp16=False, frame_num=4, frame_step=4, fullvc=True, gpu=0, gpu_ids=None, 
                     image_size=224, image_text_folder='data/clevr_for_mmvid/train', insert_sep=False, iters=200000, learning_rate=0.0001, 
                     limit_train_batches=1, log_every=200, log_root='logs', loss_img_weight=7, lr_decay=True, lr_scheduler='warmuplr', 
                     lr_scheduler_every=1, lr_scheduler_step_size=10000, lr_scheduler_warmup=5000, mask_predict_steps=[10, 20, 30], 
                     mask_predict_steps1=20, max_k=5, mp_B=1, mp_N1n=0.9, mp_N1t=0.0, mp_N2n=0.1, mp_N2t=0.0, mp_N3n=0.125, mp_N3t=0.0, 
                     mp_N4n=0.0625, mp_N4t=0.0, mp_T=20, mp_T1n=10, mp_T1t=10, mp_T2n=10, mp_T2t=5, mp_T3n=30, mp_T3t=35, 
                     mp_config={'T1_n': 10, 'T2_n': 10, 'T3_n': 30, 'N1_n': 0.9, 'N2_n': 0.1, 'N3_n': 0.125, 'N4_n': 0.0625, 'T1_t': 10, 'T2_t': 5, 'T3_t': 35, 'N1_t': 0.0, 'N2_t': 0.0, 'N3_t': 0.0, 'N4_t': 0.0, 'T': 20, 'B': 1}, 
                     msm_bernoulli_prob=[0.2, 0.2], msm_strategy_prob=[0.7, 0.1, 0.1, 0.1], multiprocessing_distributed=True, n_per_sample=4, n_sample=4, 
                     name='train_image_text', negvc=False, no_lr_decay=False, num_targets=4, num_visuals=1, num_workers=16, openai_clip_model_path='ViT-B-32.pt', 
                     optimizer='adam', pc_prob=0, pnag_argmax=False, pnag_dynamic=False, rand_visual=False, rank=0, rel_no_fully_masked=True, resize_ratio=1, 
                     sample_every=5000, save_every_n_steps=5000, seed=42, slow=False, st_transformer_path=None, start_iter=None, text_emb_bottleneck=None, 
                     text_seq_len=24, transformer_path=None, truncate_captions=True, use_cvae=False, use_html=True, use_separate_visual_emb=False, 
                     vae_path='vqgan_path/logs/2025-03-19_imagenet_f16_1024_finetune_size224/checkpoints/epoch=000035.ckpt', vc_mode=None, 
                     vid_strategy_prob=[0.25, 0.25, 0.25, 0.25], video_only=False, vision_layers=12, visual=True, visual_aug_mode=None, weight_decay=0, 
                     which_tokenizer='simple', which_transformer='openai_clip_visual', which_vae='vqgan1024', workers=16, world_batch_size=4, world_size=1)
    
    dalle_params = {'num_text_tokens': 49408, 'text_seq_len': 24, 'dim': 768, 'loss_img_weight': 7, 'text_feature_dim': 0, 
                    'fixed_language_model': None, 'text_emb_bottleneck': None, 'which_transformer': 'openai_clip_visual', 
                    'num_targets': 4, 'num_visuals': 1, 'use_separate_visual_emb': False, 'insert_sep': False, 'openai_clip_path': '/data/sjy/VFI4IDC_test/MMVID/ViT-B-32.pt', 
                    'vision_layers': 12}
    vae_path = "/data/sjy/VFI4IDC_test/MMVID/vqgan_path/logs/2025-03-19_imagenet_f16_1024_finetune_size224/checkpoints/epoch=000035.ckpt"
    image_size = 224
    vae = VQGanVAE1024(vae_path, image_size, config_path="./data/vqgan.1024.224.config.yml")
    model = BERT(vae=vae, cvae=None, **dalle_params).to("cuda")
    print(model)

    text = torch.randint(0, 49408, (2, 24)).to("cuda")
    visuals = None
    target = torch.randn(2, 4, 3, 224, 224).to("cuda") # B T C H W
    loss_msm, loss_rel, loss_vid = model(
            text,
            visual=visuals if
            (args.visual and
             (args.fullvc or random.random() >= args.dropout_vc)) else None,
            target=target,
            erase_visual=args.rand_visual,
            return_loss=True,
            rel=args.beta_rel > 0,
            vid=args.beta_vid > 0,
            msm_strategy_prob=args.msm_strategy_prob,
            msm_bernoulli_prob=args.msm_bernoulli_prob,
            vid_strategy_prob=args.vid_strategy_prob,
            rel_no_fully_masked=args.rel_no_fully_masked,
            negvc=args.negvc,
            visual_neg=None,
            text_neg=None,
            pc_prob=args.pc_prob,
            vc_mode=args.vc_mode,
            face_mode=None,  # NOTE: face_mode is used in testing, for specifying the desired occlusion pattern
            visual_aug_mode=args.visual_aug_mode,
        )
    print(loss_msm, loss_rel, loss_vid)

    # tmp_img_tokens = torch.randint(0, 1024, (2, 784)).to("cuda")
    # sample_vc, tmp, _ = model.generate_images(
    #     text,
    #     visual=None,
    #     erase_visual=args.rand_visual,
    #     dynamic=args.pnag_dynamic,
    #     debug=args.debug,
    #     mask_predict_steps=1,
    #     vc_mode=args.vc_mode,
    #     face_mode=None,
    #     mp_config=args.mp_config,
    #     preserve=tmp_img_tokens,
    #     origin_frames=target,
    # )