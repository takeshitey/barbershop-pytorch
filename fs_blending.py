import argparse
import math
import os
import numpy as np
import pickle
from PIL import Image
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision import utils
from tqdm import tqdm

from networks import lpips
from networks.face_parse_bisenet import BiSeNet
from networks.style_gan_2 import Generator, EqualLinear, PixelNorm
from losses.style_loss import StyleLoss
from losses.appearance_loss import AppearanceLoss
from datasets.alignment_process import process_image_mask
from face_parse_bisenet_infer import vis_parsing_maps


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.empty_like(latent).uniform_(0, 1) * strength

    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )

def save_image(img, file_name, save_dir):
    to_img_path = os.path.join(save_dir, file_name)
    Image.fromarray(img).save(to_img_path)


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


def get_transform(resize=None):
    transform_list = []
    if resize is not None:
        transform_list.append(transforms.Resize(resize))
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    return transforms.Compose(transform_list)


# atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
#         10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
def custom_decode_labels(mask, region_index: int):
    region_mask = torch.where(mask == region_index, torch.ones_like(mask), torch.zeros_like(mask))
    return region_mask


class Mapper(nn.Module):
    def __init__(self, nc_in, nc_out, lr_mlp=0.01):
        super(Mapper, self).__init__()
        layers = [PixelNorm(),
                  EqualLinear(nc_in, nc_out, lr_mul=lr_mlp, activation='fused_lrelu')]

        for i in range(3):
            layers.append(EqualLinear(nc_out, nc_out, lr_mul=lr_mlp, activation='fused_lrelu'))

        self.mapping = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mapping(x)
        return x


class LatentBlender(nn.Module):
    def __init__(
        self,
        z_dim=512,
        lr_mlp=0.01,
    ):
        super().__init__()

        self.coarse_blender = Mapper(2 * z_dim, z_dim, lr_mlp=lr_mlp)
        self.fine_blender = Mapper(2 * z_dim, z_dim, lr_mlp=lr_mlp)

    def forward(self, s_a, s_b):
        coarse_cat = torch.cat([s_a[:, :4, :], s_b[:, :4, :]], dim=2)
        fine_cat = torch.cat([s_a[:, 4:, :], s_b[:, 4:, :]], dim=2)
        s_coarse = self.coarse_blender(coarse_cat)
        s_fine = self.fine_blender(fine_cat)
        out = torch.cat([s_coarse, s_fine], dim=1)
        return out



if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default='results')
    parser.add_argument("--stylegan_weights", type=str, required=True)
    parser.add_argument("--bisenet_weights", type=str, required=True)
    parser.add_argument("--psp_encoder_weights", type=str)
    parser.add_argument("--pretrained_weights", type=str)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr_rampdown", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--noise_ramp", type=float, default=0.75)
    parser.add_argument("--step", type=int, default=50000)
    parser.add_argument("--noise_regularize", type=float, default=1e3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ref_image_hair", type=str, default="data/images/00761.jpg")
    parser.add_argument("--ref_image_other", type=str, default="data/images/01012.jpg")

    args = parser.parse_args()
    exp_dir = args.exp_dir

    fs_dir = f'{exp_dir}/fs_inversion/'
    masks_dir = 'data/celeba_masks/'
    alignment_dir = f'{exp_dir}/seg_alignment/'

    save_dir = f'{exp_dir}/blending_result/'
    os.makedirs(save_dir, exist_ok=True)

    n_mean_latent = 10000

    ref_hair_img_id = os.path.basename(args.ref_image_hair).split('.')[0]
    ref_other_img_id = os.path.basename(args.ref_image_other).split('.')[0]

    ref_hair_fs_codes = torch.load(os.path.join(fs_dir, ref_hair_img_id + '.pt'))
    ref_other_fs_codes = torch.load(os.path.join(fs_dir, ref_other_img_id + '.pt'))
    ref_hair_fs_fmap32 = ref_hair_fs_codes['fmap32'].requires_grad_(False).to(device)
    ref_other_fs_fmap32 = ref_other_fs_codes['fmap32'].requires_grad_(False).to(device)

    ref_hair_fs_latent = ref_hair_fs_codes['latent'].requires_grad_(False).to(device)

    ref_other_fs_latent = ref_other_fs_codes['latent'].requires_grad_(False).to(device)
    ref_other_fs_noises = ref_other_fs_codes['noise']
    for noise in ref_other_fs_noises:
        noise.requires_grad = False

    ############ Aligned Code ##############
    ref_hair_aligned_codes = torch.load(os.path.join(alignment_dir, ref_hair_img_id + '_aligned.pt'))
    ref_hair_aligned_w_plus = ref_hair_aligned_codes['latent'].requires_grad_(False).to(device)
    ref_hair_aligned_noises = ref_hair_aligned_codes['noise']
    for noise in ref_hair_aligned_noises:
        noise.requires_grad = False

    ref_other_aligned_codes = torch.load(os.path.join(alignment_dir, ref_other_img_id + '_aligned.pt'))
    ref_other_aligned_w_plus = ref_other_aligned_codes['latent'].requires_grad_(False).to(device)
    ref_other_aligned_noises = ref_other_aligned_codes['noise']
    for noise in ref_other_aligned_noises:
        noise.requires_grad = False
    ############ Aligned Code ##############

    target_mask = Image.open(os.path.join(alignment_dir, 'target_mask.png'))
    target_label = np.array(target_mask).astype(np.int64)
    target_label = torch.from_numpy(target_label).unsqueeze(0).to(device)

    target_one_hot = torch.zeros((19, target_label.shape[1], target_label.shape[2])).to(device)
    target_one_hot = target_one_hot.scatter(0, target_label, 1).unsqueeze(0)
    target_one_hot = F.interpolate(target_one_hot, size=256, mode="nearest")


    ref_hair_data = process_image_mask(args.ref_image_hair,
                                       mask_path=os.path.join(masks_dir, os.path.basename(args.ref_image_hair).split('.')[0] + '.png'),
                                       seg_parts=[13],
                                       size=256
                                       )
    ref_other_data = process_image_mask(args.ref_image_other,
                                       mask_path=os.path.join(masks_dir, os.path.basename(args.ref_image_other).split('.')[0] + '.png'),
                                       seg_parts=list(range(13)) + list(range(14, 19)),
                                       size=256
                                       )

    ref_hair_image = ref_hair_data['img'].unsqueeze(0).to(device)
    ref_other_image = ref_other_data['img'].unsqueeze(0).to(device)

    # GENERATOR
    g_ema = Generator(args.size, 512, 8, channel_multiplier=2)
    g_ema.load_state_dict(torch.load(args.stylegan_weights)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)

    with torch.no_grad():
        g_output, _ = g_ema([ref_hair_aligned_w_plus], input_is_latent=True, noise=ref_hair_aligned_noises)
        ref_hair_aligned_fmap32 = g_output['g_feats'][32]

        g_output, _ = g_ema([ref_other_aligned_w_plus], input_is_latent=True, noise=ref_other_aligned_noises)
        ref_other_aligned_fmap32 = g_output['g_feats'][32]

    ## hair
    ref_hair_mask = ref_hair_data['parts_out'][13]['mask'].unsqueeze(0).to(device)
    ref_hair_mask_rz = F.interpolate(ref_hair_mask, size=32, mode="nearest")
    target_hair_mask = target_one_hot[:, 13:14, :, :]
    target_hair_mask_rz = F.interpolate(target_hair_mask, size=32, mode="nearest")

    hair_safe_mask_rz = ref_hair_mask_rz * target_hair_mask_rz
    f_align_hair = hair_safe_mask_rz * ref_hair_fs_fmap32 + (1 - hair_safe_mask_rz) * ref_hair_aligned_fmap32

    ## background
    ref_background_mask = ref_other_data['parts_out'][0]['mask'].unsqueeze(0).to(device)
    ref_background_mask_rz = F.interpolate(ref_background_mask, size=32, mode="nearest")
    target_background_mask = target_one_hot[:, 0:1, :, :]
    target_background_mask_rz = F.interpolate(target_background_mask, size=32, mode="nearest")

    background_safe_mask_rz = ref_background_mask_rz * target_background_mask_rz
    f_align_background = background_safe_mask_rz * ref_other_fs_fmap32 + (1 - background_safe_mask_rz) * ref_other_aligned_fmap32

    ## other
    ref_other_mask_rz = torch.zeros((1, 1, 32, 32), dtype=torch.uint8).to(device)
    target_other_mask = torch.zeros_like(target_background_mask).to(device)
    target_other_mask_rz = torch.zeros((1, 1, 32, 32)).to(device)
    for idx in list(range(1, 13)) + list(range(14, 19)):
        ref_part_mask = ref_other_data['parts_out'][idx]['mask'].unsqueeze(0).to(device)
        ref_part_mask_rz = F.interpolate(ref_part_mask, size=32, mode="nearest")
        target_part_mask = target_one_hot[:, idx:idx + 1, :, :]
        target_part_mask_rz = F.interpolate(target_part_mask, size=32, mode="nearest")

        ref_other_mask_rz = torch.where(ref_part_mask_rz > 0, ref_part_mask_rz, ref_other_mask_rz)
        target_other_mask_rz = torch.where(target_part_mask_rz > 0, target_part_mask_rz, target_other_mask_rz)
        target_other_mask = torch.where(target_part_mask > 0, target_part_mask, target_other_mask)

    other_safe_mask_rz = ref_other_mask_rz * target_other_mask_rz
    f_align_other = other_safe_mask_rz * ref_other_fs_fmap32 + (1 - other_safe_mask_rz) * ref_other_aligned_fmap32


    f_blend = target_hair_mask_rz * f_align_hair + target_background_mask_rz * f_align_background \
              + target_other_mask_rz * f_align_other

    injected_noise = ref_other_fs_noises


    ref_other_s = ref_other_fs_latent.detach().clone()

    g_feats_ref_other_aligned = {}
    with torch.no_grad():
        out = f_blend
        out = g_ema.convs[6](out, ref_other_s[:, 0], noise=injected_noise[7])
        out = g_ema.convs[7](out, ref_other_s[:, 1], noise=injected_noise[8])
        g_feats_ref_other_aligned[64] = out

        skip = g_ema.to_rgbs[3](out, ref_other_s[:, 2])

        _index = 2
        # 128x128 ---> higher res
        for up_conv, conv, noise1, noise2, to_rgb in zip(
                g_ema.convs[8::2], g_ema.convs[9::2], injected_noise[9::2], injected_noise[10::2], g_ema.to_rgbs[4:]
        ):
            res = 2**((_index + 12) // 2)
            out = up_conv(out, ref_other_s[:, _index], noise=noise1)
            out = conv(out, ref_other_s[:, _index + 1], noise=noise2)
            skip = to_rgb(out, ref_other_s[:, _index + 2], skip)

            g_feats_ref_other_aligned[res] = out

            _index += 2

        gen_ref_other_aligned = skip


    gen_ref_other_aligned = F.interpolate(gen_ref_other_aligned, size=256, mode="nearest")

    # LOSSES
    lpips_percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", vgg_blocks=["1", "2", "3", "4", "5"], use_gpu=device.startswith("cuda")
    )

    style_vgg_layers = [3, 8, 15, 22]
    style = StyleLoss(
        distance="l2", VGG16_ACTIVATIONS_LIST=style_vgg_layers, normalize=False
    )
    style.cuda()

    appearance_vgg_layers = [1]
    appearance = AppearanceLoss(
        distance="l2", VGG16_ACTIVATIONS_LIST=appearance_vgg_layers, normalize=False
    )
    appearance.cuda()

    # PARAMETERS
    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = g_ema.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5


    s_blend_init = torch.cat([ref_other_fs_latent[:, :2, :].detach().clone(),
                         torch.cat([ref_other_fs_latent[:, 2:, :256].detach().clone(),
                                    ref_hair_fs_latent[:, 2:, 256:].detach().clone()], dim=2)],
                        dim=1
                        )

    latent_blender = LatentBlender().to(device)

    optimizer = optim.Adam(latent_blender.parameters(), lr=args.lr)

    pbar = tqdm(range(args.step))

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr

        s_blend = latent_blender(ref_other_fs_latent.detach(), ref_hair_fs_latent.detach())

        g_feats = {}

        out = f_blend
        out = g_ema.convs[6](out, s_blend[:, 0], noise=injected_noise[7])
        out = g_ema.convs[7](out, s_blend[:, 1], noise=injected_noise[8])
        g_feats[64] = out

        skip = g_ema.to_rgbs[3](out, s_blend[:, 2])

        _index = 2
        # 128x128 ---> higher res
        for up_conv, conv, noise1, noise2, to_rgb in zip(
                g_ema.convs[8::2], g_ema.convs[9::2], injected_noise[9::2], injected_noise[10::2], g_ema.to_rgbs[4:]
        ):
            res = 2**((_index + 12) // 2)
            out = up_conv(out, s_blend[:, _index], noise=noise1)
            out = conv(out, s_blend[:, _index + 1], noise=noise2)
            skip = to_rgb(out, s_blend[:, _index + 2], skip)

            g_feats[res] = out

            _index += 2

        gen_blend = skip

        if (i + 1) % 5000 == 0:
            gen_out = make_image(gen_blend).squeeze(0)
            img_name = save_dir + f"blending_res_{ref_hair_img_id}x{ref_other_img_id}_iter{i}.png"
            pil_img = Image.fromarray(gen_out)
            pil_img.save(img_name)

        batch, channel, height, width = gen_blend.shape

        if height > 256:
            factor = height // 256

            gen_blend = gen_blend.reshape(
                batch, channel, height // factor, factor, width // factor, factor
            )
            gen_blend = gen_blend.mean([3, 5])

        # masked lpips loss
        hair_rec_loss = lpips_percept(gen_blend, ref_hair_image, mask=target_hair_mask).mean()
        hair_pixel_loss = F.mse_loss(ref_hair_image * target_hair_mask, gen_blend * target_hair_mask) * 2e1
        background_rec_loss = lpips_percept(gen_blend, gen_ref_other_aligned, mask=target_background_mask).mean()
        otherpart_rec_loss = lpips_percept(gen_blend, gen_ref_other_aligned, mask=target_other_mask).mean()

        s_latent_norm = F.mse_loss(s_blend, s_blend_init.detach()) * 0.2

        fmap_loss = F.mse_loss(g_feats[64], g_feats_ref_other_aligned[64].detach())

        loss = hair_rec_loss + background_rec_loss + otherpart_rec_loss + fmap_loss + hair_pixel_loss + s_latent_norm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 5000 == 0:
            result_file = {}
            result_file['f_blend'] = f_blend
            result_file['s_blend'] = s_blend
            result_file['noise'] = injected_noise
            result_file['latent_blender'] = latent_blender.state_dict()
            torch.save(result_file, os.path.join(save_dir, f'blending_res_latents_iter{i}.pt'))

        pbar.set_description(
            (
                f"hair_rec_loss: {hair_rec_loss.item():.4f}; "
                f"background_rec_loss: {background_rec_loss.item():.4f}; "
                f"otherpart_rec_loss: {otherpart_rec_loss.item():.4f}; "
                f"fmap_loss: {fmap_loss.item():.4f}; "
                f"s_latent_norm: {s_latent_norm.item():.4f}; "
                f"hair_pixel_loss: {hair_pixel_loss.item():.4f}; "
            )
        )

    result_file = {}
    result_file['f_blend'] = f_blend
    result_file['s_blend'] = s_blend
    result_file['noise'] = injected_noise
    result_file['latent_blender'] = latent_blender.state_dict()
    torch.save(result_file, os.path.join(save_dir, 'blending_res_latents.pt'))

