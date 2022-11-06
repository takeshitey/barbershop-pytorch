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
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from networks import lpips
from networks.face_parse_bisenet import BiSeNet
from networks.style_gan_2 import Generator
from networks.encoders import psp_encoders
from losses.noise_loss import noise_regularize, noise_normalize_


celebamask_label_list = ['background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip',
                         'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

faceparsing_relabel_index = [0, 1, 10, 6, 4, 5, 2, 3, 7, 8, 11, 12, 13, 17, 18, 9, 15, 14, 16]


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

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


class LatentsDataset(Dataset):
    def __init__(self, opts, index):
        self.opts = opts
        self.latents_source = torch.load(opts.latents_source, map_location='cpu')
        self.latents_edited = torch.load(opts.latents_edited, map_location='cpu')

        self.source_img_paths = []
        img_path_file = open(opts.source_img_path_file, 'r')
        for line in img_path_file:
            self.source_img_paths.append(line.strip('\n'))

        self.edited_img_paths = sorted(make_dataset(opts.edited_img_path))

        minidata_size = len(self.source_img_paths) // 6
        self.latents_source = self.latents_source[index * minidata_size:(index + 1) * minidata_size]
        self.latents_edited = self.latents_edited[index * minidata_size:(index + 1) * minidata_size]
        self.source_img_paths = self.source_img_paths[index * minidata_size:(index + 1) * minidata_size]
        self.edited_img_paths = self.edited_img_paths[index * minidata_size:(index + 1) * minidata_size]

        self.transforms = get_transform()

    def __len__(self):
        return self.latents_source.shape[0]

    def __getitem__(self, index):
        source_img_path = self.source_img_paths[index]
        source_img_name = source_img_path.split('/')[-1]
        source_img = Image.open(source_img_path).convert("RGB")
        source_tensor = self.transforms(source_img)

        edited_img_path = self.edited_img_paths[index]
        edited_img = Image.open(edited_img_path).convert("RGB")
        edited_tensor = self.transforms(edited_img)

        return {'latents_source': self.latents_source[index],
                'latents_edited': self.latents_edited[index],
                'source_image': source_tensor,
                'source_img_name': source_img_name,
                'edited_image': edited_tensor,
                }


def run_bisenet(tensor):
    if tensor.shape[2] != 512:
        tensor = F.interpolate(tensor, size=512, mode="nearest")
    gen_seg = tensor.clamp_(min=-1, max=1).add(1).div_(2)
    bisenet_norm_mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=gen_seg.dtype, device=gen_seg.device).view(1, -1, 1, 1)
    bisenet_norm_std = torch.as_tensor([0.229, 0.224, 0.225], dtype=gen_seg.dtype, device=gen_seg.device).view(1, -1, 1, 1)
    gen_seg = gen_seg.sub_(bisenet_norm_mean).div_(bisenet_norm_std)

    seg_out = bisenet(gen_seg)[0]
    seg_out = seg_out[:, faceparsing_relabel_index]
    seg_pred = torch.max(seg_out, 1)[1].type(torch.uint8)
    return seg_pred

def custom_decode_labels(mask, region_index: int):
    region_mask = torch.where(mask == region_index, torch.ones_like(mask), torch.zeros_like(mask))
    return region_mask


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default='results')
    parser.add_argument("--stylegan_weights", type=str, required=True)
    parser.add_argument("--bisenet_weights", type=str, required=True)

    parser.add_argument("--latents_source", type=str, required=True)
    parser.add_argument("--latents_edited", type=str, required=True)
    parser.add_argument("--source_img_path_file", type=str, required=True)
    parser.add_argument("--edited_img_path", type=str, required=True)

    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr_rampdown", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--noise", type=float, default=0.005)
    parser.add_argument("--noise_ramp", type=float, default=0.75)
    parser.add_argument("--step", type=int, default=50000)
    parser.add_argument("--noise_regularize", type=float, default=1e3)
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--lambda_fmap", type=float, default=2.0)
    parser.add_argument("--device", type=str, default="cuda")


    args = parser.parse_args()
    exp_dir = args.exp_dir

    save_dir = f'{exp_dir}/fs_inversion/'
    os.makedirs(save_dir, exist_ok=True)

    n_mean_latent = 4096

    dataset = LatentsDataset(opts=args, index=5)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, drop_last=False)

    # BiSeNet MODEL
    bisenet = BiSeNet(n_classes=19)
    bisenet.load_state_dict(torch.load(args.bisenet_weights))
    bisenet.cuda()
    bisenet.eval()

    g_ema = Generator(args.size, 512, 8, channel_multiplier=2)
    g_ema.load_state_dict(torch.load(args.stylegan_weights)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)
    # trunc = g_ema.mean_latent(4096)

    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = g_ema.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

    # LPIPS MODEL
    lpips_percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", vgg_blocks=["1", "2", "3", "4", "5"], spatial=True, use_gpu=device.startswith("cuda")
    )

    for batch_in in tqdm(dataloader):
        latent_source = batch_in['latents_source'].cuda()
        latent_edited = batch_in['latents_edited'].cuda()
        source_images = batch_in['source_image'].cuda()
        source_img_names = batch_in['source_img_name']
        edited_images = batch_in['edited_image'].cuda()

        ### Source and Edited Seg Pred ###
        source_seg_pred = run_bisenet(source_images)
        edited_seg_pred = run_bisenet(edited_images)

        latent_in = latent_source[:, 7:, :].detach().clone()
        latent_in.requires_grad = True

        with torch.no_grad():
            g_output, _ = g_ema([latent_source], input_is_latent=True, randomize_noise=False)
            inversion_imgs = g_output['image']
            g_feats_source = g_output['g_feats']

            g_output, _ = g_ema([latent_edited], input_is_latent=True, randomize_noise=False)
            edited_imgs = g_output['image']
            g_feats_edited = g_output['g_feats']

        # copy from g_feat 32x32
        fmap32_in = g_feats_source[32].detach().clone()
        fmap32_in.requires_grad = True

        noises_single = g_ema.make_noise()
        injected_noise = []
        for noise in noises_single:
            injected_noise.append(noise.repeat(latent_source.shape[0], 1, 1, 1).normal_())

        for noise in injected_noise:
            noise.requires_grad = True

        optimizer = optim.Adam([latent_in, fmap32_in] + injected_noise, lr=args.lr)

        for i in range(args.step):
            t = i / args.step
            lr = get_lr(t, args.lr)
            optimizer.param_groups[0]["lr"] = lr
            noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
            latent_in = latent_noise(latent_in, noise_strength.item())

            # 64x64
            out = fmap32_in
            out = g_ema.convs[6](out, latent_in[:, 0], noise=injected_noise[7])
            out = g_ema.convs[7](out, latent_in[:, 1], noise=injected_noise[8])
            skip = g_ema.to_rgbs[3](out, latent_in[:, 2])

            _index = 2
            # 128x128 ---> higher res
            for up_conv, conv, noise1, noise2, to_rgb in zip(
                    g_ema.convs[8::2], g_ema.convs[9::2], injected_noise[9::2], injected_noise[10::2], g_ema.to_rgbs[4:]
            ):
                out = up_conv(out, latent_in[:, _index], noise=noise1)
                out = conv(out, latent_in[:, _index + 1], noise=noise2)
                skip = to_rgb(out, latent_in[:, _index + 2], skip)

                _index += 2

            img_gen = skip

            batch, channel, height, width = img_gen.shape

            recon_loss = lpips_percept(img_gen, source_images).mean() * args.lambda_recon
            n_loss = noise_regularize(injected_noise) * args.noise_regularize
            # mse_loss = F.mse_loss(img_gen, imgs) * args.lambda_pixel
            fmap_loss = F.mse_loss(fmap32_in, g_feats_source[32]) * args.lambda_fmap

            loss = recon_loss + fmap_loss + n_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            noise_normalize_(injected_noise)

            if (i + 1) % 10 == 0:
                gen_out = make_image(img_gen)
                for n, filename in enumerate(source_img_names):
                    img_name = save_dir + os.path.splitext(filename)[0] + "-project-iter{}.png".format(i)
                    pil_img = Image.fromarray(gen_out[n])
                    pil_img.save(img_name)

        ## other
        source_other_mask_rz = torch.zeros((1, 1, 32, 32), dtype=torch.uint8).to(device)
        edited_other_mask_rz = torch.zeros((1, 1, 32, 32), dtype=torch.uint8).to(device)
        for idx in list(range(0, 13)):
            source_part_mask = custom_decode_labels(source_seg_pred, idx).unsqueeze(1).to(device)
            source_part_mask_rz = F.interpolate(source_part_mask, size=32, mode="nearest")
            edited_part_mask = custom_decode_labels(edited_seg_pred, idx).unsqueeze(1).to(device)
            edited_part_mask_rz = F.interpolate(edited_part_mask, size=32, mode="nearest")

            source_other_mask_rz = torch.where(source_part_mask_rz > 0, source_part_mask_rz, source_other_mask_rz)
            edited_other_mask_rz = torch.where(edited_part_mask_rz > 0, edited_part_mask_rz, edited_other_mask_rz)

        other_safe_mask_rz = source_other_mask_rz * edited_other_mask_rz
        f_blend = other_safe_mask_rz * fmap32_in + (1 - other_safe_mask_rz) * g_feats_edited[32]

        source_hat_cloth_mask_rz = torch.zeros((1, 1, 32, 32), dtype=torch.uint8).to(device)
        for idx in list(range(14, 19)):
            source_part_mask = custom_decode_labels(source_seg_pred, idx).unsqueeze(1).to(device)
            source_part_mask_rz = F.interpolate(source_part_mask, size=32, mode="nearest")
            source_hat_cloth_mask_rz = torch.where(source_part_mask_rz > 0, source_part_mask_rz, source_hat_cloth_mask_rz)

        f_blend = source_hat_cloth_mask_rz * fmap32_in + (1 - source_hat_cloth_mask_rz) * f_blend
        s_blend = latent_in.detach().clone()

        with torch.no_grad():
            out = f_blend
            out = g_ema.convs[6](out, s_blend[:, 0], noise=injected_noise[7])
            out = g_ema.convs[7](out, s_blend[:, 1], noise=injected_noise[8])

            skip = g_ema.to_rgbs[3](out, s_blend[:, 2])

            _index = 2
            # 128x128 ---> higher res
            for up_conv, conv, noise1, noise2, to_rgb in zip(
                    g_ema.convs[8::2], g_ema.convs[9::2], injected_noise[9::2], injected_noise[10::2], g_ema.to_rgbs[4:]
            ):
                res = 2 ** ((_index + 12) // 2)
                out = up_conv(out, s_blend[:, _index], noise=noise1)
                out = conv(out, s_blend[:, _index + 1], noise=noise2)
                skip = to_rgb(out, s_blend[:, _index + 2], skip)

                _index += 2

            gen_blend = skip

        img_gen = F.interpolate(img_gen, size=512, mode="bicubic")
        gen_blend = F.interpolate(gen_blend, size=512, mode="bicubic")
        gen_out = make_image(img_gen)
        blend_out = make_image(gen_blend)
        concat_out = np.concatenate([gen_out, blend_out], axis=2)

        for n, filename in enumerate(source_img_names):
            img_name = exp_dir + os.path.splitext(filename)[0] + ".png"
            pil_img = Image.fromarray(concat_out[n])
            pil_img.save(img_name)
