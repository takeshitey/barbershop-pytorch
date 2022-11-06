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
from tqdm import tqdm

from networks import lpips
from networks.style_gan_2 import Generator
from networks.encoders import psp_encoders
from losses.noise_loss import noise_regularize, noise_normalize_


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


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default='results')
    parser.add_argument("--stylegan_weights", type=str, required=True)
    parser.add_argument("--psp_encoder_weights", type=str)
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
    parser.add_argument("files", metavar="FILES", nargs="+")

    args = parser.parse_args()
    exp_dir = args.exp_dir

    save_dir = f'{exp_dir}/fs_inversion/'
    os.makedirs(save_dir, exist_ok=True)

    n_mean_latent = 10000

    img_names = []
    imgs = []
    encode_imgs = []
    for file_path in args.files:
        file_name = file_path.split('/')[-1]
        encoder_transform = get_transform(256)
        image = Image.open(file_path).convert("RGB")
        img = get_transform()(image)
        encode_img = encoder_transform(image)

        img_names.append(file_name)
        imgs.append(img)
        encode_imgs.append(encode_img)

    imgs = torch.stack(imgs, 0).to(device)
    encode_imgs = torch.stack(encode_imgs, 0).to(device)

    g_ema = Generator(args.size, 512, 8, channel_multiplier=2)
    g_ema.load_state_dict(torch.load(args.stylegan_weights)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)
    # trunc = g_ema.mean_latent(4096)

    psp_ckpt = torch.load(args.psp_encoder_weights, map_location='cpu')
    psp_opts = argparse.Namespace(**psp_ckpt['opts'])
    encoder = psp_encoders.Encoder4Editing(50, 'ir_se', psp_opts)
    encoder.load_state_dict(get_keys(psp_ckpt, 'encoder'), strict=True)
    encoder.eval()
    encoder = encoder.to(device)
    latent_avg = psp_ckpt['latent_avg'].to(device)

    with torch.no_grad():
        w_plus_codes = encoder(encode_imgs)
        if encoder.opts.start_from_latent_avg:
            if w_plus_codes.ndim == 2:
                w_plus_codes = w_plus_codes + latent_avg.repeat(w_plus_codes.shape[0], 1, 1)[:, 0, :]
            else:
                w_plus_codes = w_plus_codes + latent_avg.repeat(w_plus_codes.shape[0], 1, 1)

        g_output, _ = g_ema([w_plus_codes], input_is_latent=True, randomize_noise=False)
        inversion_imgs = g_output['image']
        g_feats = g_output['g_feats']

    del encoder


    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = g_ema.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

    # LPIPS MODEL
    lpips_percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", vgg_blocks=["1", "2", "3", "4", "5"], use_gpu=device.startswith("cuda")
    )

    noises_single = g_ema.make_noise()
    injected_noise = []
    for noise in noises_single:
        injected_noise.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

    # copy over to W+
    latent_in = w_plus_codes[:, 7:, :].detach().clone()
    latent_in.requires_grad = True

    # copy from g_feat 32x32
    fmap32_in = g_feats[32].detach().clone()
    fmap32_in.requires_grad = True

    for noise in injected_noise:
        noise.requires_grad = True

    optimizer = optim.Adam([latent_in, fmap32_in] + injected_noise, lr=args.lr)

    pbar = tqdm(range(args.step))

    imgs.detach()

    for i in pbar:
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

        recon_loss = lpips_percept(img_gen, imgs).mean() * args.lambda_recon
        n_loss = noise_regularize(injected_noise) * args.noise_regularize
        fmap_loss = F.mse_loss(fmap32_in, g_feats[32]) * args.lambda_fmap

        loss = recon_loss + fmap_loss + n_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        noise_normalize_(injected_noise)

        if (i + 1) % 200 == 0:
            gen_out = make_image(img_gen)
            for n, filename in enumerate(img_names):
                img_name = save_dir + os.path.splitext(filename)[0] + "-project-iter{}.png".format(i)
                pil_img = Image.fromarray(gen_out[n])
                pil_img.save(img_name)


        pbar.set_description(
            (
                f"recon_loss: {recon_loss.item():.4f}; noise regularize: {n_loss.item():.4f};"
                f" fmap_loss: {fmap_loss.item():.4f}; lr: {lr:.4f}"
            )
        )


    for i, filename in enumerate(img_names):
        noise_single = []
        for noise in injected_noise:
            noise_single.append(noise[i:i + 1])

        result_file = {}
        result_file['latent'] = latent_in[i:i + 1]
        result_file['noise'] = noise_single
        result_file['fmap32'] = fmap32_in[i:i + 1]
        torch.save(result_file, os.path.join(save_dir, os.path.splitext(filename)[0] + '.pt'))

