import argparse
import math
import os
import cv2
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
from networks.style_gan_2 import Generator, Discriminator
from networks.encoders import psp_encoders

from losses.style_loss import StyleLoss
from losses.appearance_loss import AppearanceLoss
from losses.seg_loss import OhemCELoss, SoftmaxFocalLoss
from losses.noise_loss import noise_regularize, noise_normalize_
from datasets.alignment_process import process_image_mask
from face_parse_bisenet_infer import vis_parsing_maps


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

# Visualization utils
def get_colors():
    # currently support up to 19 classes (for the celebs-hq-mask dataset)
    colors = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255],
              [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204],
              [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
    return colors

def tensor2map(var):
    mask = np.argmax(var.data.cpu().numpy(), axis=0)
    colors = get_colors()
    mask_image = np.ones(shape=(mask.shape[0], mask.shape[1], 3))
    for class_idx in np.unique(mask):
        mask_image[mask == class_idx] = colors[class_idx]
    mask_image = mask_image.astype('uint8')
    return mask_image


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


# atts = [0 'background', 1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear',
# 9 'ear_r', 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']

celebamask_label_list = ['background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip',
                         'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

faceparsing_relabel_index = [0, 1, 10, 6, 4, 5, 2, 3, 7, 8, 11, 12, 13, 17, 18, 9, 15, 14, 16]

def custom_decode_labels(mask, region_index: int):
    region_mask = torch.where(mask == region_index, torch.ones_like(mask), torch.zeros_like(mask))
    return region_mask


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default='results')
    parser.add_argument("--stylegan_weights", type=str, required=True)
    parser.add_argument("--bisenet_weights", type=str, required=True)
    parser.add_argument("--psp_encoder_weights", type=str)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr_rampdown", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--noise_ramp", type=float, default=0.075)
    parser.add_argument("--step", type=int, default=50000)
    parser.add_argument("--noise_regularize", type=float, default=1e3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--refer_images", type=str, nargs='+', default="data/images/00761.jpg")

    args = parser.parse_args()
    exp_dir = args.exp_dir
    fs_dir = f'{exp_dir}/fs_inversion/'
    masks_dir = 'data/celeba_masks/'

    save_dir = f'{exp_dir}/seg_alignment/'
    os.makedirs(save_dir, exist_ok=True)

    n_mean_latent = 10000
    n_imgs = len(args.refer_images)

    img_names = []
    refer_imgs_data = []
    refer_images = []
    refer_latent_s = []
    for img_path in args.refer_images:
        img_name = img_path.split('/')[-1].split('.')[0]
        refer_data = process_image_mask(img_path,
                                        mask_path=os.path.join(masks_dir, img_name + '.png'),
                                        seg_parts=list(range(19)),
                                        size=256
                                        )
        ref_img = refer_data['img']

        ref_fs_codes = torch.load(os.path.join(fs_dir, img_name + '.pt'))
        ref_s_code = ref_fs_codes['latent'].requires_grad_(False)

        img_names.append(img_name)
        refer_imgs_data.append(refer_data)
        refer_images.append(ref_img)
        refer_latent_s.append(ref_s_code)

    refer_images = torch.stack(refer_images, 0).to(device)
    refer_latent_s = torch.cat(refer_latent_s, dim=0)

    ref_parts_masks = {}
    ref_parts_masked_imgs = {}
    for idx in range(19):
        part_mask = [refer_imgs_data[i]['parts_out'][idx]['mask'] for i in range(len(refer_imgs_data))]
        part_mask = torch.stack(part_mask, 0).to(device)
        part_masked_img = [refer_imgs_data[i]['parts_out'][idx]['masked_image'] for i in range(len(refer_imgs_data))]
        part_masked_img = torch.stack(part_masked_img, 0).to(device)
        ref_parts_masks[idx] = part_mask
        ref_parts_masked_imgs[idx] = part_masked_img

    target_mask = Image.open(os.path.join(save_dir, 'target_mask.png'))
    target_mask = np.array(target_mask).astype(np.int64)
    target_mask = torch.from_numpy(target_mask).unsqueeze(0).to(device)

    target_one_hot = torch.zeros((19, target_mask.shape[1], target_mask.shape[2])).to(device)
    target_one_hot = target_one_hot.scatter(0, target_mask, 1).unsqueeze(0)
    target_one_hot = F.interpolate(target_one_hot, size=256, mode="nearest")
    target_one_hot = target_one_hot.repeat(n_imgs, 1, 1, 1)

    target_label = target_mask.repeat(n_imgs, 1, 1)

    # Encoder Init W Plus Code
    psp_ckpt = torch.load(args.psp_encoder_weights, map_location='cpu')
    psp_opts = argparse.Namespace(**psp_ckpt['opts'])
    psp_opts.n_styles = int(math.log(1024, 2)) * 2 - 2

    encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', psp_opts)
    encoder.load_state_dict(get_keys(psp_ckpt, 'encoder'), strict=True)
    encoder.eval()
    encoder = encoder.to(device)
    latent_avg = psp_ckpt['latent_avg'].to(device)

    # BiSeNet MODEL
    bisenet = BiSeNet(n_classes=19)
    bisenet.load_state_dict(torch.load(args.bisenet_weights))
    bisenet.cuda()
    bisenet.eval()

    # GENERATOR
    g_ema = Generator(args.size, 512, 8, channel_multiplier=2)
    g_ema.load_state_dict(get_keys(psp_ckpt, 'decoder'), strict=True)
    g_ema.eval()
    g_ema = g_ema.to(device)

    with torch.no_grad():
        w_plus_encoded_target_mask = encoder(target_one_hot)
        if psp_opts.start_from_latent_avg:
            if w_plus_encoded_target_mask.ndim == 2:
                w_plus_encoded_target_mask = w_plus_encoded_target_mask + latent_avg.repeat(w_plus_encoded_target_mask.shape[0], 1, 1)[:, 0, :]
            else:
                w_plus_encoded_target_mask = w_plus_encoded_target_mask + latent_avg.repeat(w_plus_encoded_target_mask.shape[0], 1, 1)

        w_plus_align_init = torch.cat([w_plus_encoded_target_mask[:, :7, :], refer_latent_s], dim=1)

        g_output, _ = g_ema([w_plus_align_init], input_is_latent=True, randomize_noise=False)
        fmap_target = g_output['g_feats'][8].detach()

    del encoder


    # LOSSES
    lpips_percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", vgg_blocks=["1", "2", "3", "4", "5"], spatial=True, use_gpu=device.startswith("cuda")
    )

    style_vgg_layers = [3, 8]
    style = StyleLoss(
        distance="l2", VGG16_ACTIVATIONS_LIST=style_vgg_layers, normalize=False
    )
    style.cuda()

    appearance_vgg_layers = [3]
    appearance = AppearanceLoss(
        distance="l2", VGG16_ACTIVATIONS_LIST=appearance_vgg_layers, normalize=False
    )
    appearance.cuda()

    LossP = SoftmaxFocalLoss(gamma=2, ignore_lb=-100)
    Loss2 = SoftmaxFocalLoss(gamma=2, ignore_lb=-100)
    Loss3 = SoftmaxFocalLoss(gamma=2, ignore_lb=-100)

    # PARAMETERS
    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = g_ema.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

    # copy over to W+
    latent_coarse = w_plus_align_init[:, :4, :].detach().clone()
    latent_fine = w_plus_align_init[:, 4:, :].detach().clone()
    latent_fine.requires_grad = True

    noises_single = g_ema.make_noise()
    injected_noise = []
    for noise in noises_single:
        injected_noise.append(noise.repeat(w_plus_encoded_target_mask.shape[0], 1, 1, 1).normal_())

    for noise in injected_noise:
        noise.requires_grad = True

    optimizer = optim.Adam([latent_fine] + injected_noise, lr=args.lr)

    pbar = tqdm(range(args.step))

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr
        noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
        latent_fine = latent_noise(latent_fine, noise_strength.item())
        latent_in = torch.cat([latent_coarse, latent_fine], dim=1)

        g_output, _ = g_ema([latent_in], input_is_latent=True, noise=injected_noise)
        fake_img = g_output['image']
        fmap_gen = g_output['g_feats'][8]

        batch, channel, height, width = fake_img.shape

        gen_seg = F.interpolate(fake_img, size=512, mode="nearest")
        gen_seg = gen_seg.clamp_(min=-1, max=1).add(1).div_(2)
        bisenet_norm_mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=gen_seg.dtype, device=gen_seg.device).view(1, -1, 1, 1)
        bisenet_norm_std = torch.as_tensor([0.229, 0.224, 0.225], dtype=gen_seg.dtype, device=gen_seg.device).view(1, -1, 1, 1)
        gen_seg = gen_seg.sub_(bisenet_norm_mean).div_(bisenet_norm_std)

        seg_out, seg_out16, seg_out32 = bisenet(gen_seg)
        seg_out = seg_out[:, faceparsing_relabel_index]
        seg_out16 = seg_out16[:, faceparsing_relabel_index]
        seg_out32 = seg_out32[:, faceparsing_relabel_index]

        lossp = LossP(seg_out, target_label)
        loss16 = Loss2(seg_out16, target_label)
        loss32 = Loss3(seg_out32, target_label)
        seg_loss = (lossp * 2 + loss16 + loss32) * 1
        seg_pred = torch.max(seg_out, 1)[1].type(torch.uint8)

        if height > 256:
            factor = height // 256

            fake_img = fake_img.reshape(
                batch, channel, height // factor, factor, width // factor, factor
            )
            fake_img = fake_img.mean([3, 5])


        style_loss = 0
        for idx in range(19):
            ref_part_mask = ref_parts_masks[idx]
            ref_part_masked_img = ref_parts_masked_imgs[idx]
            target_part_mask = target_one_hot[:, idx:idx+1, :, :]

            fake_part_masked_img = fake_img * target_part_mask

            part_rec_loss = lpips_percept(refer_images, fake_img, mask=ref_part_mask * target_part_mask).mean()
            part_style_loss = style(ref_part_masked_img, fake_part_masked_img, mask1=ref_part_mask, mask2=target_part_mask)
            style_loss += part_style_loss * 3e6 + part_rec_loss * 2e-2

        n_loss = noise_regularize(injected_noise) * args.noise_regularize

        w_domain_regularizer = F.l1_loss(latent_in, w_plus_align_init.detach()) * 1e2

        fmap_loss = F.mse_loss(fmap_gen, fmap_target) * 1e1

        loss = style_loss + n_loss + w_domain_regularizer + fmap_loss + seg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        noise_normalize_(injected_noise)

        if (i + 1) % 10 == 0:
            gen_out = make_image(fake_img)
            target_parsing = target_one_hot[0].detach().cpu().numpy().argmax(0)
            target_parsing = cv2.resize(target_parsing, (gen_out.shape[2], gen_out.shape[1]), interpolation=cv2.INTER_NEAREST)
            for n, name in enumerate(img_names):
                img_name = save_dir + name + "-aligned-iter{}.png".format(i)
                gen_seg_parsing = seg_pred[n].detach().cpu().numpy()
                gen_seg_parsing = cv2.resize(gen_seg_parsing, (gen_out.shape[2], gen_out.shape[1]), interpolation=cv2.INTER_NEAREST)

                vis_gen_seg = vis_parsing_maps(gen_out[n], gen_seg_parsing, stride=1)
                vis_gen_parsing = vis_parsing_maps(gen_out[n], target_parsing, stride=1)
                vis_res = np.concatenate([gen_out[n], vis_gen_seg[..., ::-1], vis_gen_parsing[..., ::-1]], axis=1)
                pil_img = Image.fromarray(vis_res)
                pil_img.save(img_name)


        pbar.set_description(
            (
                f"seg_loss: {seg_loss.item():.4f}; "
                f"style_loss: {style_loss.item():.4f}; "
                f"noise regularize: {n_loss.item():.4f}; "
                f"fmap_loss: {fmap_loss.item():.4f}; "
                f"w_domain_regularizer: {w_domain_regularizer.item():.4f}; "
            )
        )

    for i, filename in enumerate(img_names):
        noise_single = []
        for noise in injected_noise:
            noise_single.append(noise[i:i + 1])

        result_file = {}
        result_file['latent'] = latent_in[i:i + 1]
        result_file['noise'] = noise_single
        torch.save(result_file, os.path.join(save_dir, filename + '_aligned.pt'))

