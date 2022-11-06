import torch
import os, tqdm
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from networks.face_parse_bisenet import BiSeNet

celebamask_label_list = ['background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear',
                         'mouth', 'u_lip',
                         'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

faceparsing_label_list = {0: 'background', 1: 'skin', 2: 'l_brow', 3: 'r_brow', 4: 'l_eye', 5: 'r_eye', 6: 'eye_g',
                          7: 'l_ear', 8: 'r_ear', 9: 'ear_r', 10: 'nose', 11: 'mouth', 12: 'u_lip', 13: 'l_lip',
                          14: 'neck',
                          15: 'neck_l', 16: 'cloth', 17: 'hair', 18: 'hat'}


def relabel_parsing_anno(parsing_anno):
    relabeled = np.zeros_like(parsing_anno)
    for idx in faceparsing_label_list.keys():
        celebamask_label = celebamask_label_list.index(faceparsing_label_list[idx])
        # print(idx, celebamask_label)
        relabeled = np.where(parsing_anno == idx, np.full_like(relabeled, int(celebamask_label)), relabeled)
    return relabeled


def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    relabeled_parsing_anno = relabel_parsing_anno(parsing_anno)
    # print(relabeled_parsing_anno.shape)
    vis_parsing_anno = relabeled_parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)

    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        # cv2.imwrite(save_path[:-4] + '.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return vis_im


def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):
    os.makedirs(respth, exist_ok=True)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join(cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if dspth.endswith('.txt'):
        img_paths = []
        img_path_file = open(dspth, 'r')
        for line in img_path_file:
            img_paths.append(line.strip('\n'))

    with torch.no_grad():
        for image_path in tqdm.tqdm(img_paths[0:2]):
            img_name = image_path.split('/')[-1]
            img = Image.open(image_path)
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(parsing)
            # print(np.unique(parsing))

            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, img_name))


# atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
#         10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']

def heuristic_gen_mask(mask1_path, mask2_path, save_path):
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    mask1 = np.array(Image.open(osp.join(mask1_path)))
    mask2 = np.array(Image.open(osp.join(mask2_path)))
    new_mask = np.zeros_like(mask1)
    new_mask = np.where(np.bitwise_or(mask2 == 1, mask1 == 1), mask2, new_mask)
    new_mask = np.where(np.bitwise_and(mask2 > 1, mask2 < 13), mask2, new_mask)
    new_mask = np.where(mask1 == 13, mask1, new_mask)
    new_mask = np.where(mask2 > 13, mask2, new_mask)

    num_of_class = np.max(new_mask)
    vis_parsing_anno_color = np.zeros((new_mask.shape[0], new_mask.shape[1], 3)) + 255
    for pi in range(1, num_of_class + 1):
        index = np.where(new_mask == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]
    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    cv2.imwrite(save_path[:-4] + '.png', new_mask)
    cv2.imwrite(save_path, vis_parsing_anno_color, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


if __name__ == "__main__":
    imgs_txt = 'latents_img_path.txt'
    save_dir = 'bowlcut_edited_celeba_masks/'
    evaluate(dspth=imgs_txt, respth=save_dir, cp='79999_iter.pth')
