import os
import time

import torch
import torch.backends.cudnn as cudnn

import cv2
import numpy as np
from utils import craft_utils, file_utils, imgproc

from nets.nn import CRAFT, RefineNet
import yaml
from collections import OrderedDict

import matplotlib.pyplot as plt


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def test_net(
    net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None
):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5
    )
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = x.unsqueeze(0)  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # refine link
    # if refine_net is not None:
    #     with torch.no_grad():
    #         y_refiner = refine_net(y, feature)
    #     score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    # t0 = time.time() - t0
    # t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly
    )

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    # t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    # if args['show_time']: print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


with open(os.path.join("scanDoc\\utils", "config.yaml")) as f:
    args = yaml.load(f, Loader=yaml.FullLoader)


def crft_init():
    net = CRAFT()  # initialize

    print("Loading weights from checkpoint (" + args["trained_model"] + ")")
    if args["cuda"]:
        net.load_state_dict(copyStateDict(torch.load(args["trained_model"])))
    else:
        net.load_state_dict(
            copyStateDict(torch.load(args["trained_model"], map_location="cpu"))
        )

    return net.eval()


def get_recog_txt(net, image):
    text_imgs = []
    coord = []  # added for grouping images order !!!
    # preprocess image
    # image = imgproc.loadImage(image_path)
    # recognize bbox
    bboxes, _, _ = test_net(
        net,
        image,
        args["text_threshold"],
        args["link_threshold"],
        args["low_text"],
        args["cuda"],
        args["poly"],
        refine_net=False,
    )

    for i, box in enumerate(bboxes):
        # print(box.shape)
        coord_w = np.array(box).astype(np.int32)  # .reshape((-1))

        # print(coord_w.shape)

        # print(coord_w[0], coord_w[2])
        x = coord_w[0][0]
        x2 = coord_w[2][0]
        y = coord_w[0][1]
        y2 = coord_w[2][1]
        img = image[y:y2, x:x2]
        text_imgs.append(img)
        coord.append((x, x2, y, y2))
    return text_imgs, coord
