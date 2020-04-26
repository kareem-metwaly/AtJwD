import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
import cv2, os
from tqdm import trange
import utils
import timeit

parser = argparse.ArgumentParser(description="Pytorch AtJw(+D) Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", type=str, default="AtJw", help="model path")
parser.add_argument("--dataset", default="./testset/", type=str, help="dataset path")


def get_image_for_save(img):
    # img = img.data[0].numpy()

    img = img * 255.
    img[img < 0] = 0
    img[img > 255.] = 255.
    img = np.rollaxis(img, 0, 3)
    img = img.astype('uint8')
    return img

def shift_img(img, dirction=(0,0)):
    h1 = h2 = w1 = w2 = 0
    if not dirction[0]==0:
        if dirction[0] < 0:
            h1 = abs(dirction[0])
            h2 = 0
        elif dirction[0] > 0:
            h1 = 0
            h2 = dirction[0]

    if not dirction[1]==0:
        if dirction[1] < 0:
            w1 = abs(dirction[1])
            w2 = 0
        elif dirction[1] > 0:
            w1 = 0
            w2 = dirction[1]
    print(">>padding ... {}".format((h1, h2, w1, w2)))
    img = np.pad(img, ((h1, h2), (w1, w2), (0, 0)), 'reflect')
    return img

def crop_image_back(img, dirction=(0,0)):
    img = img.data[0].numpy()
    _, H, W = img.shape
    h1 = w1 = 0
    h2 = H
    w2 = W
    if not dirction[0] == 0:
        if dirction[0] < 0:
            h1 = abs(dirction[0])
            h2 = H
        elif dirction[0] > 0:
            h1 = 0
            h2 = -dirction[0]

    if not dirction[1] == 0:
        if dirction[1] < 0:
            w1 = abs(dirction[1])
            w2 = W
        elif dirction[1] > 0:
            w1 = 0
            w2 = -dirction[1]

    img = img[:, h1:h2, w1:w2]
    return img

def avg_imgs(imgs):
    number_of_imgs = len(imgs)
    img = sum(imgs) / np.float32(number_of_imgs)
    return img

opt = parser.parse_args()
cuda = opt.cuda
save_path = os.path.join('results', opt.model)
utils.checkdirctexist(save_path)

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

model_path = os.path.join('model', "{}.pth".format(opt.model))
model, _,dict = utils.load_checkpoint(model_path, iscuda=opt.cuda)

if cuda:
    model = model.cuda()

image_list = glob.glob(os.path.join(opt.dataset,'*.png'))

count = 0.0
with torch.no_grad():
    if cuda:
        model = model.cuda()

    else:
        model = model.cpu()

    model.eval()

    os.makedirs(save_path, exist_ok=True)

    start = timeit.default_timer()
    for image_name in image_list:
        count += 1
        print("Processing ", image_name)
        og_img = cv2.imread(image_name)
        og_img = og_img.astype(float)
        og_H, og_W, og_C = og_img.shape
        pads = [(0,0)]
        imgs_J_total = []
        for pad in pads:
            img = shift_img(og_img, pad)
            H, W, C = img.shape
            Wk = W
            Hk = H
            if W % 32:
                Wk = W + (32 - W % 32)
            if H % 32:
                Hk = H + (32 - H % 32)
            img = np.pad(img, ((0, Hk-H), (0,Wk-W), (0,0)), 'reflect')

            im_input = img/255.0
            im_input = np.expand_dims(np.rollaxis(im_input, 2),  axis=0)
            im_input = Variable(torch.from_numpy(im_input).float())
            if cuda:
                im_input = im_input.cuda()

            im_output_run, J_direct, J_AT, im_A, im_t, w = model(im_input, opt)
            im_output_J_total = crop_image_back(im_output_run.cpu(), dirction=pad)
            imgs_J_total.append(im_output_J_total)
            
        num_imgs = np.float32(len(pads))
        img_avg = np.empty((3, og_H, og_W))
        for img in imgs_J_total:
            img_croped = img[:, 0:og_H, 0:og_W]
            img_avg += img_croped
        im_output_forsave = get_image_for_save(img_avg/num_imgs)
        path, filename = os.path.split(image_name)
        cv2.imwrite(os.path.join(save_path, "{}".format(filename)), im_output_forsave)

    stop = timeit.default_timer()
print("Save_path=", save_path)
print("It takes average {}s for processing".format((stop-start)/count))

