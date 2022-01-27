from typing import Any
from thinc import config
import torch
from vit_pytorch import ViT
from models.binae import BINMODEL
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
from einops import rearrange
import loadData2 as loadData
import utils as utils
from  config import Configs
import os


cfg = Configs().parse()

FLIPPED = False
THRESHOLD = 0.5


SPLITSIZE = cfg.split_size
SETTING = cfg.vit_model_size
TPS = cfg.vit_patch_size

batch_size = cfg.batch_size

experiment = SETTING +'_'+ str(SPLITSIZE)+'_' + str(TPS)

patch_size = TPS
image_size =  (SPLITSIZE,SPLITSIZE)

MASKINGRATIO = 0.5
VIS_RESULTS = True
TEST_DIBCO = cfg.testing_dataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
count_psnr = utils.count_psnr
imvisualize = utils.imvisualize
load_data_func = loadData.loadData_sets


if SETTING == 'base':
    ENCODERLAYERS = 6
    ENCODERHEADS = 8
    ENCODERDIM = 768

if SETTING == 'small':
    ENCODERLAYERS = 3
    ENCODERHEADS = 4
    ENCODERDIM = 512

if SETTING == 'large':
    ENCODERLAYERS = 12
    ENCODERHEADS = 16
    ENCODERDIM = 1024



best_psnr = 0
best_epoch = 0


def sort_batch(batch):
    n_batch = len(batch)
    train_index = []
    train_in = []
    train_out = []
    for i in range(n_batch):
        idx, img, gt_img = batch[i]

        train_index.append(idx)
        train_in.append(img)
        train_out.append(gt_img)

    train_index = np.array(train_index)
    train_in = np.array(train_in, dtype='float32')
    train_out = np.array(train_out, dtype='float32')

    train_in = torch.from_numpy(train_in)
    train_out = torch.from_numpy(train_out)

    return train_index, train_in, train_out


def test_data_loader():
    _, _, data_test = load_data_func(flipped=FLIPPED)
    test_loader = torch.utils.data.DataLoader(data_test, collate_fn=sort_batch, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return  test_loader

test_loader = test_data_loader()




v = ViT(
    image_size = image_size,
    patch_size = patch_size,
    num_classes = 1000,
    dim = ENCODERDIM,
    depth = ENCODERLAYERS,
    heads = ENCODERHEADS,
    mlp_dim = 2048
)

model = BINMODEL(
    encoder = v,
    masking_ratio = MASKINGRATIO,   # __ doesnt matter for binarization
    decoder_dim = ENCODERDIM,      
    decoder_depth = ENCODERLAYERS,
    decoder_heads = ENCODERHEADS       
)



model = model.to(device)
optimizer = optim.AdamW(model.parameters(),lr=1.5e-4, betas=(0.9, 0.95), eps=1e-08, weight_decay=0.05, amsgrad=False)


def visualize(epoch):
    losses = 0
    for i, (test_index, test_in, test_out) in enumerate(test_loader):
        # inputs, labels = data
        bs = len(test_in)

        inputs = test_in.to(device)
        outputs = test_out.to(device)

        with torch.no_grad():
            loss,_, pred_pixel_values = model(inputs,outputs)
            
            rec_patches = pred_pixel_values

            rec_images = rearrange(rec_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size,  h=image_size[0]//patch_size)
            
            for j in range (0,bs):
                imvisualize(inputs[j].cpu(),outputs[j].cpu(),rec_images[j].cpu(),test_index[j],epoch,experiment)
            
            losses += loss.item()
    
    print('valid loss: ', losses / len(test_loader))



def valid_model(epoch):
    
    psnr  = count_psnr(epoch,valid_data=TEST_DIBCO,setting=experiment,flipped=FLIPPED , thresh=THRESHOLD)
    print('Test PSNR: ', psnr)



model_name = cfg.model_weights_path
model.load_state_dict(torch.load('./weights/'+model_name))

epoch = "_testing"

visualize(str(epoch))
valid_model(epoch)
