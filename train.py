from typing import Any
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



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



count_psnr = utils.count_psnr
imvisualize = utils.imvisualize
load_data_func = loadData.loadData_sets


transform = transforms.Compose([transforms.RandomResizedCrop(256),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])




SPLITSIZE = cfg.split_size
SETTING = cfg.vit_model_size
TPS = cfg.vit_patch_size

batch_size = cfg.batch_size

experiment = SETTING +'_'+ str(SPLITSIZE)+'_' + str(TPS)

patch_size = TPS
image_size =  (SPLITSIZE,SPLITSIZE)

MASKINGRATIO = 0.5
VIS_RESULTS = True
VALID_DIBCO = cfg.validation_dataset


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


def all_data_loader():
    data_train, data_valid, data_test = load_data_func()
    train_loader = torch.utils.data.DataLoader(data_train, collate_fn=sort_batch, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(data_valid, collate_fn=sort_batch, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test, collate_fn=sort_batch, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, valid_loader, test_loader

trainloader, validloader, testloader = all_data_loader()




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
    masking_ratio = MASKINGRATIO,   ## __ doesnt matter for binarization
    decoder_dim = ENCODERDIM,      
    decoder_depth = ENCODERLAYERS,
    decoder_heads = ENCODERHEADS       # anywhere from 1 to 8
)



model = model.to(device)


optimizer = optim.AdamW(model.parameters(),lr=1.5e-4, betas=(0.9, 0.95), eps=1e-08, weight_decay=0.05, amsgrad=False)


def visualize(epoch):
    losses = 0
    for i, (valid_index, valid_in, valid_out) in enumerate(validloader):
        # inputs, labels = data
        bs = len(valid_in)

        inputs = valid_in.to(device)
        outputs = valid_out.to(device)

        with torch.no_grad():
            loss,_, pred_pixel_values = model(inputs,outputs)
            
            rec_patches = pred_pixel_values

            rec_images = rearrange(rec_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size,  h=image_size[0]//patch_size)
            
            for j in range (0,bs):
                imvisualize(inputs[j].cpu(),outputs[j].cpu(),rec_images[j].cpu(),valid_index[j],epoch,experiment)
            
            losses += loss.item()
    
    print('valid loss: ', losses / len(validloader))



def valid_model(epoch):
    global best_psnr
    global best_epoch

    print('last best psnr: ', best_psnr, 'epoch: ', best_epoch)
    
    psnr  = count_psnr(epoch,valid_data=VALID_DIBCO,setting=experiment)
    print('curr psnr: ', psnr)


    if psnr >= best_psnr:
        best_psnr = psnr
        best_epoch = epoch
        
        if not os.path.exists('./weights/'):
            os.makedirs('./weights/')
    
        torch.save(model.state_dict(), './weights/best-model_'+str(TPS)+'_'+VALID_DIBCO+experiment+'.pt')

        dellist = os.listdir('vis'+experiment)
        dellist.remove('epoch'+str(epoch))

        for dl in dellist:
            os.system('rm -r vis'+experiment+'/'+dl)
    else:
        os.system('rm -r vis'+experiment+'/epoch'+str(epoch))
    


for epoch in range(1,cfg.epochs): 

    running_loss = 0.0
    
    for i, (train_index, train_in, train_out) in enumerate(trainloader):
        
        inputs = train_in.to(device)
        outputs = train_out.to(device)

        optimizer.zero_grad()

        loss, _,_= model(inputs,outputs)

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        show_every = int(len(trainloader) / 7)

        if i % show_every == show_every-1:    # print every 20 mini-batches
            print('[Epoch: %d, Iter: %5d] Train loss: %.3f' % (epoch, i + 1, running_loss / show_every))
            running_loss = 0.0
       
    
    if VIS_RESULTS:
        visualize(str(epoch))
        valid_model(epoch)


