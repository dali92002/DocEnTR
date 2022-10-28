import torch
from vit_pytorch import ViT
from models.binae import BinModel
import torch.optim as optim
from einops import rearrange
import load_data
import utils as utils
from  config import Configs
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# get utils functions
count_psnr = utils.count_psnr
imvisualize = utils.imvisualize
load_data_func = load_data.load_datasets

def build_model(setting, image_size, patch_size):
    """
    Build model depending on its size

    Args:
        setting (str): model size (small/base/large)
        image_size (int, int): ihabe height and width
        patch_size (int): patch size for the vit
    Returns:
        model (BinModel): the built model to be trained
    """
    # define hyperparameters for the models depending on size
    hyper_params = {"base": [6, 8, 768],
                    "small": [3, 4, 512],
                    "large": [12, 16, 1024]} 

    encoder_layers = hyper_params[setting][0]
    encoder_heads = hyper_params[setting][1]
    encoder_dim = hyper_params[setting][2]

    # define encoder
    v = ViT(
        image_size = image_size,
        patch_size = patch_size,
        num_classes = 1000,
        dim = encoder_dim,
        depth = encoder_layers,
        heads = encoder_heads,
        mlp_dim = 2048
    )

    # define full model
    model = BinModel(
        encoder = v,
        decoder_dim = encoder_dim,      
        decoder_depth = encoder_layers,
        decoder_heads = encoder_heads  
    )
    return model

def visualize(model, epoch, validloader, image_size, patch_size):
    """
    Visualize the result on the validation set and show the validation loss

    Args:
        model (BinModel): the model
        epoch (str): the current epoch
        validloader (Dataloder): the vald data loader
        image_size (int, int): image size
        patch_size (int): ViT used patch size
    """
    losses = 0
    for _, (valid_index, valid_in, valid_out) in enumerate(validloader):
        bs = len(valid_in)
        inputs = valid_in.to(device)
        outputs = valid_out.to(device)
        with torch.no_grad():
            loss,_, pred_pixel_values = model(inputs,outputs)            
            rec_patches = pred_pixel_values
            rec_images = rearrange(rec_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                                 p1 = patch_size, p2 = patch_size,  h=image_size[0]//patch_size)
            for j in range (0,bs):
                imvisualize(inputs[j].cpu(), outputs[j].cpu(),
                            rec_images[j].cpu(), valid_index[j],
                            epoch, experiment)
            losses += loss.item()
    print('valid loss: ', losses / len(validloader))

def valid_model(model, data_path, epoch, experiment, valid_dibco):
    """
    Count PSNR of current epoch and priny it compared to the last best
    one

    Args:
        model (BinModel): the model
        data_path (str): path of the data folder
        epoch (int): the current epoch
        experiment (str): the name of the experiment
        valid_dibco (str): the validation data set

    """
    global best_psnr
    global best_epoch
    print('last best psnr: ', best_psnr, 'epoch: ', best_epoch)
    psnr  = count_psnr(epoch, data_path, valid_data=valid_dibco, setting=experiment)
    print('curr psnr: ', psnr)

    # change the best psnr to best epoch and save model if it is the case
    if psnr >= best_psnr:
        best_psnr = psnr
        best_epoch = epoch
        if not os.path.exists('./weights/'):
            os.makedirs('./weights/')
        torch.save(model.state_dict(), './weights/best-model_' +
                 str(TPS)+'_' + valid_dibco + experiment + '.pt')
        # keep only the best epoch images (for storage constraints)    
        dellist = os.listdir('vis'+experiment)
        dellist.remove('epoch'+str(epoch))
        for dl in dellist:
            os.system('rm -r vis'+experiment+'/'+dl)
    else:
        os.system('rm -r vis'+experiment+'/epoch'+str(epoch))

best_psnr = 0
best_epoch = 0

if __name__ == "__main__":
    # get configs
    cfg = Configs().parse()
    SPLITSIZE = cfg.split_size
    setting = cfg.vit_model_size
    TPS = cfg.vit_patch_size
    batch_size = cfg.batch_size
    valid_dibco = cfg.validation_dataset
    data_path = cfg.data_path
    patch_size = TPS
    image_size =  (SPLITSIZE,SPLITSIZE)
    vis_results = True
    
    # set experiment name
    experiment = setting +'_'+ str(SPLITSIZE)+'_' + str(TPS)
    
    # get dataloaders
    trainloader, validloader, _ = load_data.all_data_loader(batch_size)
    
    # get model
    model = build_model(setting, image_size, patch_size)
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(),lr=1.5e-4, betas=(0.9, 0.95),
                         eps=1e-08, weight_decay=0.05, amsgrad=False)

    # train the model for the specified epochs
    for epoch in range(1,cfg.epochs): 
        running_loss = 0.0
        for i, (train_index, train_in, train_out) in enumerate(trainloader):
            # get input/target pairs
            inputs = train_in.to(device)
            outputs = train_out.to(device)
            optimizer.zero_grad()
            # forward pass
            loss, _,_= model(inputs,outputs)
            # backward pass
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # display loss
            show_every = int(len(trainloader) / 7)
            if i % show_every == show_every-1:    # print every n mini-batches. here n = len(data)/7
                print('[Epoch: %d, Iter: %5d] Train loss: %.3f' % (epoch, i + 1, running_loss / show_every))
                running_loss = 0.0
        # visialize result and valid loss
        if vis_results:
            visualize(model, str(epoch), validloader, image_size, patch_size)
            valid_model(model, data_path, epoch, experiment, valid_dibco)