import torch
from vit_pytorch import ViT
from models.binae import BinModel
from einops import rearrange
import  load_data
import utils as utils
from  config import Configs

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
    
    # build encodet ViT
    v = ViT(
        image_size = image_size,
        patch_size = patch_size,
        num_classes = 1000,
        dim = encoder_dim,
        depth = encoder_layers,
        heads = encoder_heads,
        mlp_dim = 2048
    )

    # build full model
    model = BinModel(
        encoder = v,
        decoder_dim = encoder_dim,      
        decoder_depth = encoder_layers,
        decoder_heads = encoder_heads  
    )
    return model

def visualize(model, epoch, testloader, image_size, patch_size):
    """
    Visualize the result on the test set and show the test loss

    Args:
        model (BinModel): the model
        epoch (str): the current epoch
        testloader (Dataloder): the test data loader
        image_size (int, int): image size
        patch_size (int): ViT used patch size
    """
    losses = 0
    for _, (test_index, test_in, test_out) in enumerate(testloader):
        bs = len(test_in)
        inputs = test_in.to(device)
        outputs = test_out.to(device)
        with torch.no_grad():
            loss,_, pred_pixel_values = model(inputs,outputs)
            rec_patches = pred_pixel_values
            rec_images = rearrange(rec_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                                 p1 = patch_size, p2 = patch_size,  h=image_size[0]//patch_size)
            for j in range (0,bs):
                imvisualize(inputs[j].cpu(), outputs[j].cpu(), rec_images[j].cpu(), test_index[j], 
                        epoch, experiment)
            losses += loss.item()
    print('test loss: ', losses / len(testloader))

def valid_model(epoch, data_path,  test_dibco, experiment, flipped, THRESHOLD):
    """
    Count PSNR of test images

    Args:
        epoch (int): the current epoch (testing)
        data_path (str): path of the data folder
        test_dibco (str): the testing data set
        experiment (str): the name of the experiment
        flipped (bool): whether the images are flipped
        THRESHOLD (float): final binarization thresold after the model output, between 0 and 1.
    Returns:
        psnr (float): the psnd of the full testing data
    """
    psnr  = count_psnr(epoch, data_path,  valid_data=test_dibco, setting=experiment, flipped=flipped , thresh=THRESHOLD)
    return psnr

if __name__ == "__main__":
    
    flipped = False
    THRESHOLD = 0.5
    epoch = "_testing"
    # get configs
    cfg = Configs().parse()
    SPLITSIZE = cfg.split_size
    setting = cfg.vit_model_size
    TPS = cfg.vit_patch_size
    batch_size = cfg.batch_size
    test_dibco = cfg.testing_dataset
    data_path = cfg.data_path
    
    # set variables
    experiment = setting +'_'+ str(SPLITSIZE)+'_' + str(TPS)
    patch_size = TPS
    image_size =  (SPLITSIZE,SPLITSIZE)

    # build model
    model  = build_model(setting, image_size, patch_size)
    model = model.to(device)

    # load trained weights
    model_path = cfg.model_weights_path
    model.load_state_dict(torch.load(model_path))
    _, _, testloader = load_data.all_data_loader(batch_size)

    # visualize images, count and print PSNR 
    visualize(model, str(epoch), testloader, image_size, patch_size)
    print('Test PSNR: ', valid_model(epoch, data_path,  test_dibco, experiment, flipped, THRESHOLD))