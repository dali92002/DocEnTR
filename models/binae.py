import torch
from torch import nn
import torch.nn.functional as F
from vit_pytorch.vit import Transformer


class BinModel(nn.Module):
    """
    The autoencoder model to enhance images in an image to image translation fashion.
    This code is built on top of the vit-pytorch code https://github.com/lucidrains/vit-pytorch.

    Args:
        encoder (model): the defined encoder, hete it is a ViT
        decoder_dim (int): decoder dim (embedding size)
        decoder_depth (int): number of decoder layers
        decoder_heads (int): number of decoder heads
        decoder_dim_head (int): decoder head dimension
    """
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64
    ):
        super().__init__()
        # extract hyperparameters and functions from the ViT encoder.
        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]

        # define your decoder here
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, img, gt_img):
        
        # get patches and their number
        patches = self.to_patch(img)
        _, num_patches, *_ = patches.shape

        # project pixel patches to tokens and add positions
        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        # encode tokens by the encoder
        encoded_tokens = self.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal.
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # decode tokens with decoder
        decoded_tokens = self.decoder(decoder_tokens)

        # project tokens to pixels
        pred_pixel_values = self.to_pixels(decoded_tokens)

        # calculate the loss with gt
        gt_patches = self.to_patch(gt_img)
        loss = F.mse_loss(pred_pixel_values, gt_patches)

        return loss, patches, pred_pixel_values
