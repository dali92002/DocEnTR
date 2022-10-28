import numpy as np
import tempfile
import torch
import cv2
from vit_pytorch import ViT
from einops import rearrange
from cog import BasePredictor, Input, Path

from models.binae import BinModel


THRESHOLD = 0.5  ## binarization threshold after the model output
SPLITSIZE = 256  ## your image will be divided into patches of 256x256 pixels
image_size = (
    SPLITSIZE,
    SPLITSIZE,
)  ## your image will be divided into patches of 256x256 pixels


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.settings = {
            "base": {
                "ENCODERLAYERS": 6,
                "ENCODERHEADS": 8,
                "ENCODERDIM": 768,
                "patch_size": 8,
            },
            "large": {
                "ENCODERLAYERS": 12,
                "ENCODERHEADS": 16,
                "ENCODERDIM": 1024,
                "patch_size": 16,
            },
        }

        v = {
            k: ViT(
                image_size=image_size,
                patch_size=self.settings[k]["patch_size"],
                num_classes=1000,
                dim=self.settings[k]["ENCODERDIM"],
                depth=self.settings[k]["ENCODERLAYERS"],
                heads=self.settings[k]["ENCODERHEADS"],
                mlp_dim=2048,
            )
            for k in ["base", "large"]
        }

        self.models = {
            k: BinModel(
                encoder=v[k],
                decoder_dim=self.settings[k]["ENCODERDIM"],
                decoder_depth=self.settings[k]["ENCODERLAYERS"],
                decoder_heads=self.settings[k]["ENCODERHEADS"],
            )
            for k in ["base", "large"]
        }

        self.models["base"].to(self.device)
        self.models["large"].to(self.device)

        base_model_path = "weights/best-model_8_2018base_256_8.pt" # weights are pre-downloaded
        self.models["base"].load_state_dict(
            torch.load(base_model_path, map_location=self.device)
        )

        large_model_path = "weights/best-model_16_2018large_256_16.pt"
        self.models["large"].load_state_dict(
            torch.load(large_model_path, map_location=self.device)
        )

    def predict(
        self,
        image: Path = Input(
            description="Input Image.",
        ),
        model_size: str = Input(
            default="base",
            choices=["base", "large"],
            description="Choose the model size.",
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        
        deg_image = cv2.imread(str(image)) / 255

        ## Split the image intop patches, an image is padded first to make it dividable by the split size
        h = ((deg_image.shape[0] // 256) + 1) * 256
        w = ((deg_image.shape[1] // 256) + 1) * 256
        deg_image_padded = np.ones((h, w, 3))
        deg_image_padded[: deg_image.shape[0], : deg_image.shape[1], :] = deg_image
        patches = split(deg_image_padded, deg_image.shape[0], deg_image.shape[1])

        ## preprocess the patches (images)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        out_patches = []
        for p in patches:
            out_patch = np.zeros([3, *p.shape[:-1]])
            for i in range(3):
                out_patch[i] = (p[:, :, i] - mean[i]) / std[i]
            out_patches.append(out_patch)

        result = []
        for patch_idx, p in enumerate(out_patches):
            print(f"({patch_idx} / {len(out_patches) - 1}) processing patch...")
            p = np.array(p, dtype="float32")
            train_in = torch.from_numpy(p)

            with torch.no_grad():
                train_in = train_in.view(1, 3, SPLITSIZE, SPLITSIZE).to(self.device)
                _ = torch.rand((train_in.shape)).to(self.device)

                loss, _, pred_pixel_values = self.models[model_size](train_in, _)

                rec_patches = pred_pixel_values

                rec_image = torch.squeeze(
                    rearrange(
                        rec_patches,
                        "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                        p1=self.settings[model_size]["patch_size"],
                        p2=self.settings[model_size]["patch_size"],
                        h=image_size[0] // self.settings[model_size]["patch_size"],
                    )
                )

                impred = rec_image.cpu().numpy()
                impred = np.transpose(impred, (1, 2, 0))

                for ch in range(3):
                    impred[:, :, ch] = (impred[:, :, ch] * std[ch]) + mean[ch]

                impred[np.where(impred > 1)] = 1
                impred[np.where(impred < 0)] = 0
            result.append(impred)

        clean_image = merge_image(
            result, deg_image_padded.shape[0], deg_image_padded.shape[1]
        )
        clean_image = clean_image[: deg_image.shape[0], : deg_image.shape[1], :]
        clean_image = (clean_image > THRESHOLD) * 255

        output_path = Path(tempfile.mkdtemp()) / "output.png"

        cv2.imwrite(str(output_path), clean_image)
        return output_path


def split(im, h, w):
    """
    split image into patches

    Args:
        im (np.array): image to be splitted
        h (int): image height
        w (int): image width
    Returns:
        patches [np.array, ..., np.array]: list of patches with size SPLITSIZExSPLITSIZE
                                         obtained from image
    """
    patches = []
    nsize1 = SPLITSIZE
    nsize2 = SPLITSIZE
    for ii in range(0, h, nsize1):  # 2048
        for iii in range(0, w, nsize2):  # 1536
            patches.append(im[ii : ii + nsize1, iii : iii + nsize2, :])

    return patches


def merge_image(splitted_images, h, w):
    """
    merge patches into image and return it

    Args:
        splitted_images [np.array, ..., np.array]: list of patches to costruct the image
        h (int): final image height
        w (int): final image width
    Returns:
        image (np.array): the constructed image
    """
    image = np.zeros(((h, w, 3)))
    nsize1 = SPLITSIZE
    nsize2 = SPLITSIZE
    ind = 0
    for ii in range(0, h, nsize1):
        for iii in range(0, w, nsize2):
            image[ii : ii + nsize1, iii : iii + nsize2, :] = splitted_images[ind]
            ind += 1
    return image
