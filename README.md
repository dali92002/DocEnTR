# DocEnTR

Use Python version 3.8.12
 
## Description
Pytorch implementation of the paper [DocEnTr: An End-to-End Document Image Enhancement Transformer](https://arxiv.org/abs/2201.10252). This model is implemented on top of the [vit-pytorch](https://github.com/lucidrains/vit-pytorch) vision transformers library. The proposed model can be used to enhance (binarize) degraded document images, as shown in the following samples.
 
<table style="padding:10px">
    <tr>
        <td style="text-align:center">
            Degraded Images 
        </td>
        <td style="text-align:center">
            Our Binarization 
        </td>
    </tr>
    <tr>
        <td style="text-align:center"> 
            <img src="./git_images/3.png"  alt="1" width = 600px height = 300px >
        </td>
        <td style="text-align:center">
            <img src="./git_images/3_pred.png"  alt="2" width = 600px height = 300px>
        </td>
    </tr>
    <tr>
        <td style="text-align:center"> 
            <img src="./git_images/14.png"  alt="1" width = 600px height = 300px >
        </td>
        <td style="text-align:center">
            <img src="./git_images/14_pred.png"  alt="2" width = 600px height = 300px>
        </td>
    </tr>

</table>

## Download Code
clone the repository:
```bash
git clone https://github.com/dali92002/DocEnTR
cd DocEnTr
```
## Requirements
- install requirements.txt
## Process Data
### Data Path
We gathered the DIBCO, H-DIBCO and PALM datasets and organized them in one folder. You can download it from this [link](https://drive.google.com/file/d/16pIO4c-mA2kHc1I3uqMs7VwD4Jb4F1Vc/view?usp=sharing). After downloading, extract the folder named DIBCOSETS and place it in your desired data path. Means:  /YOUR_DATA_PATH/DIBCOSETS/
 
### Data Splitting
Specify the data path, split size, validation and testing sets to prepare your data. In this example, we set the split size as (256 X 256), the validation set as 2016 and the testing as 2018 while running the process_dibco.py file.
 
```bash
python process_dibco.py --data_path /YOUR_DATA_PATH/ --split_size 256 --testing_dataset 2018 --validation_dataset 2016
```
 
## Using DocEnTr
### Training
For training, specify the desired settings (batch_size, patch_size, model_size, split_size and training epochs) when running the file train.py. For example, for a base model with a patch_size of (16 X 16) and a batch_size of 32 we use the following command:
 
```bash
python train.py --data_path /YOUR_DATA_PATH/ --batch_size 32 --vit_model_size base --vit_patch_size 16 --epochs 151 --split_size 256 --validation_dataset 2016
```
You will get visualization results from the validation dataset on each epoch in a folder named vis+"YOUR_EXPERIMENT_SETTINGS" (it will be created). In the previous case it will be named visbase_256_16. Also, the best weights will be saved in the folder named "weights".
 
### Testing on a DIBCO dataset
To test the trained model on a specific DIBCO dataset (should be matched with the one specified in Section Process Data, if not, run process_dibco.py again). Download the model weights (In section Model Zoo), or use your own trained model weights. Then, run the following command. Here, I test on H-DIBCO 2018, using the Base model with 8X8 patch_size, and a batch_size of 16. The binarized images will be in the folder ./vis+"YOUR_CONFIGS_HERE"/epoch_testing/ 
```bash
python test.py --data_path /YOUR_DATA_PATH/ --model_weights_path  /THE_MODEL_WEIGHTS_PATH/  --batch_size 16 --vit_model_size base --vit_patch_size 8 --split_size 256 --testing_dataset 2018
```
### Demo
In this demo, we show how we can use our pretrained models to binarize a single degraded image, this is detailed with comments in the file named demo.ipynb for simplicity we make it a jupyter notebook where you can modify all the code parts and visualize your progresssive results.

## Model Zoo
In this section we release the pre-trained weights for all the best DocEnTr model variants trained on DIBCO benchmarks. 
<!-- <style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-amwm{font-weight:bold;text-align:center;vertical-align:top}
</style> -->
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow"></th>
    <th class="tg-c3ow">Testing data</th>
    <th class="tg-c3ow">Models</th>
    <th class="tg-c3ow">Patch size</th>
    <th class="tg-c3ow">URL</th>
    <th class="tg-baqh">PSNR</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow" rowspan="2">0</td>
    <td class="tg-c3ow" rowspan="2"><br>DIBCO 2011</td>
    <td class="tg-c3ow">DocEnTr-Base</td>
    <td class="tg-c3ow">8x8</td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/113X6gzFHTIkHZ3XYbyTcCWpQGV8QQzAs/view?usp=sharing" target="_blank" rel="noopener noreferrer">model</a></td>
    <td class="tg-amwm">20.81</td>
  </tr>
  <tr>
    <td class="tg-c3ow">DocEnTr-Large</td>
    <td class="tg-c3ow">16x16</td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/12UpSAVFJ90xly5hCqnaAu1_5gmxwFlD_/view?usp=sharing" target="_blank" rel="noopener noreferrer">model</a></td>
    <td class="tg-baqh">20.62</td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="2">1</td>
    <td class="tg-c3ow" rowspan="2"><br>H-DIBCO 2012</td>
    <td class="tg-c3ow">DocEnTr-Base</td>
    <td class="tg-c3ow">8x8</td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/1FKXAS8BetcB2pCwkOTNHIX4Rj5-tq-Ep/view?usp=sharing" target="_blank" rel="noopener noreferrer">model</a></td>
    <td class="tg-amwm">22.29</td>
  </tr>
  <tr>
    <td class="tg-c3ow">DocEnTr-Large</td>
    <td class="tg-c3ow">16x16</td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/1SwZLVJVJmm_o_kDcYDvLgx74gatUVGUh/view?usp=sharing" target="_blank" rel="noopener noreferrer">model</a></td>
    <td class="tg-baqh">22.04</td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="2">2</td>
    <td class="tg-c3ow" rowspan="2"><br>DIBCO 2017</td>
    <td class="tg-c3ow">DocEnTr-Base</td>
    <td class="tg-c3ow">8x8</td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/1ABR48OpTXV3hcGNGfkSNfvAHCQlztV1o/view?usp=sharing" target="_blank" rel="noopener noreferrer">model</a></td>
    <td class="tg-amwm">19.11</td>
  </tr>
  <tr>
    <td class="tg-c3ow">DocEnTr-Large</td>
    <td class="tg-c3ow">16x16</td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/1AlyPZJ7xviDggOKgeXE2kcVQQuz5BK6-/view?usp=sharing" target="_blank" rel="noopener noreferrer">model</a></td>
    <td class="tg-baqh">18.85</td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="2">3</td>
    <td class="tg-c3ow" rowspan="2"><br>H-DIBCO 2018</td>
    <td class="tg-c3ow">DocEnTr-Base</td>
    <td class="tg-c3ow">8x8</td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/1qnIDVA7C5BGInEIBT65OogT0N9ca_E97/view?usp=sharing" target="_blank" rel="noopener noreferrer">model</a></td>
    <td class="tg-baqh">19.46</td>
  </tr>
  <tr>
    <td class="tg-baqh">DocEnTr-Large</td>
    <td class="tg-baqh">16x16</td>
    <td class="tg-baqh"><a href="https://drive.google.com/file/d/1yCnFLTE6Yg3qHNCuERiTP5ErOka8jzZl/view?usp=sharing" target="_blank" rel="noopener noreferrer">model</a></td>
    <td class="tg-amwm">19.47</td>
  </tr>
</tbody>
</table>

## Citation

If you find this useful for your research, please cite it as follows:

```bash
@inproceedings{souibgui2022docentr,
  title={DocEnTr: An end-to-end document image enhancement transformer},
  author={ Souibgui, Mohamed Ali and Biswas, Sanket and  Jemni, Sana Khamekhem and Kessentini, Yousri and Forn{\'e}s, Alicia and Llad{\'o}s, Josep and Pal, Umapada},
  booktitle={2022 26th International Conference on Pattern Recognition (ICPR)},
  year={2022}
}
```
## Authors
- [Mohamed Ali Souibgui](https://github.com/dali92002)
- [Sanket Biswas](https://github.com/biswassanket)
## Conclusion
There should be no bugs in this code, but if there is, we are sorry for that :') !! 
