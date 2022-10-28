import argparse

class Configs():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--data_path', type=str, default='/home/mohamed/vit/DocEnTR/data/', help='specify your data path, better ending with the "/" ')
        self.parser.add_argument('--split_size', type=int, default=256, help= "better be a multiple of 8, like 128, 256, etc ..")
        self.parser.add_argument('--vit_patch_size', type=int, default=16 , help=" better be a multiple of 2 like 8, 16 etc ..")
        self.parser.add_argument('--vit_model_size', type=str, default='base', choices=['small', 'base', 'large'])
        self.parser.add_argument('--testing_dataset', type=str, default='2018')
        self.parser.add_argument('--validation_dataset', type=str, default='2016')
        self.parser.add_argument('--batch_size', type=int, default=8)
        self.parser.add_argument('--epochs', type=int, default=151, help= 'the desired training epochs')
        self.parser.add_argument('--model_weights_path', type=str, help= 'the desired trained model')
        
    def parse(self):
        return self.parser.parse_args()
