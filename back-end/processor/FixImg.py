# -*-coding:utf-8-*-
import torch
from torchvision.utils import save_image
from models.tiny_unet_2 import Decoder
from torchvision import transforms as T
from PIL import Image
import time
import numpy as np

class FixImg(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.generator = self.init_model()

    def init_model(self):
        """create a generator and a discrimin"""
        model = Decoder(32)
        # self.print_network(self.generator, 'G')
        model.to(self.device)
        model.load_state_dict(torch.load("weights/offlinemodel.ckpt", map_location=lambda storage, loc: storage))
        return model

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        out = out.squeeze()
        out = out.permute(1,2,0)
        return out.clamp_(0, 1)

    def generateone(self, img):
        start_time = time.time()
        img = self.transform(img)
        img = img.to(self.device)
        img = img.unsqueeze(0)
        image_info = {}
        with torch.no_grad():
            fixed_img = self.generator(img).data.cpu()
            fixed_img = self.denorm(fixed_img)
        fixed_img = fixed_img * 255
        fixed_img = fixed_img.numpy().astype(np.uint8)
        end_time = time.time()
        image_info[f"{fixed_img.shape[0]}x{fixed_img.shape[1]}"] = [np.round(end_time - start_time, 3),
                              np.round(1 / (end_time - start_time), 3)]
        return fixed_img, image_info


def main():
    fix = FixImg()
    output = fix.generateone("data/input/2.png")
    print(output)


if __name__ == '__main__':
    # parser.add_argument('--global_G_ngf', type=int, default=32, help='number of conv filters in the first layer of G')
    main()

