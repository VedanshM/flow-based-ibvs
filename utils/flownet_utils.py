import torch
import numpy as np
from PIL import Image

from flownet.models import FlowNet2  
from flownet.utils.flow_utils import flow2img, visulize_flow_file, writeFlow

WEIGHTS_PATH = './data/FlowNet2_checkpoint.pth.tar'
IMG1_PATH = './data/a.png'
IMG2_PATH = './data/b.png'
OUTPATH = './data/flow.png'

class args:
    fp16 = False
    rgb_max = 255.0

net = FlowNet2(args()).cuda()
net.load_state_dict(torch.load(WEIGHTS_PATH)["state_dict"])


def get_flow(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    images = [img1, img2]
    images = np.array(images).transpose(3, 0, 1, 2)
    images = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()

    flow = net(images).squeeze()
    flow = flow.data.cpu().numpy().transpose(1, 2, 0)
    return flow


flow_arr_to_file = writeFlow

flow_file_to_png = visulize_flow_file

flow_arr_to_imgarr = flow2img

def flow_arr_to_img(flow, path):
    img_arr = flow2img(flow)
    Image.fromarray(img_arr).save(path)

def load_img_from_path(path:str)->np.ndarray:
    return np.array(Image.open(path).convert('RGB').resize((512, 384)))

def main():
    flow = get_flow(load_img_from_path(IMG1_PATH),
                    load_img_from_path(IMG2_PATH))
    print(flow.shape)
    flow_arr_to_img(flow, OUTPATH)

if __name__ == '__main__':
    main()
