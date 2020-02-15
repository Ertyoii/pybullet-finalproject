import torch
from model import *


def infer(seg_img, record):
    model = NaiveNet()
    model.load_state_dict(torch.load(record, map_location=torch.device('cpu')))
    seg_img = seg_img.reshape(1, 1, 256, 256)
    seg_img = torch.Tensor(seg_img)

    return model(seg_img).cpu().detach().numpy()[0]