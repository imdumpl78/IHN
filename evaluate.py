import sys

sys.path.append('core')

from PIL import Image
import argparse
import os
import numpy as np
import torch
import torchvision

import datasets_4cor_img as datasets
from utils import *

@torch.no_grad()
def validate_process(model, args):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    mace_list = []
    args.batch_size = 1
    val_dataset = datasets.fetch_dataloader(args, split='validation')
    for i_batch, data_blob in enumerate(val_dataset):
        image1, image2, flow_gt,  H  = [x.to(model.device) for x in data_blob]
        flow_gt = flow_gt.squeeze(0)
        flow_4cor = torch.zeros((2, 2, 2))
        flow_4cor[:, 0, 0] = flow_gt[:, 0, 0]
        flow_4cor[:, 0, 1] = flow_gt[:, 0, -1]
        flow_4cor[:, 1, 0] = flow_gt[:, -1, 0]
        flow_4cor[:, 1, 1] = flow_gt[:, -1, -1]

        if i_batch == 0:
            if not os.path.exists('watch'):
                os.makedirs('watch')
            save_img(torchvision.utils.make_grid(image1, nrow=16, padding = 16, pad_value=255),
                    './watch/' + "b1_epoch_" + str(i_batch).zfill(5) + "_iter_" + '.bmp')
            save_img(torchvision.utils.make_grid(image2, nrow=16, padding = 16, pad_value=255),
                    './watch/' + "b2_epoch_" + str(i_batch).zfill(5) + "_iter_" + '.bmp')

        image1 = image1.to(model.device)
        image2 = image2.to(model.device)
        four_pr = model(image1, image2, iters_lev0 = args.iters_lev0, iters_lev1 = args.iters_lev1, test_mode=True)
        mace = torch.sum((four_pr[0, :, :, :].cpu() - flow_4cor) ** 2, dim=0).sqrt()
        mace_list.append(mace.view(-1).numpy())
        torch.cuda.empty_cache()
        if i_batch>300:
            break

    model.train()
    mace = np.mean(np.concatenate(mace_list))
    print("Validation MACE: %f" % mace)
    return {'chairs_mace': mace}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', default=True, help='use mixed precision')
    parser.add_argument('--model_name')

    # Ablations
    parser.add_argument('--replace', default=False, action='store_true',
                        help='Replace local motion feature with aggregated motion features')
    parser.add_argument('--no_alpha', default=False, action='store_true',
                        help='Remove learned alpha, set it to 1')
    parser.add_argument('--no_residual', default=False, action='store_true',
                        help='Remove residual connection. Do not add local features with the aggregated features.')

    args = parser.parse_args()
