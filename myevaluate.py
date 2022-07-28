import numpy as np
import os
import torch
import argparse
from network import IHN
from utils import *
import datasets_4cor_img as datasets
import scipy.io as io
import torchvision
import numpy as np
import time

setup_seed(2022)
def evaluate_SNet(model, val_dataset, batch_size=0, args = None):

    assert batch_size > 0, "batchsize > 0"

    total_mace = torch.empty(0)
    timeall=[]
    total_mace_dict={}
    for i_batch, data_blob in enumerate(val_dataset):
        img1, img2, flow_gt,  H  = [x.to(model.device) for x in data_blob]

        if i_batch==0:
            if not os.path.exists('watch'):
                os.makedirs('watch')
            save_img(torchvision.utils.make_grid((img1)),
                     './watch/' + "b1_epoch_" + str(i_batch).zfill(5) + "_finaleval_" + '.bmp')
            save_img(torchvision.utils.make_grid((img2)),
                     './watch/' + "b2_epoch_" + str(i_batch).zfill(5) + "_finaleval_" + '.bmp')

        img1 = img1.to(model.device)
        img2 = img2.to(model.device)

        time_start = time.time()
        four_pred = model(img1, img2, iters_lev0=args.iters_lev0, iters_lev1=args.iters_lev1, test_mode=True)
        time_end = time.time()
        timeall.append(time_end-time_start)
        print(time_end-time_start)

        flow_4cor = torch.zeros((four_pred.shape[0], 2, 2, 2))
        flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]
        flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]
        flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]
        flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]

        mace_ = (flow_4cor - four_pred.cpu().detach())**2
        mace_ = ((mace_[:,0,:,:] + mace_[:,1,:,:])**0.5)
        mace_vec = torch.mean(torch.mean(mace_, dim=1), dim=1)
      
        total_mace = torch.cat([total_mace,mace_vec], dim=0)
        final_mace = torch.mean(total_mace).item()
        print(mace_.mean())
        print("MACE Metric: ", final_mace)
    if not os.path.exists("res_mat"):
        os.makedirs("res_mat")
    if not os.path.exists("res_npy"):
        os.makedirs("res_npy")
    print(np.mean(np.array(timeall[1:-1])))
    io.savemat('res_mat/' + args.savemat, {'matrix': total_mace.numpy()})
    np.save('res_npy/' + args.savedict, total_mace_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='results/IHN/IHN.pth',help="restore checkpoint")
    parser.add_argument('--iters_lev0', type=int, default=6)
    parser.add_argument('--iters_lev1', type=int, default=3)
    parser.add_argument('--mixed_precision', default=False, action='store_true',
                        help='use mixed precision')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gpuid', type=int, nargs='+', default=[0])
    parser.add_argument('--savemat', type=str,  default='resmat')
    parser.add_argument('--savedict', type=str, default='resnpy')
    parser.add_argument('--dataset', type=str, default='mscoco', help='dataset')    
    parser.add_argument('--lev0', default=False, action='store_true',
                        help='warp no')
    parser.add_argument('--lev1', default=False, action='store_true',
                        help='warp once')
    parser.add_argument('--weight', default=False, action='store_true',
                        help='weight')
    parser.add_argument('--model_name_lev0', default='', help='specify model0 name')
    parser.add_argument('--model_name_lev1', default='', help='specify model0 name')

    args = parser.parse_args()
    device = torch.device('cuda:'+ str(args.gpuid[0]))

    model = IHN(args)
    model_med = torch.load(args.model, map_location='cuda:1')
    model.load_state_dict(model_med)

    model.to(device) 
    model.eval()

    batchsz = 1

    if args.dataset=='ggearth' or args.dataset=='ggmap':
        import dataset as datasets

    args.batch_size = batchsz
    val_dataset = datasets.fetch_dataloader(args, split='val')
    evaluate_SNet(model, val_dataset, batch_size=batchsz, args=args)