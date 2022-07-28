import os, json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as data
import torchgeometry as tgm
import cv2

class GoogleEarth(Dataset):
    def __init__(self, args, split='train'):
        if split == 'train':
            self.img_name = os.listdir('/googleearth/train/train2014_input/')
            self.input_path = '/googleearth/train/train2014_input/'
            self.label_path = '/googleearth/train/train2014_label/'
            self.template_path = '/googleearth/train/train2014_template/'

        elif split == 'val':
            if args.dataset=='ggearth':
                self.img_name = os.listdir('/googleearth/val2014_input/')
                self.input_path = '/googleearth/val2014_input/'
                self.label_path = '/googleearth/val2014_label/'
                self.template_path = '/googleearth/val2014_template/'
            if args.dataset=='ggmap':
                self.img_name = os.listdir('/googlemap/val2014_input/')
                self.input_path = '/googlemap/val2014_input/'
                self.label_path = '/googlemap/val2014_label/'
                self.template_path = '/googlemap/val2014_template/'

        print(len(self.img_name))

    def __len__(self):  
        return len(self.img_name)

    def __getitem__(self, index):  

        input_img = plt.imread(self.input_path + self.img_name[index])
        template_img = plt.imread(self.template_path + self.img_name[index])

        with open(self.label_path + self.img_name[index].split('.')[0] + '_label.txt', 'r') as outfile:
            data = json.load(outfile)

        top_left = [data['location'][0]['top_left_u'], data['location'][0]['top_left_v']]
        top_right = [data['location'][1]['top_right_u'], data['location'][1]['top_right_v']]
        bottom_left = [data['location'][2]['bottom_left_u'], data['location'][2]['bottom_left_v']]
        bottom_right = [data['location'][3]['bottom_right_u'], data['location'][3]['bottom_right_v']]

        input_img, template_img = torch.tensor(input_img).permute(2, 0, 1), torch.tensor(template_img).permute(2, 0, 1)
        input_img = input_img[:,32:160,32:160]
        points_per = torch.tensor([top_left, top_right, bottom_left, bottom_right])-torch.tensor([[32,32], [159,32], [32,159], [159,159]])

        y_grid, x_grid = np.mgrid[0:input_img.shape[1], 0:input_img.shape[2]]
        point = np.vstack((x_grid.flatten(), y_grid.flatten())).transpose()
        four_point_org = torch.zeros((2, 2, 2))
        four_point_org[:, 0, 0] = torch.Tensor([0, 0])
        four_point_org[:, 0, 1] = torch.Tensor([128 - 1, 0])
        four_point_org[:, 1, 0] = torch.Tensor([0, 128 - 1])
        four_point_org[:, 1, 1] = torch.Tensor([128 - 1, 128 - 1])

        four_point_1 = torch.zeros((2, 2, 2))
        four_point_1[:, 0, 0] = points_per[0] + torch.Tensor([0, 0])
        four_point_1[:, 0, 1] = points_per[1] + torch.Tensor([128 - 1, 0])
        four_point_1[:, 1, 0] = points_per[2] + torch.Tensor([0, 128 - 1])
        four_point_1[:, 1, 1] = points_per[3] + torch.Tensor([128 - 1, 128 - 1])
        four_point_org = four_point_org.flatten(1).permute(1, 0).unsqueeze(0)
        four_point_1 = four_point_1.flatten(1).permute(1, 0).unsqueeze(0)
        H = tgm.get_perspective_transform(four_point_org, four_point_1)
        H = H.squeeze()
        point_transformed_branch1 = cv2.perspectiveTransform(np.array([point], dtype=np.float64), H.numpy()).squeeze()

        diff_branch1 = point_transformed_branch1 - np.array(point, dtype=np.float64)
        diff_x_branch1 = diff_branch1[:, 0]
        diff_y_branch1 = diff_branch1[:, 1]

        diff_x_branch1 = diff_x_branch1.reshape((input_img.shape[1], input_img.shape[2]))
        diff_y_branch1 = diff_y_branch1.reshape((input_img.shape[1], input_img.shape[2]))
        pf_patch = np.zeros((128, 128, 2))
        pf_patch[:, :, 0] = diff_x_branch1
        pf_patch[:, :, 1] = diff_y_branch1
        flow = torch.from_numpy(pf_patch).permute(2, 0, 1).float()

        H = H.squeeze()

        return template_img.float(),input_img.float(), flow, H 

def fetch_dataloader(args, split='train'):

    train_dataset = GoogleEarth(args, split)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   pin_memory=True, shuffle=True, num_workers=8, drop_last=False)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader