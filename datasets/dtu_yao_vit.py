from torch.utils.data import Dataset
import numpy as np
import os, cv2, time, math
from PIL import Image
from datasets.data_io import *
from torchvision import transforms
from copy import deepcopy
from .color_jittor import ColorJitter
import torch

class RandomGamma():
    def __init__(self, min_gamma=0.7, max_gamma=1.5, clip_image=False):
        self._min_gamma = min_gamma
        self._max_gamma = max_gamma
        self._clip_image = clip_image

    @staticmethod
    def get_params(min_gamma, max_gamma):
        return np.random.uniform(min_gamma, max_gamma)

    @staticmethod
    def adjust_gamma(image, gamma, clip_image):
        adjusted = torch.pow(image, gamma)
        if clip_image:
            adjusted.clamp_(0.0, 1.0)
        return adjusted

    def __call__(self, img, gamma):
        # gamma = self.get_params(self._min_gamma, self._max_gamma)
        return self.adjust_gamma(img, gamma, self._clip_image)

# the DTU dataset preprocessed by Yao Yao (only for training)
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, interval_scale=1.06, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = 192  # Hardcode
        self.interval_scale = interval_scale
        self.kwargs = kwargs
        self.rt = kwargs.get("rt", False)
        self.use_raw_train = kwargs.get("use_raw_train", False)
        self.augment = kwargs.get("augment", False)
        self.color_augment = transforms.ColorJitter(brightness=0.5, contrast=0.5)

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

        if self.mode == 'train':
            self.color_jittor = ColorJitter(brightness=0.2, contrast=0.1,
                                            saturation=0.1, hue=0.05)
            self.to_tensor = transforms.ToTensor()
            self.random_gamma = RandomGamma(min_gamma=0.9, max_gamma=1.1,
                                            clip_image=True)
            self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = "Cameras/pair.txt"
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    for light_idx in range(7):
                        metas.append((scan, light_idx, ref_view, src_views))
        # print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img2(self, filename):
        img = Image.open(filename)
        # if self.mode == 'train':
        #     img = self.color_augment(img)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img

    def read_img(self, filename):
        img = Image.open(filename).convert('RGB')
        img = np.array(img)

        return img

    def get_image(self, filename):
        try:
            im = Image.open(filename)
            # print(image_path)
            return np.array(im)
        except OSError:
            raise

    def crop_img(self, img):
        raw_h, raw_w = img.shape[:2]
        start_h = (raw_h-1024)//2
        start_w = (raw_w-1280)//2
        return img[start_h:start_h+1024, start_w:start_w+1280, :]  # 1024, 1280, C

    def prepare_img(self, hr_img):
        h, w = hr_img.shape
        if not self.use_raw_train:
            #w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128
            #downsample
            hr_img_ds = cv2.resize(hr_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
            h, w = hr_img_ds.shape
            target_h, target_w = 512, 640
            start_h, start_w = (h - target_h)//2, (w - target_w)//2
            hr_img_crop = hr_img_ds[start_h: start_h + target_h, start_w: start_w + target_w]
        elif self.use_raw_train:
            hr_img_crop = hr_img[h//2-1024//2:h//2+1024//2, w//2-1280//2:w//2+1280//2]  # 1024, 1280, c
        return hr_img_crop

    def read_mask_hr(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img > 10).astype(np.float32)
        np_img = self.prepare_img(np_img)

        h, w = np_img.shape
        np_img_ms = {
            "stage1": cv2.resize(np_img, (w//8, h//8), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(np_img, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
            "stage3": cv2.resize(np_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
            "stage4": np_img,
        }
        return np_img_ms


    def read_depth_hr(self, filename, scale):
        # read pfm depth file
        #w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128
        depth_hr = np.array(read_pfm(filename)[0], dtype=np.float32) * scale
        depth_lr = self.prepare_img(depth_hr)

        h, w = depth_lr.shape
        depth_lr_ms = {
            "stage1": cv2.resize(depth_lr, (w//8, h//8), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth_lr, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
            "stage3": cv2.resize(depth_lr, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
            "stage4": depth_lr,
        }
        return depth_lr_ms
    
        
    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views

        if self.mode == 'train' and self.rt:
            num_src_views = len(src_views)
            index = random.sample(range(num_src_views), self.nviews - 1)
            view_ids = [ref_view] + [src_views[i] for i in index]
            scale = random.uniform(0.8, 1.25)
        else:
            view_ids = [ref_view] + src_views[:self.nviews - 1]
            scale = 1
        imgs = []
        imgs_raw = []
        mask = None
        depth_values = None
        proj_matrices = []

        if self.mode == 'train':
            fn_idx = torch.randperm(4)
            brightness_factor = torch.tensor(1.0).uniform_(self.color_jittor.brightness[0], self.color_jittor.brightness[1]).item()
            contrast_factor = torch.tensor(1.0).uniform_(self.color_jittor.contrast[0], self.color_jittor.contrast[1]).item()
            saturation_factor = torch.tensor(1.0).uniform_(self.color_jittor.saturation[0], self.color_jittor.saturation[1]).item()
            hue_factor = torch.tensor(1.0).uniform_(self.color_jittor.hue[0], self.color_jittor.hue[1]).item()
            gamma_factor = self.random_gamma.get_params(self.random_gamma._min_gamma, self.random_gamma._max_gamma)
        else:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor, gamma_factor = None, None, None, None, None, None

        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            if not self.use_raw_train:
                img_filename = os.path.join(self.datapath, 'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
            else:
                img_filename = os.path.join(self.datapath, 'Rectified_raw/{}/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
            mask_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_visual_{:0>4}.png'.format(scan, vid))
            depth_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, 'Cameras/train/{:0>8}_cam.txt').format(vid)

            img = self.read_img(img_filename)

            # img_raw = self.get_image(img_filename)
            if self.use_raw_train:
                img = self.crop_img(img)
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)
            if self.rt:
                extrinsics[:3,3] *= scale
            if self.use_raw_train:
                intrinsics[:2, :] *= 2.0

            if i == 0:

                mask_read_ms = self.read_mask_hr(mask_filename_hr)
                depth_ms = self.read_depth_hr(depth_filename_hr, scale)
                #get depth values
                depth_max = depth_interval * self.ndepths + depth_min
                depth_values = np.array([depth_min * scale, depth_max * scale], dtype=np.float32)
                mask = mask_read_ms

            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)

            img = Image.fromarray(img)
            imgs_raw.append(np.array(img).transpose(2,0,1))
            if not self.mode == 'train':
                img = self.transforms(img)
            else:
                img = self.color_jittor(img, fn_idx, brightness_factor, contrast_factor, saturation_factor,
                                        hue_factor)
                img = self.to_tensor(img)
                img = self.random_gamma(img, gamma_factor)
                img = self.normalize(img)
            imgs.append(img)

        # all
        # imgs = torch.stack(imgs)
        # ms proj_mats
        proj_matrices = np.stack(proj_matrices)
        stage1_pjmats = proj_matrices.copy()
        stage1_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 2.0
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
        stage4_pjmats = proj_matrices.copy()
        stage4_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4

          
        proj_matrices_ms = {
            "stage1": stage1_pjmats,
            "stage2": proj_matrices,
            "stage3": stage3_pjmats,
            "stage4": stage4_pjmats
        }

        return {"imgs": imgs,  # Nv C H W
                "imgs_raw": imgs_raw,  # Nv C H W
                "proj_matrices": proj_matrices_ms,  # 4 stage of Nv 2 4 4
                "depth": depth_ms,
                "depth_values": depth_values,
                "mask": mask}