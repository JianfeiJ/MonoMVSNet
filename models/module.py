import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import math
import numbers
from einops import rearrange
import matplotlib, cv2
import matplotlib.pyplot as plt
import numpy as np


def mono_depth_alignment(Dm, Dc, Conf, top_p=0.8):

    Dm = F.interpolate(Dm, size=Dc.shape[2:], mode='bilinear', align_corners=False)
    B, _, H, W = Conf.shape
    top_k = int(H * W * top_p)  # Get the top P% of pixels
    Ccup_flat = Conf.view(B, -1)  # Flatten to (B, H*W)
    _, indices = torch.topk(Ccup_flat, top_k, dim=-1, largest=True, sorted=False)

    M = torch.zeros_like(Conf, dtype=torch.bool)
    M.view(B, -1).scatter_(1, indices, 1)
    M = M.view(B, 1, H, W)
    Dup_c = Dc.view(B, -1)
    Dm_c = Dm.view(B, -1)
    M_flat = M.view(B, -1).float()

    def solve_least_squares(Dm_c, Dup_c, M_flat):
        A = torch.cat([Dm_c.unsqueeze(-1), torch.ones_like(Dm_c).unsqueeze(-1)], dim=-1)  # (B, H*W, 2)
        Y = Dup_c.unsqueeze(-1)  # (B, H*W, 1)
        M_flat_expanded = M_flat.unsqueeze(-1)  # Shape: (B, H*W, 1)
        A_weighted = A * M_flat_expanded
        Y_weighted = Y * M_flat_expanded

        # Solve for a and b using the weighted least squares
        a_b = torch.linalg.lstsq(A_weighted, Y_weighted).solution
        a, b = a_b[:, 0], a_b[:, 1]
        return a, b

    # Compute a and b
    a, b = solve_least_squares(Dm_c, Dup_c, M_flat)
    a = a.view(Dm.shape[0], 1, 1, 1)  # 扩展 a 的形状为 (B, 1, 1, 1)
    b = b.view(Dm.shape[0], 1, 1, 1)  # 扩展 b 的形状为 (B, 1, 1, 1)

    # Compute the final aligned depth map
    Da = b + a * Dm

    return Da

def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    C = src_fea.shape[1]
    Hs, Ws = src_fea.shape[-2:]
    B, num_depth, Hr, Wr = depth_values.shape

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, Hr, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, Wr, dtype=torch.float32, device=src_fea.device)])
        y = y.reshape(Hr * Wr)
        x = x.reshape(Hr * Wr)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.reshape(B, 1, num_depth,
                                                                                               -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.reshape(B, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        # FIXME divide 0
        temp = proj_xyz[:, 2:3, :, :]
        temp[temp == 0] = 1e-9
        proj_xy = proj_xyz[:, :2, :, :] / temp  # [B, 2, Ndepth, H*W]
        # proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]

        proj_x_normalized = proj_xy[:, 0, :, :] / ((Ws - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((Hs - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy
    if len(src_fea.shape) == 4:
        warped_src_fea = F.grid_sample(src_fea, grid.reshape(B, num_depth * Hr, Wr, 2), mode='bilinear',
                                       padding_mode='zeros', align_corners=True)
        warped_src_fea = warped_src_fea.reshape(B, C, num_depth, Hr, Wr)
    elif len(src_fea.shape) == 5:
        warped_src_fea = []
        for d in range(src_fea.shape[2]):
            warped_src_fea.append(
                F.grid_sample(src_fea[:, :, d], grid.reshape(B, num_depth, Hr, Wr, 2)[:, d], mode='bilinear',
                              padding_mode='zeros', align_corners=True))
        warped_src_fea = torch.stack(warped_src_fea, dim=2)

    return warped_src_fea


def scale_depth_init(depth, min_depth, max_depth):

    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_depth = min_disp + (max_disp - min_disp) * depth
    depth = 1 / scaled_depth
    return scaled_depth, depth


def init_inverse_range(cur_depth, ndepths, device, dtype, H, W):
    inverse_depth_min = 1. / cur_depth[:, 0]  # (B,)
    inverse_depth_max = 1. / cur_depth[:, -1]
    itv = torch.arange(0, ndepths, device=device, dtype=dtype, requires_grad=False).reshape(1, -1, 1, 1).repeat(1, 1, H,
                                                                                                                W) / (
                      ndepths - 1)  # 1 D H W
    inverse_depth_hypo = inverse_depth_max[:, None, None, None] + (inverse_depth_min - inverse_depth_max)[:, None, None,
                                                                  None] * itv

    return 1. / inverse_depth_hypo


def init_inverse_range_mono(cur_depth, ndepths, device, dtype, H, W, mono_depth, ref_edge, edge_thres):
    B, _, _, _ = mono_depth.shape

    # get inverse depth_min & depth_max
    inverse_depth_min = 1. / cur_depth[:, 0]  # (B,)
    inverse_depth_max = 1. / cur_depth[:, -1]

    # get inverse_mono
    depth_min = cur_depth[:, 0]
    depth_max = cur_depth[:, -1]
    inverse_mono_depth_scale = scale_depth_init(mono_depth, depth_min[0], depth_max[0])[0]
    inverse_mono_depth_scale = F.interpolate(inverse_mono_depth_scale, size=(H, W), mode='bilinear', align_corners=True)
    inverse_mono_depth_scale_out = inverse_mono_depth_scale
    if ref_edge is not None:
        ref_edge = F.interpolate(ref_edge, size=(H, W), mode='bilinear', align_corners=True)
        inverse_mono_depth_scale = inverse_mono_depth_scale * (ref_edge > edge_thres).float()
    inverse_mono_depth_scale = inverse_mono_depth_scale[:, 0, :, :]  # (B, H, W)

    # 将 min_depth 和 max_depth 广播到所有批次、像素位置
    inverse_min_depth = 1. / depth_min[:, None, None]  # (B, 1, 1)
    inverse_max_depth = 1. / depth_max[:, None, None]  # (B, 1, 1)

    # 生成采样点
    itv = torch.arange(0, ndepths, device=device, dtype=dtype, requires_grad=False).reshape(1, -1, 1, 1).repeat(1, 1, H,
                                                                                                                W) / (
                  ndepths - 1)  # (1, ndepths, H, W)

    inverse_depth_hypo = inverse_depth_max[:, None, None, None] + (inverse_depth_min - inverse_depth_max)[:, None, None,
                                                                  None] * itv

    # 查找哪些 inverse_mono_depth_scale 非零且在深度范围内
    valid_edge_mask = (inverse_mono_depth_scale != 0) & (inverse_mono_depth_scale >= inverse_max_depth) & (
                inverse_mono_depth_scale <= inverse_min_depth)  # (B, H, W)

    # 计算深度差并找到最接近的深度
    depth_hypo_flat = inverse_depth_hypo.view(B, ndepths, -1)  # (B, ndepths, H*W)
    mono_depth_flat = inverse_mono_depth_scale.view(B, 1, -1)  # (B, 1, H*W)

    # 计算每个像素的深度差
    diff = torch.abs(depth_hypo_flat - mono_depth_flat)  # (B, ndepths, H*W)

    # 查找最接近的深度索引
    min_diff_idx = torch.argmin(diff, dim=1)  # (B, H*W)

    # 将 idx 转换为 (B, H, W) 形状
    min_diff_idx = min_diff_idx.view(B, H, W)  # (B, H, W)

    # 通过掩码操作替换深度
    for b in range(B):
        # 获取有效位置的 mask
        valid_mask = valid_edge_mask[b]  # (H, W)

        # 替换深度：对于每个有效的像素位置，更新为最接近的 edge_depth
        inverse_depth_hypo[b, min_diff_idx[b][valid_mask], valid_mask] = inverse_mono_depth_scale[b, valid_mask]

    return 1. / inverse_depth_hypo, 1./ inverse_mono_depth_scale_out


def schedule_inverse_range(inverse_min_depth, inverse_max_depth, ndepths, H, W):
    # cur_depth_min, (B, H, W)
    # cur_depth_max: (B, H, W)
    itv = torch.arange(0, ndepths, device=inverse_min_depth.device, dtype=inverse_min_depth.dtype,
                       requires_grad=False).reshape(1, -1, 1, 1).repeat(1, 1, H // 2, W // 2) / (ndepths - 1)  # 1 D H W

    inverse_depth_hypo = inverse_max_depth[:, None, :, :] + (inverse_min_depth - inverse_max_depth)[:, None, :,
                                                            :] * itv  # B D H W
    inverse_depth_hypo = F.interpolate(inverse_depth_hypo.unsqueeze(1), [ndepths, H, W], mode='trilinear',
                                       align_corners=True).squeeze(1)

    return 1. / inverse_depth_hypo

def schedule_inverse_range_mono(inverse_min_depth, inverse_max_depth, ndepths, H, W, mono_depth, ref_edge, edge_thres,
                                coarse_depth, confidence):
    # cur_depth_min, (B, H, W)
    # cur_depth_max: (B, H, W)

    B, _, _, _ = mono_depth.shape

    itv = torch.arange(0, ndepths, device=inverse_min_depth.device, dtype=inverse_min_depth.dtype,
                       requires_grad=False).reshape(1, -1, 1, 1).repeat(1, 1, H // 2, W // 2) / (ndepths - 1)  # 1 D H W

    inverse_depth_hypo = inverse_max_depth[:, None, :, :] + (inverse_min_depth - inverse_max_depth)[:, None, :,
                                                            :] * itv  # B D H W

    inverse_mono_depth_scale = mono_depth_alignment(mono_depth, 1./coarse_depth.unsqueeze(1), confidence.unsqueeze(1), top_p=0.8)

    inverse_mono_depth_scale_out = inverse_mono_depth_scale


    if ref_edge is not None:
        ref_edge = F.interpolate(ref_edge, size=(H // 2, W // 2), mode='bilinear', align_corners=True)
        inverse_mono_depth_scale = inverse_mono_depth_scale * (ref_edge > edge_thres).float()

    inverse_mono_depth_scale = inverse_mono_depth_scale[:, 0, :, :]  # (B, H, W)

    # 将 min_depth 和 max_depth 广播到所有批次、像素位置
    min_depth = inverse_min_depth  # (B, 1, 1)
    max_depth = inverse_max_depth  # (B, 1, 1)

    # 查找符合条件的inverse_mono_depth_scale
    valid_mono_mask = (inverse_mono_depth_scale != 0) & (inverse_mono_depth_scale >= max_depth) & (
                inverse_mono_depth_scale <= min_depth)  # (B, H, W)

    # 计算深度差并找到最接近的深度
    depth_hypo_flat = inverse_depth_hypo.view(B, ndepths, -1)  # (B, ndepths, H*W)
    inverse_mono_depth_scale_flat = inverse_mono_depth_scale.view(B, 1, -1)  # (B, 1, H*W)

    # 计算每个像素的深度差
    diff = torch.abs(depth_hypo_flat - inverse_mono_depth_scale_flat)  # (B, ndepths, H*W)

    # 查找最接近的深度索引
    min_diff_idx = torch.argmin(diff, dim=1)  # (B, H*W)

    # 将 idx 转换为 (B, H, W) 形状
    min_diff_idx = min_diff_idx.view(B, H // 2, W // 2)  # (B, H, W)

    # 通过掩码操作替换深度
    for b in range(B):
        # 获取有效位置的 mask
        valid_mask = valid_mono_mask[b]  # (H, W)

        # 替换深度：对于每个有效的像素位置，更新为最接近的 inverse_mono_depth_scale
        inverse_depth_hypo[b, min_diff_idx[b][valid_mask], valid_mask] = inverse_mono_depth_scale[b, valid_mask]

    inverse_depth_hypo = F.interpolate(inverse_depth_hypo.unsqueeze(1), [ndepths, H, W], mode='trilinear',
                                       align_corners=True).squeeze(1)
    inverse_mono_depth_scale_out = F.interpolate(inverse_mono_depth_scale_out, size=(H, W), mode='bilinear', align_corners=True)

    return 1. / inverse_depth_hypo, 1./inverse_mono_depth_scale_out


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return


def depth_regression(p, depth_values):
    if depth_values.dim() <= 2:
        # print("regression dim <= 2")
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)

    return depth



class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn_momentum=0.1, init_method="xavier", gn=False, group_channel=8, **kwargs):
        super(Conv2d, self).__init__()
        bn = not gn
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.gn = nn.GroupNorm(int(max(1, out_channels / group_channel)), out_channels) if gn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        else:
            x = self.gn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)



class reg2d(nn.Module):
    def __init__(self, input_channel=128, base_channel=32, conv_name='ConvBnReLU3D'):
        super(reg2d, self).__init__()
        module = importlib.import_module("models.module")
        stride_conv_name = 'ConvBnReLU3D'
        self.conv0 = getattr(module, stride_conv_name)(input_channel, base_channel, kernel_size=(1, 3, 3),
                                                       pad=(0, 1, 1))
        self.conv1 = getattr(module, stride_conv_name)(base_channel, base_channel * 2, kernel_size=(1, 3, 3),
                                                       stride=(1, 2, 2), pad=(0, 1, 1))
        self.conv2 = getattr(module, conv_name)(base_channel * 2, base_channel * 2)

        self.conv3 = getattr(module, stride_conv_name)(base_channel * 2, base_channel * 4, kernel_size=(1, 3, 3),
                                                       stride=(1, 2, 2), pad=(0, 1, 1))
        self.conv4 = getattr(module, conv_name)(base_channel * 4, base_channel * 4)

        self.conv5 = getattr(module, stride_conv_name)(base_channel * 4, base_channel * 8, kernel_size=(1, 3, 3),
                                                       stride=(1, 2, 2), pad=(0, 1, 1))
        self.conv6 = getattr(module, conv_name)(base_channel * 8, base_channel * 8)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(base_channel * 8, base_channel * 4, kernel_size=(1, 3, 3), padding=(0, 1, 1),
                               output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(base_channel * 4),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(base_channel * 4, base_channel * 2, kernel_size=(1, 3, 3), padding=(0, 1, 1),
                               output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(base_channel * 2),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base_channel * 2, base_channel, kernel_size=(1, 3, 3), padding=(0, 1, 1),
                               output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(base_channel),
            nn.ReLU(inplace=True))

        self.prob_conv = nn.Conv3d(8, 1, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)

        x = self.prob_conv(x)

        return x.squeeze(1)


class reg3d(nn.Module):
    def __init__(self, in_channels, base_channels, down_size=3):
        super(reg3d, self).__init__()
        self.down_size = down_size
        self.conv0 = ConvBnReLU3D(in_channels, base_channels, kernel_size=3, pad=1)
        self.conv1 = ConvBnReLU3D(base_channels, base_channels * 2, kernel_size=3, stride=2, pad=1)
        self.conv2 = ConvBnReLU3D(base_channels * 2, base_channels * 2)
        if down_size >= 2:
            self.conv3 = ConvBnReLU3D(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, pad=1)
            self.conv4 = ConvBnReLU3D(base_channels * 4, base_channels * 4)
        if down_size >= 3:
            self.conv5 = ConvBnReLU3D(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, pad=1)
            self.conv6 = ConvBnReLU3D(base_channels * 8, base_channels * 8)
            self.conv7 = nn.Sequential(
                nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1, output_padding=1,
                                   stride=2, bias=False),
                nn.BatchNorm3d(base_channels * 4),
                nn.ReLU(inplace=True))
        if down_size >= 2:
            self.conv9 = nn.Sequential(
                nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1, output_padding=1,
                                   stride=2, bias=False),
                nn.BatchNorm3d(base_channels * 2),
                nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True))
        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        if self.down_size == 3:
            conv0 = self.conv0(x)
            conv2 = self.conv2(self.conv1(conv0))
            conv4 = self.conv4(self.conv3(conv2))
            x = self.conv6(self.conv5(conv4))
            x = conv4 + self.conv7(x)
            x = conv2 + self.conv9(x)
            x = conv0 + self.conv11(x)
            x = self.prob(x)
        elif self.down_size == 2:
            conv0 = self.conv0(x)
            conv2 = self.conv2(self.conv1(conv0))
            x = self.conv4(self.conv3(conv2))
            x = conv2 + self.conv9(x)
            x = conv0 + self.conv11(x)
            x = self.prob(x)
        else:
            conv0 = self.conv0(x)
            x = self.conv2(self.conv1(conv0))
            x = conv0 + self.conv11(x)
            x = self.prob(x)
        return x.squeeze(1)  # B D H W



class FPN4(nn.Module):
    """
    FPN aligncorners downsample 4x"""

    def __init__(self, base_channels, gn=False):
        super(FPN4, self).__init__()
        self.base_channels = base_channels

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels, base_channels, 3, 1, padding=1, gn=gn),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2, gn=gn),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1, gn=gn),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2, gn=gn),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1, gn=gn),
        )

        self.conv3 = nn.Sequential(
            Conv2d(base_channels * 4, base_channels * 8, 5, stride=2, padding=2, gn=gn),
            Conv2d(base_channels * 8, base_channels * 8, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels * 8, base_channels * 8, 3, 1, padding=1, gn=gn),
        )

        self.out_channels = [8 * base_channels]
        final_chs = base_channels * 8

        self.inner1 = nn.Conv2d(base_channels * 4, final_chs, 1, bias=True)
        self.inner2 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
        self.inner3 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

        self.out1 = nn.Conv2d(final_chs, base_channels * 8, 1, bias=False)
        self.out2 = nn.Conv2d(final_chs, base_channels * 4, 3, padding=1, bias=False)
        self.out3 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
        self.out4 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)

        self.out_channels.append(base_channels * 4)
        self.out_channels.append(base_channels * 2)
        self.out_channels.append(base_channels)

    def forward(self, imgs, mono_feature):

        ref_img, src_imgs = imgs[0], imgs[1:]

        ref_outputs, src_outputs = [], []


        ref_conv0 = self.conv0(ref_img)
        ref_conv1 = self.conv1(ref_conv0)
        ref_conv2 = self.conv2(ref_conv1)
        if mono_feature is not None:
            ref_conv3 = self.conv3(ref_conv2) + mono_feature
        else:
            ref_conv3 = self.conv3(ref_conv2)

        intra = ref_conv3

        ref_out1 = self.out1(intra)

        intra = F.interpolate(intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner1(ref_conv2)
        ref_out2 = self.out2(intra)

        intra = F.interpolate(intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner2(ref_conv1)
        ref_out3 = self.out3(intra)

        intra = F.interpolate(intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner3(ref_conv0)
        ref_out4 = self.out4(intra)



        ref_outputs = {}

        ref_outputs["stage1"] = ref_out1
        ref_outputs["stage2"] = ref_out2
        ref_outputs["stage3"] = ref_out3
        ref_outputs["stage4"] = ref_out4

        for src_idx, (src_img) in enumerate(src_imgs):
            src_out = {}
            src_conv0 = self.conv0(src_img)
            src_conv1 = self.conv1(src_conv0)
            src_conv2 = self.conv2(src_conv1)
            src_conv3 = self.conv3(src_conv2)
            src_intra = src_conv3.clone()

            src_out1 = self.out1(src_intra)
            src_intra = F.interpolate(src_intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner1(
                src_conv2)
            src_out2 = self.out2(src_intra)
            src_intra = F.interpolate(src_intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner2(
                src_conv1)
            src_out3 = self.out3(src_intra)
            src_intra = F.interpolate(src_intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner3(
                src_conv0)
            src_out4 = self.out4(src_intra)

            src_out["stage1"] = src_out1
            src_out["stage2"] = src_out2
            src_out["stage3"] = src_out3
            src_out["stage4"] = src_out4
            src_outputs.append(src_out)

        return ref_outputs, src_outputs


class stagenet(nn.Module):
    def __init__(self, inverse_depth=False, attn_fuse_d=True, attn_temp=2, in_channel=64, in_dim=8):
        super(stagenet, self).__init__()
        self.inverse_depth = inverse_depth
        self.attn_fuse_d = attn_fuse_d
        self.attn_temp = attn_temp

    def forward(self, ref_feature, src_features, proj_matrices, depth_hypo, regnet,
                group_cor=False,
                group_cor_dim=8, split_itv=1):

        # step 1. feature extraction
        proj_matrices = torch.unbind(proj_matrices, 1)
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        B, D, H, W = depth_hypo.shape
        C = ref_feature.shape[1]

        cor_weight_sum = 1e-8
        cor_feats = 0
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, D, 1, 1)

        ref_proj_new = ref_proj[:, 0].clone()
        ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])

        # step 2. Epipolar Transformer Aggregation
        for src_idx, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)):

            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            warped_src = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_hypo)  # B C D H W

            if group_cor:
                warped_src = warped_src.reshape(B, group_cor_dim, C // group_cor_dim, D, H, W)
                ref_volume = ref_volume.reshape(B, group_cor_dim, C // group_cor_dim, D, H, W)
                cor_feat = (warped_src * ref_volume).mean(2)  # B G D H W
            else:
                cor_feat = (ref_volume - warped_src) ** 2  # B C D H W

            del warped_src, src_proj, src_fea

            if not self.attn_fuse_d:
                cor_weight = torch.softmax(cor_feat.sum(1), 1).max(1)[0]  # B H W
                cor_weight_sum += cor_weight  # B H W
                cor_feats += cor_weight.unsqueeze(1).unsqueeze(1) * cor_feat  # B C D H W
            else:
                cor_weight = torch.softmax(cor_feat.sum(1) / self.attn_temp, 1) / math.sqrt(C)  # B D H W
                cor_weight_sum += cor_weight  # B D H W
                cor_feats += cor_weight.unsqueeze(1) * cor_feat  # B G D H W

            del cor_weight, cor_feat

        if not self.attn_fuse_d:
            cor_feats = cor_feats / cor_weight_sum.unsqueeze(1).unsqueeze(1)  # B C D H W
        else:
            cor_feats = cor_feats / cor_weight_sum.unsqueeze(1)  # B G D H W

        del cor_weight_sum, src_features

        cost_volume = regnet(cor_feats)  # B D H W

        del cor_feats

        prob_volume = F.softmax(cost_volume, dim=1)  # B D H W

        # step 4. depth argmax
        prob_max_indices = prob_volume.max(1, keepdim=True)[1]  # B 1 H W
        depth = torch.gather(depth_hypo, 1, prob_max_indices).squeeze(1)  # B H W
        depth_reg = depth_regression(prob_volume, depth_hypo)

        with torch.no_grad():
            photometric_confidence = prob_volume.max(1)[0]  # B H W
            photometric_confidence = F.interpolate(photometric_confidence.unsqueeze(1), scale_factor=1, mode='bilinear',
                                                   align_corners=True).squeeze(1)

        ret_dict = {"depth": depth,"depth_reg":depth_reg, "photometric_confidence": photometric_confidence, "hypo_depth": depth_hypo,
                    "attn_weight": prob_volume}

        if self.inverse_depth:
            last_depth_itv = 1. / depth_hypo[:, 2, :, :] - 1. / depth_hypo[:, 1, :, :]
            inverse_min_depth = 1 / depth + split_itv * last_depth_itv  # B H W
            inverse_max_depth = 1 / depth - split_itv * last_depth_itv  # B H W
            ret_dict['inverse_min_depth'] = inverse_min_depth
            ret_dict['inverse_max_depth'] = inverse_max_depth

        return ret_dict