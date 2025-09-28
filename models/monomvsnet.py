import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.module import *
from models.depth_anything_v2.dpt import DepthAnythingV2
from models.ted import TED
from models.FMT_with_CVPE import FMT_with_pathway

class MonoMVSNet(nn.Module):
    def __init__(self, arch_mode="fpn", reg_net='reg2d', num_stage=4, fpn_base_channel=8,
                 reg_channel=8, stage_splits=[8, 8, 4, 4], depth_interals_ratio=[0.5, 0.5, 0.5, 0.5],
                 group_cor=False, group_cor_dim=[8, 8, 8, 8],
                 inverse_depth=False,
                 agg_type='ConvBnReLU3D',
                 attn_temp=2,
                 attn_fuse_d=True,
                 mono_sampling=True,
                 edge_guide=True,
                 attention=True,
                 max_h=512,
                 max_w=640,
                 ):
        super(MonoMVSNet, self).__init__()
        self.arch_mode = arch_mode
        self.num_stage = num_stage
        self.depth_interals_ratio = depth_interals_ratio
        self.group_cor = group_cor
        self.group_cor_dim = group_cor_dim
        self.inverse_depth = inverse_depth
        self.mono_sampling = mono_sampling
        if self.mono_sampling:
            self.edge_guide = edge_guide
            self.edge_thres = 0.8
        else:
            self.edge_guide = False
        self.attention = attention
        if self.attention:
            self.transformer = FMT_with_pathway()
            self.pe_num = 8
        if arch_mode == "fpn":
            self.feature = FPN4(base_channels=fpn_base_channel, gn=False)
        self.stage_splits = stage_splits
        self.reg = nn.ModuleList()
        self.stagenet = stagenet(inverse_depth, attn_fuse_d, attn_temp)
        if reg_net == 'reg3d':
            self.down_size = [3, 3, 2, 2]
        for idx in range(num_stage):
            if self.group_cor:
                in_dim = group_cor_dim[idx]
            else:
                in_dim = self.feature.out_channels[idx]
            if reg_net == 'reg2d':
                self.reg.append(reg2d(input_channel=in_dim, base_channel=reg_channel, conv_name=agg_type))
            elif reg_net == 'reg3d':
                self.reg.append(reg3d(in_channels=1, base_channels=reg_channel, down_size=self.down_size[idx]))

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        encoder = 'vits'  # or 'vits', 'vitb', 'vitg'
        model = DepthAnythingV2(**model_configs[encoder])
        model.load_state_dict(torch.load(f'pre_trained_weights/depth_anything_v2_{encoder}.pth', map_location='cpu',
                                         weights_only=True))
        self.mono = model
        self.mono.eval()
        self.max_h = max_h
        self.max_w = max_w
        for param in self.mono.pretrained.parameters():
            param.requires_grad = False
        for param in self.mono.depth_head.parameters():
            param.requires_grad = False
        self.vit_channel = model_configs[encoder]['out_channels'][-1]
        self.vit_down = nn.Sequential(nn.Conv2d(self.vit_channel, 256, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(256), nn.SiLU(),
                                      nn.Conv2d(256, 128, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(128), nn.SiLU(),
                                      nn.Conv2d(128, 64, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(64), nn.SiLU()
                                      )
        if self.edge_guide:
            edge_detection_model = TED()
            edge_detection_model.load_state_dict(torch.load('pre_trained_weights/TEED_model.pth'), strict=True)
            self.edge = edge_detection_model

    def forward(self, imgs, imgs_raw, proj_matrices, depth_values):

        ref_img = imgs[0]
        _, _, ori_h, ori_w = ref_img.shape

        resize_h, resize_w = ori_h // 14 * 14, ori_w // 14 * 14
        ref_img_resize = F.interpolate(ref_img, (resize_h, resize_w), mode="bilinear", align_corners=True)

        with torch.no_grad():

            # depth anything v2
            mono_depth, mono_intermediate_features = self.mono.infer_mono(ref_img_resize, ori_h, ori_w)
            # normalized depth
            b, _, h, w = mono_depth.shape
            mono_depth = mono_depth.flatten(1)
            mono_depth_max = torch.max(mono_depth, dim=-1, keepdim=True)[0]
            mono_depth_min = torch.min(mono_depth, dim=-1, keepdim=True)[0]
            mono_depth = (mono_depth - mono_depth_min) / (mono_depth_max - mono_depth_min)
            mono_depth = mono_depth.reshape(b, 1, h, w).contiguous()

        curr_features = (
            mono_intermediate_features[-1]
            .reshape(ref_img.shape[0], resize_h // 14, resize_w // 14, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        # resize to 1/8 resolution
        curr_features = F.interpolate(
            curr_features,
            (ori_h // 8, ori_w // 8),
            mode="bilinear",
            align_corners=True,
        )
        mono_feature = self.vit_down(curr_features)

        if self.edge_guide:
            ref_edge = self.edge(imgs_raw[0])
            ref_edge = torch.sigmoid(ref_edge)
        else:
            ref_edge = None

        ref_outputs, src_outputs = self.feature(imgs, mono_feature)

        outputs = {}
        for stage_idx in range(self.num_stage):
            ref_features_stage = ref_outputs["stage{}".format(stage_idx + 1)]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            B, C, H, W = ref_features_stage.shape

            if stage_idx == 0:
                if self.mono_sampling:
                    depth_hypo, mono_depth_scale = init_inverse_range_mono(depth_values,
                                                                                self.stage_splits[stage_idx],
                                                                                imgs[0][0].device, imgs[0][0].dtype,
                                                                                H, W, mono_depth, ref_edge,
                                                                                self.edge_thres)
                    if self.attention:
                        depth_hypo_cvpe = init_inverse_range(depth_values, self.pe_num,
                                                             imgs[0][0].device, imgs[0][0].dtype, H, W)
                        src_outputs = self.transformer(ref_features_stage, src_outputs, proj_matrices_stage,
                                                       depth_hypo_cvpe)
                else:
                    depth_hypo = init_inverse_range(depth_values, self.stage_splits[stage_idx], imgs[0][0].device,
                                                    imgs[0][0].dtype, H, W)
                    if self.attention:
                        depth_hypo_cvpe = init_inverse_range(depth_values, self.pe_num,
                                                             imgs[0][0].device, imgs[0][0].dtype, H, W)
                        src_outputs = self.transformer(ref_features_stage, src_outputs, proj_matrices_stage,
                                                       depth_hypo_cvpe)
            else:
                if self.mono_sampling:
                    depth_hypo, mono_depth_scale = schedule_inverse_range_mono(
                        outputs_stage['inverse_min_depth'].detach(),
                        outputs_stage['inverse_max_depth'].detach(),
                        self.stage_splits[stage_idx], H, W, mono_depth,
                        ref_edge, self.edge_thres, outputs_stage['depth'].detach(),
                        outputs_stage['photometric_confidence'].detach())  # B D H W
                else:
                    depth_hypo = schedule_inverse_range(outputs_stage['inverse_min_depth'].detach(),
                                                        outputs_stage['inverse_max_depth'].detach(),
                                                        self.stage_splits[stage_idx], H, W)  # B D H W

            src_features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in src_outputs]

            outputs_stage = self.stagenet(ref_features_stage, src_features_stage, proj_matrices_stage,
                                          depth_hypo=depth_hypo, regnet=self.reg[stage_idx],
                                          group_cor=self.group_cor, group_cor_dim=self.group_cor_dim[stage_idx],
                                          split_itv=self.depth_interals_ratio[stage_idx])

            outputs_stage['mono_depth'] = mono_depth

            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)

        return outputs


def dtu_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    stage_lw = kwargs.get("stage_lw", [1,1,1,1])
    threshold = [0.15, 0.15, 0.3, 0.3]
    inverse = kwargs.get("inverse_depth", False)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    stage_ce_loss = []
    stage_mono_loss = []
    range_err_ratio = []
    for stage_idx, (stage_inputs, stage_key) in enumerate([(inputs[k], k) for k in inputs.keys() if "stage" in k]):
        depth_pred = stage_inputs['depth']
        depth_reg = stage_inputs['depth_reg']
        hypo_depth = stage_inputs['hypo_depth']
        attn_weight = stage_inputs['attn_weight']
        confidence = stage_inputs['photometric_confidence']
        mono_depth = stage_inputs['mono_depth']

        mask = mask_ms[stage_key]
        mask = mask > 0.5
        conf_mask = confidence > threshold[stage_idx]
        mono_mask = torch.logical_or(mask, conf_mask)
        depth_gt = depth_gt_ms[stage_key]

        # mask range
        if inverse:
            depth_itv = (1 / hypo_depth[:, 2, :, :] - 1 / hypo_depth[:, 1, :, :]).abs()  # B H W
            mask_out_of_range = ((1 / hypo_depth - 1 / depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(
                1) == 0  # B H W
        else:
            depth_itv = (hypo_depth[:, 2, :, :] - hypo_depth[:, 1, :, :]).abs()  # B H W
            mask_out_of_range = ((hypo_depth - depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(
                1) == 0  # B H W
        range_err_ratio.append(mask_out_of_range[mask].float().mean())

        # cross-entropy loss for all stages
        this_stage_ce_loss = cross_entropy_loss(mask, hypo_depth, depth_gt, attn_weight)

        # relative consistency loss for the last stage
        if stage_idx == 3:
            mono_depth = F.interpolate(mono_depth, size=depth_reg.shape[1:], mode='bilinear', align_corners=False)
            this_stage_mono_loss = relative_consistency_loss(depth_reg[mono_mask], mono_depth.squeeze(1)[mono_mask], sample_n=512*(4**stage_idx))
            total_loss = total_loss + this_stage_ce_loss + 0.01 * this_stage_mono_loss
        else:
            this_stage_mono_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
            total_loss = total_loss + this_stage_ce_loss

        stage_ce_loss.append(this_stage_ce_loss)
        stage_mono_loss.append(0.01 * this_stage_mono_loss)

    return total_loss, stage_ce_loss, stage_mono_loss, range_err_ratio


def bld_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    stage_lw = kwargs.get("stage_lw", [1, 1, 1, 1])
    threshold = [0.15, 0.15, 0.3, 0.3]
    inverse = kwargs.get("inverse_depth", False)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    stage_ce_loss = []
    range_err_ratio = []
    for stage_idx, (stage_inputs, stage_key) in enumerate([(inputs[k], k) for k in inputs.keys() if "stage" in k]):
        depth_pred = stage_inputs['depth']
        hypo_depth = stage_inputs['hypo_depth']
        attn_weight = stage_inputs['attn_weight']

        mono_depth = stage_inputs['mono_depth']
        confidence = stage_inputs['photometric_confidence']

        mask = mask_ms[stage_key]
        mask = mask > 0.5
        conf_mask = confidence > threshold[stage_idx]
        mono_mask = torch.logical_or(mask, conf_mask)
        depth_gt = depth_gt_ms[stage_key]

        # mask range
        if inverse:
            depth_itv = (1 / hypo_depth[:, 2, :, :] - 1 / hypo_depth[:, 1, :, :]).abs()  # B H W
            mask_out_of_range = ((1 / hypo_depth - 1 / depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(
                1) == 0  # B H W
        else:
            depth_itv = (hypo_depth[:, 2, :, :] - hypo_depth[:, 1, :, :]).abs()  # B H W
            mask_out_of_range = ((hypo_depth - depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(
                1) == 0  # B H W
        range_err_ratio.append(mask_out_of_range[mask].float().mean())

        # cross-entropy
        this_stage_ce_loss = cross_entropy_loss(mask, hypo_depth, depth_gt, attn_weight)

        stage_ce_loss.append(this_stage_ce_loss)
        total_loss = total_loss + this_stage_ce_loss

    depth_interval = hypo_depth[:, 0, :, :] - hypo_depth[:, 1, :, :]

    abs_err = torch.abs(depth_gt[mask] - depth_pred[mask])
    abs_err_scaled = abs_err / (depth_interval[mask] * 192. / 128.)
    epe = abs_err_scaled.mean()
    err3 = (abs_err_scaled < 3).float().mean()
    err1 = (abs_err_scaled < 1).float().mean()

    return total_loss, stage_ce_loss, range_err_ratio, epe, err3, err1


def cross_entropy_loss(mask_true, hypo_depth, depth_gt, attn_weight):
    B, D, H, W = attn_weight.shape
    valid_pixel_num = torch.sum(mask_true, dim=[1, 2]) + 1e-6
    gt_index_image = torch.argmin(torch.abs(hypo_depth - depth_gt.unsqueeze(1)), dim=1)
    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1)  # B, 1, H, W
    gt_index_volume = torch.zeros(B, D, H, W).type(mask_true.type()).scatter_(1, gt_index_image, 1)
    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(attn_weight + 1e-6), dim=1).squeeze(1)  # B, 1, H, W
    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image)
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])
    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num)

    return masked_cross_entropy

def relative_consistency_loss(mv_depth, mono_depth, sample_n=512):
    mv_depth, mono_depth = mv_depth.view(-1), mono_depth.view(-1)
    selected_idxs = np.random.choice(mv_depth.shape[0], sample_n * 2, replace=False)
    sample_idx0 = selected_idxs[:sample_n]
    sample_idx1 = selected_idxs[sample_n:]
    mono_depth_0, mono_depth_1 = mono_depth[sample_idx0], mono_depth[sample_idx1]
    mv_depth_0, mv_depth_1 = mv_depth[sample_idx0], mv_depth[sample_idx1]

    mask = torch.where(mono_depth_0 < mono_depth_1, True, False)
    d0 = mv_depth_0 - mv_depth_1
    d1 = mv_depth_1 - mv_depth_0

    rc_loss = torch.zeros_like(mv_depth_0)
    rc_loss[mask] += d1[mask]
    rc_loss[~mask] += d0[~mask]
    return torch.mean(torch.clamp(rc_loss, min=0.0))
