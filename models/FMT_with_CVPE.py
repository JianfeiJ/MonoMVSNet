import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

'''
- We provide two different positional encoding methods as shown below.
- You can easily switch different pos-enc in the __init__() function of FMT.
- In our experiments, PositionEncodingSuperGule usually cost more GPU memory.
'''

def homo_warping_pe(feature, ref_proj, src_proj, depth_values, fea='ref'):
    # ref_fea: [B, C, H, W]
    # ref_proj: [B, 4, 4]
    # src_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] or [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    C = feature.shape[1]
    Hs, Ws = feature.shape[-2:]
    B, num_depth, Hr, Wr = depth_values.shape

    with torch.no_grad():

        if fea == 'ref':
            proj = torch.matmul(ref_proj, torch.inverse(src_proj))
        elif fea == 'src':
            proj = torch.matmul(src_proj, torch.inverse(ref_proj))

        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        # Generate pixel grid for source image (Hs, Ws)
        y, x = torch.meshgrid([torch.arange(0, Hr, dtype=torch.float32, device=feature.device),
                               torch.arange(0, Wr, dtype=torch.float32, device=feature.device)])
        y = y.reshape(Hr * Wr)
        x = x.reshape(Hr * Wr)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]

        # Project from reference to source
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.reshape(B, 1, num_depth,
                                                                                               -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.reshape(B, 3, 1, 1)  # [B, 3, Ndepth, H*W]

        # Handle divide-by-zero cases
        temp = proj_xyz[:, 2:3, :, :]
        temp[temp == 0] = 1e-9
        proj_xy = proj_xyz[:, :2, :, :] / temp  # [B, 2, Ndepth, H*W]

        # Normalize the projected coordinates
        proj_x_normalized = proj_xy[:, 0, :, :] / ((Ws - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((Hs - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    if len(feature.shape) == 4:
        # Warp the reference features to the source view
        warped_feature = F.grid_sample(feature, grid.reshape(B, num_depth * Hr, Wr, 2), mode='bilinear',
                                       padding_mode='zeros', align_corners=True)
        warped_feature = warped_feature.reshape(B, C, num_depth, Hr, Wr)
    elif len(feature.shape) == 5:
        # Handle the case where the input is 5D
        warped_feature = []
        for d in range(feature.shape[2]):
            warped_feature.append(
                F.grid_sample(feature[:, :, d], grid.reshape(B, num_depth, Hr, Wr, 2)[:, d], mode='bilinear',
                              padding_mode='zeros', align_corners=True))
        warped_feature = torch.stack(warped_feature, dim=2)

    return warped_feature



class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class cam_param_encoder(nn.Module):
    def __init__(self, in_channels, mid_channels, embed_dims):
        super(cam_param_encoder, self).__init__()
        self.embed_dims = embed_dims
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.context_ch = self.embed_dims
        self.cam_param_len = 16

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.context_conv = nn.Conv2d(mid_channels,
                                      self.context_ch,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0
                                      )
        self.bn = nn.BatchNorm1d(self.cam_param_len)

        self.context_mlp = Mlp(self.cam_param_len, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)

    def forward(self, feat, cam_params):
        '''
        Input:
            feat: shape (B, C, H, W)
        Output:
            context: (B, C, H, W)
        '''
        B, C, H, W = feat.shape
        cam_params = cam_params.view(B, -1)  # Left shape: (B, 16)

        mlp_input = self.bn(cam_params)  # mlp_input shape: (B, 16)
        # mlp_input = self.ln(cam_params)  # mlp_input shape: (B, 16)

        feat = self.reduce_conv(feat)  # feat shape: (B, mid_ch, H, W)

        context_se = self.context_mlp(mlp_input)[..., None, None]  # context_se shape: (B, mid_ch, 1, 1)
        context = self.context_se(feat, context_se)
        context = self.context_conv(context)  # context_se shape: (B, context_ch, H, W)

        context = context.view(B, context.shape[-3], context.shape[-2], context.shape[-1])

        return context

class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(600, 600), temp_bug_fix=True):
        """
        Args:
            max_shape (tuple): for 1/4 featmap, the max length of 600 corresponds to 2400 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        if temp_bug_fix:
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
        else:  # a buggy implementation (for backward compatability only)
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / d_model//2))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]
        # self.register_buffer('pe11', pe.unsqueeze(0))  # [1, C, H, W]
        # self.mlp = Mlp(d_model, d_model)

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super(LinearAttention, self).__init__()
        self.feature_map = lambda x: torch.nn.functional.elu(x) + 1
        self.eps = eps

    def forward(self, queries, keys, values):
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        KV = torch.einsum("nshd,nshm->nhmd", K, values)

        # Compute the normalizer
        Z = 1/(torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1))+self.eps)

        # Finally compute and return the new values
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)

        return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        # Extract the dimensions into local variables
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(N, L, H, -1)
        keys = self.key_projection(keys).view(N, S, H, -1)
        values = self.value_projection(values).view(N, S, H, -1)

        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
        ).view(N, L, -1)

        # Project the output and return
        return self.out_projection(new_values)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, d_ff=None, dropout=0.0,
                 activation="relu"):
        super(EncoderLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        inner_attention = LinearAttention()
        attention = AttentionLayer(inner_attention, d_model, n_heads, d_keys, d_values)

        d_ff = d_ff or 2 * d_model
        self.attention = attention
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, x, source):
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]

        # Run self attention and add it to the input
        x = x + self.dropout(self.attention(
            x, source, source,
        ))

        # Run the fully connected part of the layer
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x+y)


class FMT_with_CVPE(nn.Module):
    def __init__(self, config):
        super(FMT_with_CVPE, self).__init__()

        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = EncoderLayer(config['d_model'], config['nhead'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()
        self.pos_encoding = PositionEncodingSine(config['d_model'])
        self.cam_encode = cam_param_encoder(in_channels=config['depth_channel'], mid_channels=config['d_model'], embed_dims=config['d_model'])

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, ref_feature=None, src_feature=None, ref_img2world=None, src_img2world=None,ref_proj=None, src_proj=None, depth_hypo=None):
        """
        Args:
            ref_feature(torch.Tensor): [B, C, H, W]
            src_feature(torch.Tensor): [B, C, H, W]
            ref_proj(torch.Tensor): [B, 4, 4]
            src_proj(torch.Tensor): [B, 2, 4, 4]
        """

        assert ref_feature is not None
        assert self.d_model == ref_feature.size(1)

        B, C, H, W = ref_feature.shape
        D = depth_hypo.size(1)

        src_proj_new = src_proj[:, 0].clone()
        src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])

        # Cross_View Position Encoding
        CVPE_ref = homo_warping_pe(ref_feature, ref_proj, src_proj_new, depth_hypo, fea='ref')  # ref->src (B,C,D,H,W)
        CVPE_src = homo_warping_pe(src_feature, ref_proj, src_proj_new, depth_hypo, fea='src')  # src->ref (B,C,D,H,W)

        CVPE_ref = self.cam_encode(CVPE_ref.reshape(B, C * D, H, W), src_img2world).contiguous()
        CVPE_src = self.cam_encode(CVPE_src.reshape(B, C * D, H, W), ref_img2world).contiguous()


        del ref_proj, src_proj, depth_hypo, src_proj_new

        ref_feature = einops.rearrange(self.pos_encoding(ref_feature) + CVPE_src, 'n c h w -> n (h w) c').contiguous()
        src_feature = einops.rearrange(self.pos_encoding(src_feature) + CVPE_ref, 'n c h w -> n (h w) c').contiguous()


        for i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):

            if name == 'self':
                src_feature = layer(src_feature, src_feature)
            elif name == 'cross':
                src_feature = layer(src_feature, ref_feature)
            else:
                raise KeyError
        return einops.rearrange(src_feature, 'n (h w) c -> n c h w', h=H).contiguous()




class FMT_with_pathway(nn.Module):
    def __init__(self,
            base_channels=8,
            config={
                'd_model': 64,
                'depth_channel': 64*8,
                'nhead': 8,
                'layer_names': ['self','cross'] * 4}):

        super(FMT_with_pathway, self).__init__()

        self.FMT_with_CVPE = FMT_with_CVPE(config)

        self.dim_reduction_1 = nn.Conv2d(base_channels * 8, base_channels * 4, 1, bias=False)
        self.dim_reduction_2 = nn.Conv2d(base_channels * 4, base_channels * 2, 1, bias=False)
        self.dim_reduction_3 = nn.Conv2d(base_channels * 2, base_channels * 1, 1, bias=False)

        self.smooth_1 = nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1, bias=False)
        self.smooth_2 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1, bias=False)
        self.smooth_3 = nn.Conv2d(base_channels * 1, base_channels * 1, 3, padding=1, bias=False)

    def _upsample_add(self, x, y):
        """_upsample_add. Upsample and add two feature maps.

        :param x: top feature map to be upsampled.
        :param y: lateral feature map.
        """

        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y


    def forward(self, ref_feature, src_features, proj_matrices, depth_hypo):

        b,c,h,w = ref_feature.shape

        intrinsics = proj_matrices[:, :, 1].clone()
        intrinsics = intrinsics[:, :, :3, :3]
        intrinsics[:, :, 0, :] *= float(w)
        intrinsics[:, :, 1, :] *= float(h)
        camk = torch.eye(4).view(1, 1, 4, 4).repeat(intrinsics.shape[0], intrinsics.shape[1], 1, 1).to(intrinsics.device).float()
        camk[:, :, :3, :3] = intrinsics
        extrinsics = proj_matrices[:, :, 0].clone()
        camk = torch.inverse(camk)
        img2world = torch.matmul(extrinsics, camk)
        img2world = torch.unbind(img2world, 1)
        ref_img2world = img2world[0]
        src_img2worlds = img2world[1:]
        proj_matrices = torch.unbind(proj_matrices, 1)
        ref_proj = proj_matrices[0]
        src_projs = proj_matrices[1:]
        ref_proj_new = ref_proj[:, 0].clone()
        ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
        # attention only for src_features
        for nview_idx, (src_features_stages, src_proj, src_img2world) in enumerate(zip(src_features, src_projs, src_img2worlds)):
            src_features_stages["stage1"] = self.FMT_with_CVPE(ref_feature.clone(), src_features_stages["stage1"].clone(), ref_img2world.clone(), src_img2world.clone(), ref_proj_new, src_proj, depth_hypo)
            src_features_stages["stage2"] = self.smooth_1(self._upsample_add(self.dim_reduction_1(src_features_stages["stage1"]), src_features_stages["stage2"]))
            src_features_stages["stage3"] = self.smooth_2(self._upsample_add(self.dim_reduction_2(src_features_stages["stage2"]), src_features_stages["stage3"]))
            src_features_stages["stage4"] = self.smooth_3(self._upsample_add(self.dim_reduction_3(src_features_stages["stage3"]), src_features_stages["stage4"]))
        return src_features