import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc1 = nn.Conv3d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False)
    
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        # return torch.sigmoid(out) * x
        return torch.sigmoid(out) * x + x


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Along the channel dimension
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Along the channel dimension
        attention_map = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.conv(attention_map)
        # return torch.sigmoid(attention_map) * x
        return torch.sigmoid(attention_map) * x + x  # return torch.sigmoid(attention_map) * x + x


class CBAM3D(nn.Module):
    def __init__(self, in_channels, reduction=8, kernel_size=3):
        super(CBAM3D, self).__init__()
        self.channel_attention = ChannelAttention3D(in_channels, reduction=reduction)
        self.spatial_attention = SpatialAttention3D(kernel_size=kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class ConvBlock3D_att(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock3D_att, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.cbam = CBAM3D(out_channels)  # Apply CBAM attention module

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.cbam(x)  # Apply attention
        return x

class ConvBlock3D_db4_att(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock3D_db4_att, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels // 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.cbam = CBAM3D(out_channels // 2)  # Apply CBAM attention module

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.cbam(x)  # Apply attention
        return x



class ConvBlock3D_db4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock3D_db4, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels // 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels // 2)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

# class PreprocessConvBlock3D(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(PreprocessConvBlock3D, self).__init__()
#         self.conv1 =  nn.Conv3d(in_channels, in_channels, kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0)
#         self.bn1 = nn.BatchNorm3d(in_channels)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0)
#         self.bn2 = nn.BatchNorm3d(in_channels)
#         self.relu2 = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
#         return x


class Down_channel_3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down_channel_3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CBAMConcatFusion3D(nn.Module):
    def __init__(self, in_channels1, in_channels2, in_channels3, reduction=16, kernel_size=7):
        super(CBAMConcatFusion3D, self).__init__()
        # 计算拼接后的通道数
        combined_channels = in_channels1 + in_channels2 + in_channels3
        
        # 定义 Channel Attention 和 Spatial Attention (3D版本)
        self.channel_attention = ChannelAttention3D(combined_channels, reduction)
        self.spatial_attention = SpatialAttention3D(kernel_size)
        
        # 1x1x1 卷积来调整输出通道数
        self.conv1x1 = nn.Conv3d(combined_channels, in_channels2, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(in_channels2)
        self.relu1 = nn.ReLU(inplace=True)
    def forward(self, x1, x2, x3):
        # 如果输入特征的空间尺寸不一致，先对 x2 进行插值处理
        # print('x1.shape[2:]', x1.shape[2:])
        # print('x1.shape', x1.shape)
        # print('x2.shape', x2.shape)
        # print('x3.shape', x3.shape)

        if x1.shape[2:] != x2.shape[2:]:
            # 使用 trilinear 插值进行空间维度的匹配
            x2 = F.interpolate(x2, size=x1.shape[2:], mode='trilinear', align_corners=False)
            # print('x2 = F.interpolate(x2, size=x1.shape[2:]')
        if x1.shape[2:] != x3.shape[2:]:
            # 使用 trilinear 插值进行空间维度的匹配
            x3 = F.interpolate(x3, size=x1.shape[2:], mode='trilinear', align_corners=False)
            # print('x3 = F.interpolate(x3, size=x1.shape[2:]')
        # 在通道维度拼接特征

        combined = torch.cat([x1, x2, x3], dim=1)
        
        # 通过 CBAM 进行加权
        channel_weighted = self.channel_attention(combined)
        spatial_weighted = self.spatial_attention(channel_weighted)
        
        # 1x1x1 卷积调整通道数
        spatial_weighted = self.relu1(self.bn1(self.conv1x1(spatial_weighted)))

        return spatial_weighted






class EncoderDiscriminative_att_withDino_3D(nn.Module):
    def __init__(self, in_channels=100, base_width=64, dino_dim=768, min_sim_dim = 2):
        super(EncoderDiscriminative_att_withDino_3D, self).__init__()


        self.down_channel1 = Down_channel_3D(in_channels, base_width * 4)
        self.down_channel2 = Down_channel_3D(base_width * 4, base_width * 2)

        # 使用 3D 卷积块替代 2D 卷积块
        self.block1 = ConvBlock3D(base_width * 2, base_width)
        self.mp1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.block2 = ConvBlock3D(base_width, base_width * 2)
        self.mp2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.block3 = ConvBlock3D(base_width * 2, base_width * 2)
        self.mp3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.block4 = ConvBlock3D(base_width * 2, base_width * 4)
        self.mp4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.block5 = ConvBlock3D(base_width * 4, base_width * 6)
        # self.mp5 = nn.MaxPool3d(2)
        self.block6 = ConvBlock3D(base_width * 6, base_width * 8)
        
        # Dino Feature Processors
        self.dino_proc1 = DinoFeatureProcessor3D(768, base_width, target_size=(2, 64, 64))
        self.dino_proc3 = DinoFeatureProcessor3D(768, base_width * 2, target_size=(2, 16, 16))
        self.dino_proc4 = DinoFeatureProcessor3D(768, base_width * 4, target_size=(2, 8, 8))
        self.dino_proc6 = DinoFeatureProcessor3D(768, base_width * 8, target_size=(2, 4, 4))

        # Dino Feature Processors
        self.min_sim_proc1 = Min_Sim_Processor3D(min_sim_dim, base_width, target_size=(2, 64, 64))
        self.min_sim_proc3 = Min_Sim_Processor3D(min_sim_dim, base_width * 2, target_size=(2, 16, 16))
        self.min_sim_proc4 = Min_Sim_Processor3D(min_sim_dim, base_width * 4, target_size=(2, 8, 8))
        self.min_sim_proc6 = Min_Sim_Processor3D(min_sim_dim, base_width * 8, target_size=(2, 4, 4))


        # 3D版本的 CBAM Fusion Modules
        self.cbam_fusion1 = CBAMConcatFusion3D(base_width, base_width, base_width, reduction=16)
        self.cbam_fusion3 = CBAMConcatFusion3D(base_width * 2, base_width * 2, base_width * 2, reduction=16)  # Block3 和 DINO2
        self.cbam_fusion4 = CBAMConcatFusion3D(base_width * 4, base_width * 4, base_width * 4, reduction=16)  # Block4 和 DINO3
        self.cbam_fusion6 = CBAMConcatFusion3D(base_width * 8, base_width * 8, base_width * 8, reduction=16)  # Block6 和 DINO4

    def forward(self, x, dino_features, min_similarity_map):

        dino1, dino3 = dino_features
        # print('dino1.shape', dino1.shape)
        x = self.down_channel1(x)
        x = self.down_channel2(x)
        # print('x.shape', x.shape) # [4, 72, 12, 64, 64]
        # Block 1 + DINO1
        b1 = self.block1(x)
        # print('b1.shape', b1.shape) # [4, 36, 12, 64, 64]
        dino1_proc = self.dino_proc1(dino1)
        # print('dino1_proc.shape', dino1_proc.shape) # [4, 36, 12, 64, 64]
        min1_sim_proc = self.min_sim_proc1(min_similarity_map)
        # print('min1_sim_proc.shape', min1_sim_proc.shape) # [4, 36, 12, 64, 64]
        b1 = self.cbam_fusion1(b1, dino1_proc, min1_sim_proc)  # Block1 和 DINO1 融合
        # print('b1.shape', b1.shape) # [4, 36, 12, 64, 64]
        mp1 = self.mp1(b1)
        # print('mp1.shape', mp1.shape) # [4, 36, 12, 32, 32]
        # print_memory_usage()
        # Block 2 (没有DINO)


        b2 = self.block2(mp1)
        # print('b2.shape', b2.shape) # [4, 72, 12, 32, 32]
        mp2 = self.mp2(b2)
        # print('mp2.shape', mp2.shape) # [4, 72, 12, 16, 16]

        # Block 3 + DINO2
        b3 = self.block3(mp2)
        # print('b3.shape', b3.shape) # [4, 72, 12, 16, 16]
        dino3_proc = self.dino_proc3(dino1)
        # print('dino3_proc.shape', dino3_proc.shape) # [4, 72, 12, 16, 16]
        min3_sim_proc = self.min_sim_proc3(min_similarity_map)
        # print('min3_sim_proc.shape', min3_sim_proc.shape) # [4, 72, 12, 16, 16]
        b3 = self.cbam_fusion3(b3, dino3_proc, min3_sim_proc)  # Block3 和 DINO2 融合
        # print('b3.shape', b3.shape) # [4, 72, 12, 16, 16]
        mp3 = self.mp3(b3)
        
        # Block 4 + DINO3
        b4 = self.block4(mp3)
        # print('b4.shape', b4.shape) # [4, 144, 12, 8, 8]
        dino4_proc = self.dino_proc4(dino3)
        # print('dino4_proc.shape', dino4_proc.shape) # [4, 144, 3, 8, 8]
        min4_sim_proc = self.min_sim_proc4(min_similarity_map)
        # print('min4_sim_proc.shape', min4_sim_proc.shape) # [4, 144, 3, 8, 8]
        b4 = self.cbam_fusion4(b4, dino4_proc, min4_sim_proc)  # Block4 和 DINO3 融合
        mp4 = self.mp4(b4)
        # print('mp4.shape', mp4.shape) # [4, 144, 12, 4, 4]
        # print_memory_usage()
        # Block 5 (没有DINO)
        b5 = self.block5(mp4)
        # mp5 = self.mp5(b5)
        # print('mp5.shape', mp5.shape) # [4, 216, 6, 2, 2]
        # Block 6 + DINO4
        # print('b5.shape', b5.shape) # [4, 216, 12, 4, 4]
        b6 = self.block6(b5)
        # print('b6.shape', b6.shape) # [4, 216, 12, 4, 4]
        dino6_proc = self.dino_proc6(dino3)
        # print('dino6_proc.shape', dino6_proc.shape) # [4, 288, 3, 4, 4]
        min6_sim_proc = self.min_sim_proc6(min_similarity_map)
        # print('min6_sim_proc.shape', min6_sim_proc.shape) # [4, 288, 3, 4, 4]
    
        b6 = self.cbam_fusion6(b6, dino6_proc, min6_sim_proc)  # Block6 和 DINO4 融合
        # print('b6.shape', b6.shape) # [4, 288, 12, 4, 4]

        return b1, b2, b3, b4, b5, b6



class UpSampleBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, change_depth_channel=True):
        super(UpSampleBlock3D, self).__init__()
        # 使用 3D 上采样
        if change_depth_channel:
            self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)  # 'trilinear' 更适合 3D 数据
        else:
            self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)  # 'trilinear' 更适合 3D 数据

        # 使用 3D 卷积
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        # 使用 3D 批归一化
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 先进行上采样
        x = self.upsample(x)
        # 然后进行卷积，批归一化，ReLU 激活
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x




class DinoFeatureProcessor3D(nn.Module):
    """
    处理 DINO 的特征，将其转换为适配 UNet 3D 编码器的特征
    输入形状: (batch_size, 1024, 768) -> 输出形状: (batch_size, output_dim, depth, height, width)
    """
    def __init__(self, input_dim=768, output_dim=64, target_size=(4, 64, 64)):
        super(DinoFeatureProcessor3D, self).__init__()
        # self.fc = nn.Linear(input_dim, output_dim)  # 压缩通道数
        self.target_size = target_size  # UNet 目标尺寸 (depth, height, width)
        # self.patch_dim = patch_dim  # DINO 的 patch grid size
        self.conv3d = nn.Conv3d(input_dim, output_dim, kernel_size=(1, 1, 1))  # 压缩通道数
        self.bn = nn.BatchNorm3d(output_dim)  # 添加 BatchNorm3d
        self.relu = nn.ReLU(inplace=True)    # 添加 ReLU
        # 通过卷积扩展深度信息
        self.depth_conv = nn.Conv3d(output_dim, output_dim, kernel_size=(3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(output_dim)  # 添加 BatchNorm3d
        self.relu2 = nn.ReLU(inplace=True)    # 添加 ReLU

    def forward(self, x):
        """
        :param x: 输入形状为 (batch_size, num_patches, channels)，通常为 (batch_size, 1024, 768)
        :return: 输出形状为 (batch_size, output_dim, depth, height, width)
        """
        batch_size, num_patches, channels = x.shape
        patch_size = int(num_patches ** 0.5)  # DINO 的 patch size
        
        # 压缩特征通道
        # x = self.fc(x)  # (batch_size, 1024, output_dim)
        
        # 将 DINO 特征 reshape 为 3D 格式
        x = x.view(batch_size, patch_size, patch_size, channels)  # (batch_size, 32, 32, 768)
        x = x.permute(0, 3, 1, 2)  # 转换为 (batch_size, channels, 32, 32)
        x = x.unsqueeze(2)  # (batch_size, channels, 1, 32, 32)
        x = self.relu(self.bn(self.conv3d(x)))  # (batch_size, output_dim, 1, 32, 32)

        # 使用插值调整空间尺寸 (height, width)
        x = F.interpolate(x, size=(self.target_size[0], self.target_size[1], self.target_size[2]), 
                          mode='trilinear', align_corners=False)

        # 使用 3D 卷积进行深度信息学习
        x = self.relu2(self.bn2(self.depth_conv(x)))  # (batch_size, output_dim, depth, height, width)
        
        return x  # 输出形状为 (batch_size, output_dim, depth, height, width)



class Min_Sim_Processor3D(nn.Module):
    """
    处理 min_similarity_map，将其转换为适配 UNet 3D 编码器的特征
    输入形状: (batch_size, 4, 32, 32) -> 输出形状: (batch_size, base_width, depth, height, width)
    """
    def __init__(self, input_dim=4, output_dim=64, target_size=(4, 64, 64)):
        super(Min_Sim_Processor3D, self).__init__()
        self.target_size = target_size  # UNet 目标尺寸 (depth, height, width)
        
        # 升维并压缩通道数
        self.conv3d = nn.Conv3d(input_dim, output_dim, kernel_size=(1, 1, 1))  # 仅作用于通道维度
        self.bn = nn.BatchNorm3d(output_dim)  # 添加 BatchNorm3d
        self.relu = nn.ReLU(inplace=True)    # 添加 ReLU
        # 用于深度信息学习的卷积
        self.depth_conv = nn.Conv3d(output_dim, output_dim, kernel_size=(3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(output_dim)  # 添加 BatchNorm3d
        self.relu2 = nn.ReLU(inplace=True)    # 添加 ReLU

    def forward(self, x):
        """
        :param x: 输入形状为 (batch_size, 12, 32, 32)
        :return: 输出形状为 (batch_size, output_dim, depth, height, width)
        """
        batch_size, channels, height, width = x.shape

        # 增加 depth 维度
        x = x.unsqueeze(2)  # (batch_size, 12, 1, 32, 32)

        # 压缩/扩展通道数
        x = self.relu(self.bn(self.conv3d(x)))  # (batch_size, output_dim, 1, 32, 32)

        # 调整到目标尺寸 (depth, height, width)
        x = F.interpolate(x, size=(self.target_size[0], self.target_size[1], self.target_size[2]), 
                          mode='trilinear', align_corners=False)

        # 使用 3D 卷积学习深度信息
        x = self.relu2(self.bn2(self.depth_conv(x)))  # (batch_size, output_dim, depth, height, width)

        return x  # 输出形状为 (batch_size, output_dim, depth, height, width)


class DecoderDiscriminative_att_3D(nn.Module):
    def __init__(self, base_width, out_channels=2, num_classes=2, cls_classes=15, use_dropout=True):
        super(DecoderDiscriminative_att_3D, self).__init__()
        
        # 3D 上采样和卷积块
        # self.up_b = UpSampleBlock3D(base_width * 8, base_width * 8)
        self.use_dropout = use_dropout
        self.db_b = ConvBlock3D(base_width * 14, base_width * 8)
        
        self.up1 = UpSampleBlock3D(base_width * 8, base_width * 4, change_depth_channel=False)
        # self.db1 = ConvBlock3D_att(base_width * 8, base_width * 4)  # With attention
        self.db1 = nn.Sequential(
            ConvBlock3D_att(base_width * 8, base_width * 4),
            nn.Dropout3d(p=0.2) if use_dropout else nn.Identity()
        )
        self.up2 = UpSampleBlock3D(base_width * 4, base_width * 2, change_depth_channel=False)
        # self.db2 = ConvBlock3D(base_width * 4, base_width * 2)
        self.db2 = nn.Sequential(
            ConvBlock3D(base_width * 4, base_width * 2),
            nn.Dropout3d(p=0.2) if use_dropout else nn.Identity()
        )
        self.up3 = UpSampleBlock3D(base_width * 2, base_width, change_depth_channel=False)
        # self.db3 = ConvBlock3D_att(base_width * 3, base_width)  # With attention
        self.db3 = nn.Sequential(
            ConvBlock3D_att(base_width * 3, base_width),
            nn.Dropout3d(p=0.2) if use_dropout else nn.Identity()
        )
        self.up4 = UpSampleBlock3D(base_width, base_width, change_depth_channel=False)
        # self.db4 = ConvBlock3D_db4_att(base_width * 2, base_width * 2)  # With attention
        self.db4 = nn.Sequential(
            ConvBlock3D_db4_att(base_width * 2, base_width * 2),
            nn.Dropout3d(p=0.2) if use_dropout else nn.Identity()
        )
        # 最终输出的 3D 卷积
        # self.fin_out = nn.Conv3d(base_width // 4, out_channels, kernel_size=3, padding=1)

        self.conv1 = nn.Conv3d(base_width, base_width, kernel_size=(2, 1, 1))
        self.conv2 = nn.Conv3d(base_width, 2, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.channel_attention_seg = ChannelAttention3D(base_width, reduction=4)     

        self.conv2d_1to2 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=1) 
        self.chsn_ = 144
        # 3D通道注意力
        self.channel_attention_cls = ChannelAttention3D(self.chsn_, reduction=16)  # 通道注意力，结合 cat4 和 db4 的通道      (128 + 16) / 64 * base_width
        
         # 分类头 (全局平均池化 -> 展平 -> 线性层)
        self.classifier = nn.Sequential(
            nn.Conv3d(self.chsn_, base_width, kernel_size=3, padding=1),  # 融合后再降维
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # 3D全局平均池化
            nn.Flatten(),
            nn.Dropout(p=0.5) if use_dropout else nn.Identity(),
            nn.Linear(base_width, cls_classes)
        )


    def forward(self, b1, b2, b3, b4, b5, b6):
        # 上采样并拼接
        # up_b = self.up_b(b6)
        # print('up_b.shape', up_b.shape) # [4, 288, 24, 8, 8]
        # print('b5.shape', b5.shape) # [4, 216, 12, 4, 4]
        cat_b = torch.cat((b6, b5), dim=1) # [4, 216, 12, 4, 4]
        # if torch.isnan(cat_b).any() or torch.isinf(cat_b).any():
        #     print("Input tensor contains NaN or Inf cat_b")
        db_b = self.db_b(cat_b)
        up1 = self.up1(db_b)
        # if torch.isnan(up1).any() or torch.isinf(up1).any():
        #     print("Input tensor contains NaN or Inf up1")
        # print('up1.shape', up1.shape) # [4, 144, 12, 8, 8]
        # print('b4.shape', b4.shape) # [4, 144, 12, 8, 8]
        cat1 = torch.cat((up1, b4), dim=1)
        db1 = self.db1(cat1)
        up2 = self.up2(db1)
        # if torch.isnan(up2).any() or torch.isinf(up2).any():
        #     print("Input tensor contains NaN or Inf up2")
        # print('up2.shape', up2.shape) # [4, 72, 12, 16, 16]
        # print('b3.shape', b3.shape) # [4, 72, 12, 16, 16]
        cat2 = torch.cat((up2, b3), dim=1)
        db2 = self.db2(cat2)
        up3 = self.up3(db2)
        # if torch.isnan(up3).any() or torch.isinf(up3).any():
        #     print("Input tensor contains NaN or Inf up3")
        # print('up3.shape', up3.shape) # [4, 36, 12, 32, 32]
        # print('b2.shape', b2.shape) # [4, 36, 12, 32, 32]
        cat3 = torch.cat((up3, b2), dim=1)
        db3 = self.db3(cat3)
        up4 = self.up4(db3)
        # if torch.isnan(up4).any() or torch.isinf(up4).any():
        #     print("Input tensor contains NaN or Inf up4")
        # print('up4.shape', up4.shape) # [4, 36, 12, 64, 64]
        # print('b1.shape', b1.shape) # [4, 36, 12, 64, 64]
        cat4 = torch.cat((up4, b1), dim=1)
        db4 = self.db4(cat4)
        # if torch.isnan(db4).any() or torch.isinf(db4).any():
        #     print("Input tensor contains NaN or Inf db4")
        # print('db4.shape', db4.shape) # [4, 48, 4, 64, 64]
      
        
        out = self.conv1(db4)  # 
        # print('out.shape', out.shape) # [4, 48, 1, 64, 64]
        out = self.channel_attention_seg(out)  # 
        # print('out.shape', out.shape) # [4, 48, 1, 64, 64]
        # out = self.conv2(out).squeeze()  # 
        # print('out.shape', out.shape) 
        out, _ = torch.min(out, dim=1)
        # print('out.shape', out.shape) # [4, 1, 64, 64]


        # out = torch.cat([1-out, out], dim=1)
        # print('out.shape', out.shape) # [4, 2, 64, 64]
        
        out = self.conv2d_1to2(out)
        # print('out.shape', out.shape) # [4, 2, 64, 64]

        # print('out[0,0]', out[0,0]) 
        # print('out[0,1]', out[0,1]) 

        
        # 最终输出
        # if torch.isnan(out).any() or torch.isinf(out).any():
        #     print("Input tensor contains NaN or Inf out")

        # print('cat4.shape', cat4.shape) # [4, 72, 12, 64, 64]
        # print('db4.shape', db4.shape) # [4, 9, 12, 64, 64]
        # 在通道维度拼接 cat4 和 db4
        combined = torch.cat([cat4, db4], dim=1)
        # print('combined.shape', combined.shape) # [4, 144, 4, 64, 64]
        # 应用通道注意力模块
        combined = self.channel_attention_cls(combined)
        # print('combined.shape', combined.shape) # [4, 81, 4, 64, 64]
        # 分类头
        class_out = self.classifier(combined)
        # print('class_out.shape', class_out.shape) # [4, 15]
        return out, class_out



class DiscriminativeSubNetwork_3d_att_dino_channel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=32, out_features=False):
        super(DiscriminativeSubNetwork_3d_att_dino_channel, self).__init__()
        base_width = base_channels

        # 使用 3D 版本的编码器和解码器

        self.encoder_segment = EncoderDiscriminative_att_withDino_3D(in_channels, base_width)
        

        self.decoder_segment = DecoderDiscriminative_att_3D(base_width, out_channels)
        
        self.out_features = out_features

    def forward(self, x, dino_features, min_similarity_map):
        # print('x.shape0', x.shape)
        # print_memory_usage()
        # x = self.conv_reduce_depth(x)
        # print('x.shape1', x.shape)
        # print_memory_usage()
        # 编码器阶段
        b1, b2, b3, b4, b5, b6 = self.encoder_segment(x, dino_features, min_similarity_map)
        # print_memory_usage()
        # 检查 NaN 或 Inf 值
        for i, b in enumerate([b1, b2, b3, b4, b5, b6], start=1):
            if torch.isnan(b).any() or torch.isinf(b).any():
                print(f"Input tensor contains NaN or Inf b{i}")
        
        # 解码器阶段
        output_segment, out_class = self.decoder_segment(b1, b2, b3, b4, b5, b6)
        # print_memory_usage()
        # 检查 NaN 或 Inf 值
        if torch.isnan(output_segment).any() or torch.isinf(output_segment).any():
            print("Input tensor contains NaN or Inf output_segment")
        
        # 如果需要返回中间特征，返回解码器输出和分类输出及中间层特征
        if self.out_features:
            return output_segment, out_class, b2, b3, b4, b5, b6
        else:
            return output_segment, out_class







def print_memory_usage():
    # Print the current GPU memory usage in MB
    allocated = torch.cuda.memory_allocated() / 1024 ** 2  # in MB
    reserved = torch.cuda.memory_reserved() / 1024 ** 2  # in MB
    print(f"Allocated memory: {allocated:.2f} MB")
    print(f"Reserved memory: {reserved:.2f} MB")



def count_parameters(model):
    """
    统计模型的参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 测试3D模型
if __name__ == "__main__":
    print("begin")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device)

    # 创建 3D 网络模型
    model = DiscriminativeSubNetwork_3d_att_dino_channel(in_channels=768, out_channels=2, base_channels=48).to(device)
    model.eval()  # 设置为评估模式

    # 打印模型的参数数量
    total_params = count_parameters(model)
    print(f"模型的总参数数量: {total_params}")

    print("model")
    
    # 创建 3D 输入张量 (batch_size, channels, depth, height, width)
    input_tensor = torch.randn(8, 768, 2, 64, 64).to(device)  # 示例 3D 输入：batch_size=16, channels=12, depth=64, height=256, width=256

    # 创建 DINO 特征
    dino1 = torch.randn(8, 784, 768 ).to(device)
    dino3 = torch.randn(8, 784, 768 ).to(device)
    # dino3 = torch.randn(8, 1024, 768 ).to(device)
    # dino4 = torch.randn(8, 1024, 768 ).to(device)

    dino_features = [dino1, dino3]  # DINO特征的四个部分
    min_similarity_map = torch.randn(8, 2, 64, 64).to(device)
    # 计算运行时间
    start_time = time.time()
    
    with torch.cuda.amp.autocast():  # 自动混合精度
        for i in range(100):
            output = model(input_tensor, dino_features, min_similarity_map)
            # 打印输出形状
    print("输出形状:", output[0].shape)  # 由于解码器输出会是两个部分，取第一个输出
    print("输出形状:", output[1].shape)
    end_time = time.time()

    # 计算并打印运行时间
    elapsed_time = end_time - start_time
    print(f"运行时间: {elapsed_time:.4f} 秒")
    print(f"模型的总参数数量: {total_params}")






