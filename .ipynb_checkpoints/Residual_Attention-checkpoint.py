import torch
from torch import nn

def BaseModule(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, relu = True, half = None):

    Conv_Block = [
        nn.Conv1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            bias = False),
        nn.BatchNorm1d(
            num_features = out_channels,
            eps = 1e-05,
            momentum = 0.1, 
            affine = True, 
            track_running_stats = True),
    ]
    if relu:
        Conv_Block.append(nn.ReLU())

    return nn.Sequential(*Conv_Block)


class Residual_Block(nn.Module):
    '''
    hyper parameters are same in tsai resnet
    '''
    def __init__(self, in_channels, out_channels, half = False):
        super(Residual_Block, self).__init__()

        if not half:
            self.short_cut = BaseModule(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, relu = False)
            conv_blocks = list()
            conv_blocks.append(
                BaseModule(in_channels, out_channels, kernel_size = 7, stride = 1, padding = 3, relu = True)
            )
            conv_blocks.append(
                BaseModule(out_channels, out_channels, kernel_size = 5, stride = 1, padding = 2, relu = True)
            )
            conv_blocks.append(
                BaseModule(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, relu = False)
            )

        if half:
            self.short_cut = BaseModule(in_channels, out_channels, kernel_size = 1, stride = 2, padding = 0, relu = False)
            conv_blocks = list()
            conv_blocks.append(
                BaseModule(in_channels, out_channels, kernel_size = 7, stride = 2, padding = 3, relu = True)
            )
            conv_blocks.append(
                BaseModule(out_channels, out_channels, kernel_size = 5, stride = 1, padding = 2, relu = True)
            )
            conv_blocks.append(
                BaseModule(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, relu = False)
            )
        
        self.Conv_Blocks = nn.Sequential(*conv_blocks)
        self.ReLU = torch.nn.ReLU()

    def forward(self, x):
        fx = self.Conv_Blocks(x)
        sx = self.short_cut(x)
        return self.ReLU(fx + sx)


class FASAM_module(nn.Module):
    def __init__(self, in_channel, out_channel, num_heads):
        super(FASAM_module, self).__init__()
        self.raw_res = Residual_Block(in_channel, out_channel // num_heads)
        self.feature_res1 = Residual_Block(in_channel, out_channel // num_heads)
        self.feature_res2 = Residual_Block(out_channel // num_heads, out_channel // num_heads)
        self.mp = torch.nn.MaxPool1d(2)
        
        self.ReLU = torch.nn.ReLU()
        
    def forward(self, x):
        raw = self.raw_res(x)

        feature = self.mp(x)
        feature = self.feature_res1(feature)
        feature = torch.nn.functional.interpolate(feature, size = x.shape[-1], mode='linear')
        feature = self.feature_res2(feature)
        
        score = torch.sigmoid(feature)
        attention = torch.mul(raw, score)
        return self.ReLU(attention + raw)


class multihead_FASAM_module(torch.nn.Module):
    def __init__(self, in_channel, out_channel, num_heads):
        super(multihead_FASAM_module, self).__init__()
        self.num_heads = num_heads
        self.heads = torch.nn.ModuleList(
            [FASAM_module(in_channel, out_channel, self.num_heads) for _ in range(self.num_heads)]
        )
        
    def forward(self, x):
        outputs = [head(x) for head in self.heads]
        
        return torch.cat(outputs, dim=1)


class ResAtt(nn.Module):
    def __init__(self, in_channels, n_classes = 1, out_channel = None, num_heads = None, n_blocks = None):
        super(ResAtt, self).__init__()


        self.MF_modules = nn.ModuleList([
            multihead_FASAM_module(in_channels, 64, 2),
            multihead_FASAM_module(64, 128, 2),
            multihead_FASAM_module(128, 256, 2),
            # multihead_FASAM_module(256, 256, 2),
        ])

        
        self.adapt = nn.AdaptiveAvgPool1d(output_size = 1)
        self.classifier = nn.Sequential(
            nn.Linear(
                in_features = 256,
                out_features = n_classes)
        )

    def forward(self, x):
            
        for i in range(len(self.MF_modules)):
            x = self.MF_modules[i](x)
        
        x = self.adapt(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.classifier(x)
        return x
