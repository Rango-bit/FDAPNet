from model.encoder_decoder import *
from model.utils import Cls_head, BasicBlock_Cls
from model.FDA_module import FDA_build
from model.DGA_module import DGA_build

class SegmentEncoder(nn.Module):
    def __init__(self, seg_classes, imgsize, n_channels=3, bilinear=True, FDA_num=2, DGA_num=2, alpha=0.):
        super().__init__()
        self.n_classes = seg_classes
        self.bilinear = bilinear
        self.scale = 4

        self.in_conv = DoubleConv(n_channels, 64 // self.scale)
        self.encoder1 = Down(64 // self.scale, 128 // self.scale)
        self.encoder2 = Down(128 // self.scale, 256 // self.scale)
        self.encoder3 = Down(256 // self.scale, 512 // self.scale)
        self.encoder4 = Down(512 // self.scale, 512 // self.scale)
        factor = 2 if bilinear else 1

        # Frequency domain attention module
        self.FDA_stage = FDA_build(512 // self.scale, 512 // self.scale, imgsize // 8, FDA_num,
                                   patch_size=1, heads=6, dim_head=128, alpha=alpha)
        self.FDA_out = nn.Conv2d(512 // self.scale, 512 // self.scale // factor, kernel_size=1, padding=0, bias=False)

        # Deformable global attention module
        self.DGA_stage = DGA_build(128 // self.scale, imgsize // 2, DGA_num=DGA_num)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x1 = self.in_conv(x)
        x2 = self.encoder1(x1)
        x_DGA = self.DGA_stage(x2)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)

        x4 = self.FDA_stage(x4)
        x4 = self.FDA_out(x4)
        return x1, x2, x_DGA, x3, x4, x5

class FDAPNet_Segment(nn.Module):
    def __init__(self, seg_classes, imgsize, n_channels=3, bilinear=True, FDA_num=2, DGA_num=2, alpha=0.):
        super(FDAPNet_Segment, self).__init__()
        self.scale = 4
        self.seg_encoder = SegmentEncoder(seg_classes, imgsize, n_channels, bilinear, FDA_num, DGA_num, alpha)
        factor = 2 if bilinear else 1
        self.decoder1 = Up(512 // self.scale // 2 * 3, 256 // self.scale)
        self.decoder2 = Up_DGA(512 // self.scale, 256 // factor // self.scale)
        self.decoder3 = Up(256 // self.scale, 128 // factor // self.scale, bilinear)
        self.decoder4 = Up(128 // self.scale, 64 // self.scale, bilinear)
        self.out_conv = OutConv(64 // self.scale, seg_classes)

    def forward(self, x):
        x1, x2, x_DGA, x3, x4, x5 = self.seg_encoder(x)
        x = self.decoder1(x5, x4)
        x = self.decoder2(x, x_DGA, x3)
        x = self.decoder3(x, x2)
        x = self.decoder4(x, x1)
        seg_out = self.out_conv(x)
        return seg_out

class FDAPNet_classify(nn.Module):
    def __init__(self, cls_classes, seg_classes, imgsize, n_channels=3, bilinear=True, FDA_num=2, DGA_num=2, alpha=0.):
        super(FDAPNet_classify, self).__init__()
        self.scale = 4
        self.seg_encoder = SegmentEncoder(seg_classes, imgsize, n_channels, bilinear, FDA_num, DGA_num, alpha)

        self.cls_stage1 = Cls_head(n_channels, 128//self.scale)
        self.cls_stage2 = BasicBlock_Cls(256//self.scale, 256//self.scale, stride=2)
        self.cls_stage3 = BasicBlock_Cls(512//self.scale, 512// self.scale, stride=2)
        self.cls_stage4 = BasicBlock_Cls(128// self.scale*7, 128// self.scale*8, stride=2)
        self.cls_stage5 = BasicBlock_Cls(128// self.scale*12, 128// self.scale*12, stride=2)

        self.cls_linear = nn.Linear(128// self.scale*12, cls_classes)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x1, x2, x_DGA, x3, x4, x5 = self.seg_encoder(x)

        # Classification module
        y2 = self.cls_stage1(x)
        y3 = self.cls_stage2(torch.cat([y2, x2], dim=1))
        y4 = self.cls_stage3(torch.cat([y3, x3], dim=1))
        y5 = self.cls_stage4(torch.cat([y4, x4, x_DGA], dim=1))
        y6 = self.cls_stage5(torch.cat([y5, x5], dim=1))
        cls_out = F.avg_pool2d(y6, 4)
        cls_out = cls_out.view(cls_out.size(0), -1)
        cls_out = self.cls_linear(cls_out)
        return cls_out

def FDAPNet_Segment_build(seg_classes=2, seg_imgsize=224, **kwargs):
    seg_model = FDAPNet_Segment(seg_classes, seg_imgsize, **kwargs)
    return seg_model

def FDAPNet_Classify_build(cls_classes=2, seg_classes=2, cls_imgsize=224, seg_params_path: str='',  **kwargs):
    cls_model = FDAPNet_classify(cls_classes, seg_classes, cls_imgsize, **kwargs)
    if seg_params_path != '':
        cls_model = params_load_and_freeze(cls_model, seg_params_path)
    return cls_model

def params_load_and_freeze(model, seg_params_path): # freeze the params in the segmentation encoder
    model_params = torch.load(seg_params_path)
    missing_keys, unexpected_keys = model.load_state_dict(model_params, strict=False)
    for name, parameter in model.named_parameters():
        if name not in missing_keys:
            parameter.requires_grad = False
    return model

if __name__ == '__main__':
    # forward test
    seg_model = FDAPNet_Segment_build(seg_classes=2, seg_imgsize=224).to('cuda:0')
    cls_model = FDAPNet_Classify_build(cls_classes=2, cls_imgsize=224).to('cuda:0')
    x = torch.rand([1,1,224,224]).to('cuda:0')
    seg_output = seg_model(x)
    cls_output = cls_model(x)
    print('Segmentation output shape:', seg_output.shape)
    print('Classification output shape:', cls_output.shape)