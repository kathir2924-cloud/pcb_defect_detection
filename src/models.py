import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv, C2f, SPPF, Concat

def autopad(k, p=None, d=1):
    if d > 1: 
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None: 
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class ConvBnAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(int(c1), int(c2), int(k), int(s), autopad(k, p, d), 
                             groups=max(int(g),1), dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(int(c2))
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class GhostConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = int(c2 // 2)
        self.cv1 = ConvBnAct(c1, c_, k, s, g=g, act=act)
        self.cv2 = ConvBnAct(c_, c_, 5, 1, p=2, g=c_, act=act)
    
    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)

class ChannelAttention(nn.Module):
    def __init__(self, c, reduction=16):
        super().__init__()
        mid = max(c // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(c, mid, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(mid, c, 1, bias=False)
        )
        self.sig = nn.Sigmoid()
    
    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        maxx = F.adaptive_max_pool2d(x, 1)
        return x * self.sig(self.mlp(avg) + self.mlp(maxx))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sig = nn.Sigmoid()
    
    def forward(self, x):
        avg = torch.mean(x, 1, keepdim=True)
        maxx = torch.max(x, 1, keepdim=True)[0]
        return x * self.sig(self.bn(self.conv(torch.cat([avg, maxx], 1))))

class CBAM(nn.Module):
    def __init__(self, c, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(c, reduction)
        self.sa = SpatialAttention(kernel_size)
    
    def forward(self, x):
        return self.sa(self.ca(x))

class GhostBottleneckLight(nn.Module):
    def __init__(self, c1, c2, shortcut=True, cbam=True, drop=0.05):
        super().__init__()
        c_ = int(c2 // 2)
        self.g1 = GhostConv(c1, c_, 1, 1)
        self.g2 = GhostConv(c_, c2, 3, 1)
        self.cbam = CBAM(c2) if cbam else nn.Identity()
        self.add = shortcut and c1 == c2
    
    def forward(self, x):
        out = self.cbam(self.g2(self.g1(x)))
        return out + x if self.add else out

class C2f_GhostCBAM(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = GhostConv(c1, 2 * self.c, 1, 1)
        self.cv2 = GhostConv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList([GhostBottleneckLight(self.c, self.c, shortcut) for _ in range(n)])
        self.cbam_out = CBAM(c2)
    
    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cbam_out(self.cv2(torch.cat(y, 1)))

class GhostSPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = int(c1 // 2)
        self.cv1 = GhostConv(c1, c_, 1, 1)
        self.cv2 = GhostConv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cbam = CBAM(c2)
    
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cbam(self.cv2(torch.cat([x, y1, y2, y3], 1)))

# Register custom modules with Ultralytics
import ultralytics.nn.modules as umod
import ultralytics.nn.tasks as tasks

custom_modules = {
    'GhostConv': GhostConv,
    'CBAM': CBAM,
    'C2f_GhostCBAM': C2f_GhostCBAM,
    'GhostSPPF': GhostSPPF,
    'GhostBottleneckLight': GhostBottleneckLight
}

for name, cls in custom_modules.items():
    setattr(umod, name, cls)
    setattr(tasks, name, cls)

print("✅ Custom Ghost + CBAM modules loaded successfully!")