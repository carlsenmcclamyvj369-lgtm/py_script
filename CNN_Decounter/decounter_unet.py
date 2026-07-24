"""
Decounter: U-Net based model for 960x540 RGB → 960x540 2-channel prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# =========================
# 1. U-Net
# =========================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.mpconv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffH = x2.size(2) - x1.size(2)
        diffW = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2,
                        diffH // 2, diffH - diffH // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


class DecounterUNet(nn.Module):
    """4 层 U-Net: 下采样 2 次，输入 960x540 → 瓶颈 240x135。"""
    def __init__(self, n_channels=3, n_classes=2, base_ch=64):
        super().__init__()
        self.inc = DoubleConv(n_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.up1 = Up(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.up2 = Up(base_ch * 2 + base_ch, base_ch)
        self.outc = OutConv(base_ch, n_classes)

    def forward(self, x):
        x1 = self.inc(x)          # 64,  H,   W
        x2 = self.down1(x1)       # 128, H/2, W/2
        x3 = self.down2(x2)       # 256, H/4, W/4  (bottleneck)
        x = self.up1(x3, x2)      # 128, H/2, W/2
        x = self.up2(x, x1)       # 64,  H,   W
        logits = self.outc(x)     # n_classes, H, W
        return logits


# =========================
# 2. Dataset
# =========================
class DecounterDataset(Dataset):
    """假设目录结构:
    data_dir/
        images/  xxx.png  (960x540 RGB)
        labels/  xxx.png  (960x540, 2-channel or grayscale)
    """
    def __init__(self, data_dir, split='train', transform=None):
        self.img_dir = os.path.join(data_dir, split, 'images')
        self.label_dir = os.path.join(data_dir, split, 'labels')
        self.files = sorted([f for f in os.listdir(self.img_dir)
                             if f.endswith(('.png', '.jpg', '.bmp'))])
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = Image.open(os.path.join(self.img_dir, name)).convert('RGB')
        label = Image.open(os.path.join(self.label_dir, name))

        img = np.array(img, dtype=np.float32) / 255.0
        label = np.array(label, dtype=np.float32)

        if label.ndim == 2:
            label = label[..., np.newaxis]
        if label.shape[-1] == 1:
            label = np.concatenate([label, 1.0 - label], axis=-1)

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        label = torch.from_numpy(label).permute(2, 0, 1).float()
        return img, label


# =========================
# 3. Training
# =========================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dataset = DecounterDataset(args.data_dir, split='train')
    val_dataset = DecounterDataset(args.data_dir, split='val')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    model = DecounterUNet(n_channels=3, n_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            pred = model(imgs)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss / len(train_dataset)
        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.6f}")

        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f"decounter_unet_{epoch+1}.pth"))
            print(f"  Model saved.")


# =========================
# 4. Inference
# =========================
@torch.no_grad()
def predict(model, img_tensor, device):
    model.eval()
    img_tensor = img_tensor.unsqueeze(0).to(device)
    logits = model(img_tensor)
    probs = torch.sigmoid(logits)
    return probs.squeeze(0).cpu().numpy()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_interval', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    train(args)
