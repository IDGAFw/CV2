import numpy as np, torch, torch.nn as nn
from PIL import Image
import matplotlib
from transformers import ViTForImageClassification

VIT_ID = "google/vit-base-patch16-224-in21k"
VIT_MEAN = VIT_STD = [0.5, 0.5, 0.5]


def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")


def sync(device):
    if device.type == "cuda": torch.cuda.synchronize()
    elif device.type == "mps": torch.mps.synchronize()


class ViTClassifier(nn.Module):
    """ViT-B/16 (ImageNet-21k pretrained) + new classification head."""
    def __init__(self, num_classes):
        super().__init__()
        self.vit = ViTForImageClassification.from_pretrained(
            VIT_ID, num_labels=num_classes, attn_implementation="eager")

    def forward(self, x):
        return self.vit(pixel_values=x).logits


@torch.no_grad()
def predict_with_rollout(model, x):
    """Returns logits and attention-rollout masks (B,14,14) in [0,1] (Abnar & Zuidema, 2020)."""
    model.eval()
    out = model.vit(pixel_values=x, output_attentions=True)
    att = torch.stack(out.attentions).mean(2)
    T = att.size(-1)
    att = att + torch.eye(T, device=att.device)
    att = att / att.sum(-1, keepdim=True)
    roll = att[0]
    for l in range(1, att.size(0)):
        roll = att[l] @ roll
    mask = roll[:, 0, 1:]
    s = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(-1, s, s)
    mn, mx = mask.amin((1, 2), keepdim=True), mask.amax((1, 2), keepdim=True)
    mask = (mask - mn) / (mx - mn + 1e-8)
    return out.logits, mask.cpu().numpy()


def overlay(img: Image.Image, mask, alpha=0.5):
    img = img.convert("RGB")
    m = Image.fromarray((mask * 255).astype("uint8")).resize(img.size, Image.BICUBIC)
    heat = matplotlib.colormaps["jet"](np.asarray(m) / 255.0)[..., :3]
    out = (1 - alpha) * np.asarray(img) / 255.0 + alpha * heat
    return (np.clip(out, 0, 1) * 255).astype("uint8")
