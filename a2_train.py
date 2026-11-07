import copy, json, os, pickle, random, re, time
from collections import Counter
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from PIL import Image
from a2_common import (get_device, sync, ViTClassifier, predict_with_rollout, overlay,
                       VIT_MEAN, VIT_STD)

SEED = 42
DATA_DIR = './dataset'
OUT = 'outputs'; os.makedirs(OUT, exist_ok=True)
device = get_device(); print("Device:", device)


def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


set_seed()
full_dataset = datasets.ImageFolder(DATA_DIR)
class_names, num_classes = full_dataset.classes, len(full_dataset.classes)
targets = np.array(full_dataset.targets)
n = len(full_dataset)
json.dump(class_names, open('classes.json', 'w'))
print(class_names, n, dict(Counter(targets.tolist())))

def dhash(path, size=8):
    im = Image.open(path).convert('L').resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten().astype(np.float32)

H = np.stack([dhash(p) for p, _ in full_dataset.samples])
dist = H.shape[1] - (H @ H.T + (1 - H) @ (1 - H).T)          # Hamming distance
THR = 6
parent = list(range(n))
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]; i = parent[i]
    return i
ii, jj = np.where(np.triu(dist <= THR, 1) & (targets[:, None] == targets[None, :]))
for i, j in zip(ii, jj): parent[find(i)] = find(j)
groups = np.array([find(i) for i in range(n)])
print(f"Near-duplicate groups: {len(set(groups))} for {n} images")

all_idx = np.arange(n)
rng = np.random.RandomState(SEED)
split_of = np.zeros(n, dtype=int)
FRAC = np.array([0.70, 0.15, 0.15])
for c in range(num_classes):
    g_ids, g_sizes = np.unique(groups[targets == c], return_counts=True)
    order = np.lexsort((rng.rand(len(g_ids)), -g_sizes))   # biggest groups first
    print(f"  {class_names[c]}: {len(g_ids)} groups, biggest = {g_sizes.max()} of {g_sizes.sum()}")
    want, cur = FRAC * g_sizes.sum(), np.zeros(3)
    for k in order:
        sp = int(np.argmax(want - cur))    # split with the biggest deficit
        split_of[groups == g_ids[k]] = sp; cur[sp] += g_sizes[k]
train_idx, val_idx, test_idx = [all_idx[split_of == k] for k in range(3)]
assert not (set(groups[train_idx]) & set(groups[val_idx]) or set(groups[train_idx]) & set(groups[test_idx])
            or set(groups[val_idx]) & set(groups[test_idx]))
print(f"Split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
for nm, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
    c = Counter(targets[idx].tolist()); print(" ", nm, {class_names[k]: c[k] for k in range(num_classes)})


class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform): self.subset, self.transform = subset, transform
    def __getitem__(self, i):
        x, y = self.subset[i]; return self.transform(x.convert('RGB')), y
    def __len__(self): return len(self.subset)


def make_loaders(mean, std, augment=True, bs=32):
    norm = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    aug = [transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(15),
           transforms.ColorJitter(0.2, 0.2, 0.2)] if augment else [transforms.Resize((224, 224))]
    t_train, t_eval = transforms.Compose(aug + norm), transforms.Compose([transforms.Resize((224, 224))] + norm)
    mk = lambda idx, t: DatasetWrapper(Subset(full_dataset, idx), t)
    g = torch.Generator().manual_seed(SEED)
    tr_d, va_d, te_d = mk(train_idx, t_train), mk(val_idx, t_eval), mk(test_idx, t_eval)
    return (DataLoader(tr_d, bs, shuffle=True, generator=g), DataLoader(va_d, bs), DataLoader(te_d, bs), te_d)


# ---------- models ----------
class BaselineCNN(nn.Module):  # A1 baseline (unchanged)
    def __init__(self, nc):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128 * 28 * 28, 128), nn.ReLU(),
                                        nn.Dropout(0.5), nn.Linear(128, nc))
    def forward(self, x): return self.classifier(self.features(x))


# 2. ResNet18 Transfer Learning
def resnet18_tl(nc):  # A1 "improved" model (unchanged)
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for p in m.parameters(): p.requires_grad = False
    m.fc = nn.Sequential(nn.Linear(m.fc.in_features, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, nc))
    return m

# 3. ViT-B/16
def count_params(m):
    return sum(p.numel() for p in m.parameters()), sum(p.numel() for p in m.parameters() if p.requires_grad)




def train_model(model, criterion, optimizer, tr_l, va_l, epochs, freeze_bn=False):
    hist = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_wts, best_acc = copy.deepcopy(model.state_dict()), 0.0
    for ep in range(epochs):
        print(f'Epoch {ep + 1}/{epochs}')
        for phase, loader in [('train', tr_l), ('val', va_l)]:
            model.train() if phase == 'train' else model.eval()
            if phase == 'train' and freeze_bn:
                for m in model.modules():
                    if isinstance(m, nn.BatchNorm2d): m.eval()
            rl, rc = 0.0, 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    out = model(x); loss = criterion(out, y)
                    if phase == 'train': loss.backward(); optimizer.step()
                rl += loss.item() * x.size(0); rc += (out.argmax(1) == y).sum().item()
            N = len(loader.dataset)
            hist[f'{phase}_loss'].append(rl / N); hist[f'{phase}_acc'].append(rc / N)
            print(f'  {phase:5s} loss {rl / N:.4f} acc {rc / N:.4f}')
            if phase == 'val' and rc / N > best_acc:
                best_acc, best_wts = rc / N, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_wts)
    return model, hist



@torch.no_grad()
def evaluate(model, loader):
    model.eval(); P, L = [], []
    for x, y in loader:
        P.extend(model(x.to(device)).argmax(1).cpu().numpy()); L.extend(y.numpy())
    return np.array(L), np.array(P)



@torch.no_grad()
def latency_ms(model, ds, n=40):
    model.eval()
    xs = [ds[i][0].unsqueeze(0).to(device) for i in range(min(n + 5, len(ds)))]
    for x in xs[:5]: model(x)
    sync(device); t = time.perf_counter()
    for x in xs[5:]: model(x)
    sync(device)
    return (time.perf_counter() - t) / max(1, len(xs) - 5) * 1000




IMNET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
EXPERIMENTS = [

    ('Baseline CNN (A1)', lambda: BaselineCNN(num_classes), IMNET,              True,  32, 15, lambda m: optim.Adam(m.parameters(), 1e-3)),
    ('ResNet18 TL (A1)',  lambda: resnet18_tl(num_classes), IMNET,              True,  32, 15, lambda m: optim.Adam(m.fc.parameters(), 1e-3)),
    ('ViT-B/16 (SOTA)',   lambda: ViTClassifier(num_classes), (VIT_MEAN, VIT_STD), True,  16, 8,  lambda m: optim.AdamW(m.parameters(), 3e-5, weight_decay=0.05)),
    ('ViT-B/16 no-aug (ablation)', lambda: ViTClassifier(num_classes), (VIT_MEAN, VIT_STD), False, 16, 8, lambda m: optim.AdamW(m.parameters(), 3e-5, weight_decay=0.05)),
]



results, models_out = {}, {}
for name, build, (mean, std), aug, bs, epochs, make_opt in EXPERIMENTS:
    print("\n" + "=" * 60, f"\n{name}\n" + "=" * 60)
    slug = re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')
    cache = f"{OUT}/cache_{slug}.pkl"
    if os.path.exists(cache):              # already trained in a previous run -> reuse
        results[name] = pickle.load(open(cache, 'rb')); print("loaded from cache")
        if name == 'ViT-B/16 (SOTA)':
            m = ViTClassifier(num_classes); m.load_state_dict(torch.load('vit_scene.pth', map_location='cpu'))
            models_out['vit'] = (m.to(device), mean, std)
        continue
    set_seed()
    tr_l, va_l, te_l, te_ds = make_loaders(mean, std, aug, bs)
    model = build().to(device)
    tot, trn = count_params(model); print(f"Params total={tot:,} trainable={trn:,}")
    t0 = time.time()
    model, hist = train_model(model, nn.CrossEntropyLoss(), make_opt(model), tr_l, va_l, epochs,
                              freeze_bn=name.startswith('ResNet'))
    train_time = time.time() - t0
    if name == 'ViT-B/16 (SOTA)': torch.save(model.state_dict(), 'vit_scene.pth')   # save before anything can crash
    L, P = evaluate(model, te_l)
    rep = classification_report(L, P, labels=list(range(num_classes)), target_names=class_names, digits=4, zero_division=0)
    print(rep); open(f"{OUT}/report_{slug}.txt", 'w').write(rep)
    results[name] = dict(history=hist, labels=L, preds=P, row={
        'Model': name, 'Test Acc': accuracy_score(L, P), 'Macro-F1': f1_score(L, P, average='macro', labels=list(range(num_classes)), zero_division=0),
        'Best Val Acc': max(hist['val_acc']), 'Params (M)': tot / 1e6, 'Trainable (M)': trn / 1e6,
        'Latency ms/img': latency_ms(model, te_ds), 'Train time s': train_time})
    if name == 'ViT-B/16 (SOTA)': models_out['vit'] = (model, mean, std)
    pickle.dump(results[name], open(cache, 'wb'))

table = pd.DataFrame([r['row'] for r in results.values()]).round(4)
print("\n", table.to_string(index=False)); table.to_csv(f"{OUT}/metrics_table.csv", index=False)



fig, ax = plt.subplots(1, 2, figsize=(15, 5))
for nm, r in results.items():
    for a, k in zip(ax, ['loss', 'acc']):
        l, = a.plot(r['history'][f'val_{k}'], label=f'{nm} val')
        a.plot(r['history'][f'train_{k}'], '--', color=l.get_color())
for a, t in zip(ax, ['Loss', 'Accuracy']):
    a.set_title(f'{t} (solid=val, dashed=train)'); a.set_xlabel('Epoch'); a.grid(True); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f'{OUT}/training_curves.png', dpi=150); plt.close()



names3 = list(results)[:3]
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
for a, nm in zip(axes, names3):
    sns.heatmap(confusion_matrix(results[nm]['labels'], results[nm]['preds'], labels=list(range(num_classes))), annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=a)
    a.set_title(nm); a.set_xlabel('Predicted'); a.set_ylabel('True'); a.tick_params(axis='x', rotation=20)
plt.tight_layout(); plt.savefig(f'{OUT}/confusion_matrices.png', dpi=150); plt.close()



vit, mean, std = models_out['vit']
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean, std)])
sel = []
for c in range(num_classes): sel += [i for i in test_idx if targets[i] == c][:2]
fig, axes = plt.subplots(2, len(sel), figsize=(2.6 * len(sel), 5.6))
for k, i in enumerate(sel):
    img = full_dataset[i][0].convert('RGB')
    logits, mask = predict_with_rollout(vit, tf(img).unsqueeze(0).to(device))
    axes[0, k].imshow(img); axes[0, k].set_title(f"True: {class_names[targets[i]]}", fontsize=7)
    axes[1, k].imshow(overlay(img, mask[0])); axes[1, k].set_title(f"Pred: {class_names[logits.argmax(1).item()]}", fontsize=7)
    axes[0, k].axis('off'); axes[1, k].axis('off')
plt.tight_layout(); plt.savefig(f'{OUT}/attention_rollout.png', dpi=150); plt.close()
print(f"\nDone. Figures/tables in ./{OUT}/, weights: vit_scene.pth, classes.json")