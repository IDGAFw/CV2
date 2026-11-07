import copy, random
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")




DATA_DIR = './dataset'

full_dataset = datasets.ImageFolder(root=DATA_DIR)
class_names = full_dataset.classes
num_classes = len(class_names)
targets = np.array(full_dataset.targets)

print(f"Classes found ({num_classes}): {class_names}")
print(f"Total images: {len(full_dataset)}")

counts = Counter(targets.tolist())
for i, name in enumerate(class_names):
    flag = "" if counts[i] >= 100 else "  <-- меньше 100!"
    print(f"  {name}: {counts[i]}{flag}")

plt.figure(figsize=(7, 4))
plt.bar(class_names, [counts[i] for i in range(num_classes)], color='steelblue')
plt.title('Images per class'); plt.ylabel('Count'); plt.xticks(rotation=20)
plt.tight_layout(); plt.savefig('class_distribution.png', dpi=150); plt.show()



n_show = 4
fig, axes = plt.subplots(num_classes, n_show, figsize=(3 * n_show, 3 * num_classes))
rng = np.random.RandomState(SEED)
for c in range(num_classes):
    idxs = rng.choice(np.where(targets == c)[0], n_show, replace=False)
    for j, idx in enumerate(idxs):
        axes[c, j].imshow(full_dataset[idx][0]); axes[c, j].axis('off')
        if j == 0:
            axes[c, j].set_title(class_names[c], loc='left', fontsize=10)
plt.tight_layout(); plt.savefig('dataset_samples.png', dpi=150); plt.show()




IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset, self.transform = subset, transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


all_idx = np.arange(len(full_dataset))
train_idx, temp_idx = train_test_split(all_idx, test_size=0.30, stratify=targets, random_state=SEED)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=targets[temp_idx], random_state=SEED)

train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)
test_dataset = Subset(full_dataset, test_idx)

train_data = DatasetWrapper(train_dataset, train_transforms)
val_data = DatasetWrapper(val_dataset, val_test_transforms)
test_data = DatasetWrapper(test_dataset, val_test_transforms)

BATCH_SIZE = 32
g = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, generator=g)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
for name, idx in [('Train', train_idx), ('Val', val_idx), ('Test', test_idx)]:
    c = Counter(targets[idx].tolist())
    print(f"  {name}: " + ", ".join(f"{class_names[i]}={c[i]}" for i in range(num_classes)))




class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_improved_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False
    n = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(n, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))
    return model


def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable



def train_model(model, criterion, optimizer, train_loader, val_loader, device,
                num_epochs=15, freeze_bn=False):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_wts, best_acc = copy.deepcopy(model.state_dict()), 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        for phase, loader in [('train', train_loader), ('val', val_loader)]:
            if phase == 'train':
                model.train()
                if freeze_bn:
                    for m in model.modules():
                        if isinstance(m, nn.BatchNorm2d):
                            m.eval()
            else:
                model.eval()

            running_loss, running_corrects = 0.0, 0
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()

            n = len(loader.dataset)
            epoch_loss, epoch_acc = running_loss / n, running_corrects / n
            print(f'  {phase.capitalize():5s} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_wts = copy.deepcopy(model.state_dict())

    print(f'Best Validation Accuracy: {best_acc:.4f}')
    model.load_state_dict(best_wts)
    return model, history


@torch.no_grad()
def evaluate_model_on_test(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for inputs, labels in loader:
        outputs = model(inputs.to(device))
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


EPOCHS = 15
criterion = nn.CrossEntropyLoss()

baseline_model = BaselineCNN(num_classes).to(device)
print("Params (total, trainable):", count_params(baseline_model))
optimizer_b = optim.Adam(baseline_model.parameters(), lr=1e-3)
baseline_model, history_baseline = train_model(
    baseline_model, criterion, optimizer_b, train_loader, val_loader, device, EPOCHS)
torch.save(baseline_model.state_dict(), 'baseline_cnn.pth')


improved_model = get_improved_model(num_classes).to(device)
print("Params (total, trainable):", count_params(improved_model))
optimizer_imp = optim.Adam(improved_model.fc.parameters(), lr=1e-3)
improved_model, history_improved = train_model(
    improved_model, criterion, optimizer_imp, train_loader, val_loader, device, EPOCHS, freeze_bn=True)
torch.save(improved_model.state_dict(), 'resnet18_tl.pth')


fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for key, title, a in [('loss', 'Loss', ax[0]), ('acc', 'Accuracy', ax[1])]:
    a.plot(history_baseline[f'train_{key}'], '--', label=f'Baseline Train')
    a.plot(history_baseline[f'val_{key}'], label=f'Baseline Val')
    a.plot(history_improved[f'train_{key}'], '--', label=f'ResNet18 Train')
    a.plot(history_improved[f'val_{key}'], label=f'ResNet18 Val')
    a.set_title(f'{title} Curves Comparison'); a.set_xlabel('Epoch'); a.set_ylabel(title)
    a.legend(); a.grid(True)
plt.tight_layout(); plt.savefig('training_curves.png', dpi=150); plt.show()




labels, baseline_preds = evaluate_model_on_test(baseline_model, test_loader, device)
_, improved_preds = evaluate_model_on_test(improved_model, test_loader, device)

baseline_acc = accuracy_score(labels, baseline_preds)
improved_acc = accuracy_score(labels, improved_preds)

print("=" * 50, "\nBASELINE CNN\n" + "=" * 50)
print(classification_report(labels, baseline_preds, target_names=class_names, digits=4))
print("=" * 50, "\nRESNET18 (TRANSFER LEARNING)\n" + "=" * 50)
print(classification_report(labels, improved_preds, target_names=class_names, digits=4))


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for a, preds, cmap, title in [(axes[0], baseline_preds, 'Blues', 'Baseline CNN'),
                              (axes[1], improved_preds, 'Greens', 'ResNet18 Transfer Learning')]:
    sns.heatmap(confusion_matrix(labels, preds), annot=True, fmt='d', cmap=cmap,
                xticklabels=class_names, yticklabels=class_names, ax=a)
    a.set_title(f'{title} - Confusion Matrix'); a.set_xlabel('Predicted'); a.set_ylabel('True')
    a.tick_params(axis='x', rotation=20)
plt.tight_layout(); plt.savefig('confusion_matrices.png', dpi=150); plt.show()


wrong = np.where(improved_preds != labels)[0]
print(f"Misclassified: {len(wrong)} / {len(labels)}")

n = min(8, len(wrong))
if n:
    fig, axes = plt.subplots(2, 4, figsize=(14, 7)); axes = axes.ravel()
    for a in axes: a.axis('off')
    for a, i in zip(axes, wrong[:n]):
        img, _ = test_dataset[i]
        a.imshow(img)
        a.set_title(f"True: {class_names[labels[i]]}\nPred: {class_names[improved_preds[i]]}", fontsize=9)
    plt.tight_layout(); plt.savefig('misclassified.png', dpi=150); plt.show()



print("=" * 60)
print(f"Baseline CNN Test Accuracy:                  {baseline_acc * 100:.2f}%")
print(f"ResNet18 (Transfer Learning) Test Accuracy:  {improved_acc * 100:.2f}%")
print("-" * 60)
if improved_acc > baseline_acc:
    print("Winner: ResNet18 (transfer learning).")
elif improved_acc < baseline_acc:
    print("Winner: Baseline CNN.")
else:
    print("Equal test accuracy.")
print("=" * 60)