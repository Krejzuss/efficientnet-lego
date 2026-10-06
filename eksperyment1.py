"""
Eksperyment 1: Wpływ augmentacji danych na skuteczność modelu
Wyniki: eksperyment1_wyniki.csv + eksperyment1_wykres.png

Użycie:
    python eksperyment1_augmentacje.py
"""

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os, time, csv, random
import matplotlib.pyplot as plt


# KONFIGURACJA
script_dir      = os.path.dirname(os.path.abspath(__file__))
data_dir        = os.path.abspath(os.path.join(script_dir, '..', 'dataset_lego'))
img_size        = 260
batch_size      = 64
warmup_epochs   = 8
finetune_epochs = 30
lr_head         = 0.001
lr_finetune     = 0.00005
patience        = 12
SEED            = 42

def build_model(num_classes, device):
    m = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    for p in m.parameters(): p.requires_grad = False
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(m.classifier[1].in_features, num_classes)
    )
    return m.to(device)

def run_epoch(model, loader, optimizer, criterion, scaler, device, use_amp, train=True, desc=''):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f'  {desc}', leave=False, unit='batch',
                colour='green' if train else 'cyan')
    for inputs, labels in pbar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with autocast('cuda', enabled=use_amp):
                outputs = model(inputs)
                loss    = criterion(outputs, labels)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * inputs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += inputs.size(0)
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return total_loss / total, correct / total

def train_variant(name, train_dataset, test_loader, num_classes, device, use_amp):
    print(f"\n{'='*55}")
    print(f"WARIANT: {name}")
    print(f"{'='*55}\n")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=6, pin_memory=True, persistent_workers=True,
    )

    model     = build_model(num_classes, device)
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)
    history   = []
    best_acc  = 0.0
    best_path = os.path.join(script_dir, f'expA_{name.replace(" ","_")}_best.pth')
    since     = time.time()

    # WARMUP
    optimizer = optim.AdamW(model.classifier.parameters(), lr=lr_head, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=warmup_epochs)
    for epoch in range(warmup_epochs):
        t_loss, t_acc = run_epoch(model, train_loader, optimizer, criterion, scaler, device, use_amp, train=True,  desc=f'W{epoch+1}/{warmup_epochs} train')
        v_loss, v_acc = run_epoch(model, test_loader,  optimizer, criterion, scaler, device, use_amp, train=False, desc=f'W{epoch+1}/{warmup_epochs} test ')
        scheduler.step()
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), best_path)
        print(f'  Warmup {epoch+1}/{warmup_epochs} | train: {t_acc:.4f} | test: {v_acc:.4f}')

    # FINE-TUNING
    for p in model.parameters(): p.requires_grad = True
    optimizer = optim.AdamW([
        {'params': model.features.parameters(),   'lr': lr_finetune},
        {'params': model.classifier.parameters(), 'lr': lr_finetune * 10},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs, eta_min=1e-7)
    no_improve = 0

    for epoch in range(finetune_epochs):
        t0 = time.time()
        t_loss, t_acc = run_epoch(model, train_loader, optimizer, criterion, scaler, device, use_amp, train=True,  desc=f'E{epoch+1}/{finetune_epochs} train')
        v_loss, v_acc = run_epoch(model, test_loader,  optimizer, criterion, scaler, device, use_amp, train=False, desc=f'E{epoch+1}/{finetune_epochs} test ')
        scheduler.step()

        if v_acc > best_acc:
            best_acc = v_acc; no_improve = 0
            torch.save(model.state_dict(), best_path)
            flag = '✓ nowy rekord'
        else:
            no_improve += 1
            flag = f'brak poprawy {no_improve}/{patience}'

        print(f'  Epoka {epoch+1:2d} | train: {t_acc:.4f} | test: {v_acc:.4f} | {flag}')
        history.append({
            'wariant': name, 'epoka': epoch + 1,
            'train_loss': round(t_loss, 4), 'train_acc': round(t_acc, 4),
            'test_loss':  round(v_loss, 4), 'test_acc':  round(v_acc, 4),
            'czas_s':     round(time.time() - t0, 1),
        })
        if no_improve >= patience:
            print(f'  Early stopping po {epoch+1} epokach.')
            break

    total_time = time.time() - since
    print(f'\n  Najlepsza acc test ({name}): {best_acc:.4f}')
    print(f'  Łączny czas treningu: {total_time/60:.1f} min')
    return history, best_acc, total_time


if __name__ == '__main__':
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == 'cuda'
    print(f"Urządzenie: {device}")

    test_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Wariant 1 — brak augmentacji
    transform_brak = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Wariant 2 — pełna augmentacja
    transform_pelna = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=12, scale=(0.9, 1.1), translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    ds_brak  = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform_brak)
    ds_pelna = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform_pelna)
    test_ds  = datasets.ImageFolder(os.path.join(data_dir, 'test'),  test_transform)
    num_classes = len(ds_brak.classes)
    print(f"Klas: {num_classes} | Train: {len(ds_brak)} | Test: {len(test_ds)}")

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=6, pin_memory=True, persistent_workers=True,
    )

    hist_brak,  best_brak,  t_brak  = train_variant("Brak augmentacji", ds_brak,  test_loader, num_classes, device, use_amp)
    hist_pelna, best_pelna, t_pelna = train_variant("Pelna augmentacja", ds_pelna, test_loader, num_classes, device, use_amp)

    # CSV
    csv_path = os.path.join(script_dir, 'eksperymentA_wyniki.csv')
    fields   = ['wariant','epoka','train_loss','train_acc','test_loss','test_acc','czas_s']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(hist_brak + hist_pelna)
    print(f"\nCSV: {csv_path}")

    # Wykres
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Eksperyment A: Wpływ augmentacji — zbiór testowy', fontweight='bold')
    for hist, label, color in [(hist_brak,'Brak augmentacji','#E24B4A'),(hist_pelna,'Pełna augmentacja','#378ADD')]:
        ep  = [r['epoka'] for r in hist]
        acc = [r['test_acc']*100 for r in hist]
        los = [r['test_loss'] for r in hist]
        axes[0].plot(ep, acc, label=label, color=color, linewidth=2)
        axes[1].plot(ep, los, label=label, color=color, linewidth=2)
    axes[0].set_title('Test accuracy'); axes[0].set_ylabel('Accuracy (%)'); axes[0].set_xlabel('Epoka'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_title('Test loss');     axes[1].set_ylabel('Loss');         axes[1].set_xlabel('Epoka'); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'eksperymentA_wykres.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n{'='*55}")
    print(f"PODSUMOWANIE EKSPERYMENTU A")
    print(f"{'Wariant':<22} {'Test acc':>10} {'Czas':>12}")
    print(f"{'-'*45}")
    print(f"{'Brak augmentacji':<22} {best_brak*100:>9.2f}% {t_brak/60:>10.1f} min")
    print(f"{'Pełna augmentacja':<22} {best_pelna*100:>9.2f}% {t_pelna/60:>10.1f} min")
    print(f"\n  Różnica accuracy: {(best_pelna-best_brak)*100:+.2f} pp")