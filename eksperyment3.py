"""
Eksperyment 3: Porównanie architektur EfficientNet (B0 vs B2 vs B4)
Wyniki: eksperyment3_wyniki.csv + eksperyment3_wykres.png

Użycie:
    python eksperyment_3_modele.py
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
batch_size      = 32   
warmup_epochs   = 8
finetune_epochs = 30
lr_head         = 0.001
lr_finetune     = 0.00005
patience        = 12
SEED            = 42

# Konfiguracje modeli: (nazwa, funkcja modelu, wagi, img_size)
MODEL_CONFIGS = [
    ('EfficientNet-B0', models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, 224),
    ('EfficientNet-B2', models.efficientnet_b2, models.EfficientNet_B2_Weights.IMAGENET1K_V1, 260),
    ('EfficientNet-B4', models.efficientnet_b4, models.EfficientNet_B4_Weights.IMAGENET1K_V1, 380),
]

def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1_000_000

def build_model(model_fn, weights, num_classes, device):
    m = model_fn(weights=weights)
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

def train_model(config_name, model_fn, weights, img_size, train_ds_path, test_ds_path, num_classes, device, use_amp):
    print(f"\n{'='*55}")
    print(f"MODEL: {config_name}  (img_size={img_size})")
    print(f"{'='*55}\n")

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=12, scale=(0.9, 1.1), translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds  = datasets.ImageFolder(train_ds_path, train_transform)
    test_ds   = datasets.ImageFolder(test_ds_path,  test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=6, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=6, pin_memory=True, persistent_workers=True)

    model      = build_model(model_fn, weights, num_classes, device)
    n_params   = count_params(model)
    criterion  = nn.CrossEntropyLoss()
    scaler     = torch.amp.GradScaler('cuda', enabled=use_amp)
    history    = []
    best_acc   = 0.0
    best_path  = os.path.join(script_dir, f'expB_{config_name.replace(" ","_")}_best.pth')
    since      = time.time()
    epoch_times = []

    print(f"  Parametry: {n_params:.1f}M")

    # WARMUP
    optimizer = optim.AdamW(model.classifier.parameters(), lr=lr_head, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=warmup_epochs)
    for epoch in range(warmup_epochs):
        t0 = time.time()
        t_loss, t_acc = run_epoch(model, train_loader, optimizer, criterion, scaler, device, use_amp, train=True,  desc=f'W{epoch+1}/{warmup_epochs} train')
        v_loss, v_acc = run_epoch(model, test_loader,  optimizer, criterion, scaler, device, use_amp, train=False, desc=f'W{epoch+1}/{warmup_epochs} test ')
        scheduler.step()
        epoch_times.append(time.time() - t0)
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
        et = time.time() - t0
        epoch_times.append(et)

        if v_acc > best_acc:
            best_acc = v_acc; no_improve = 0
            torch.save(model.state_dict(), best_path)
            flag = '✓ nowy rekord'
        else:
            no_improve += 1
            flag = f'brak poprawy {no_improve}/{patience}'

        print(f'  Epoka {epoch+1:2d} | train: {t_acc:.4f} | test: {v_acc:.4f} | {et:.0f}s | {flag}')
        history.append({
            'model': config_name, 'epoka': epoch + 1,
            'train_loss': round(t_loss, 4), 'train_acc': round(t_acc, 4),
            'test_loss':  round(v_loss, 4), 'test_acc':  round(v_acc, 4),
            'czas_s':     round(et, 1),
        })
        if no_improve >= patience:
            print(f'  Early stopping po {epoch+1} epokach.')
            break

    total_time     = time.time() - since
    avg_epoch_time = np.mean(epoch_times)
    print(f'\n  Najlepsza acc test: {best_acc:.4f}')
    print(f'  Łączny czas: {total_time/60:.1f} min | Śr. epoka: {avg_epoch_time:.1f}s')

    return history, best_acc, total_time, avg_epoch_time, n_params


if __name__ == '__main__':
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == 'cuda'
    print(f"Urządzenie: {device}")

    train_path = os.path.join(data_dir, 'train')
    test_path  = os.path.join(data_dir, 'test')
    num_classes = len([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
    print(f"Klas: {num_classes}\n")

    all_history = []
    summary     = []

    for config_name, model_fn, weights, img_size in MODEL_CONFIGS:
        hist, best_acc, total_t, avg_epoch_t, n_params = train_model(
            config_name, model_fn, weights, img_size,
            train_path, test_path, num_classes, device, use_amp
        )
        all_history.extend(hist)
        summary.append({
            'model':           config_name,
            'parametry_M':     round(n_params, 1),
            'img_size':        img_size,
            'best_test_acc':   round(best_acc, 4),
            'total_czas_min':  round(total_t / 60, 1),
            'avg_epoka_s':     round(avg_epoch_t, 1),
        })

    # CSV historia
    csv_hist = os.path.join(script_dir, 'eksperyment3_historia.csv')
    fields   = ['model','epoka','train_loss','train_acc','test_loss','test_acc','czas_s']
    with open(csv_hist, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_history)

    # CSV podsumowanie
    csv_sum = os.path.join(script_dir, 'eksperyment3_podsumowanie.csv')
    with open(csv_sum, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    print(f"\nCSV: {csv_hist}, {csv_sum}")

    # Wykres
    colors = {'EfficientNet-B0': '#1D9E75', 'EfficientNet-B2': '#378ADD', 'EfficientNet-B4': '#E24B4A'}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Eksperyment B: Porównanie architektur EfficientNet', fontweight='bold')

    for s in summary:
        hist_m = [r for r in all_history if r['model'] == s['model']]
        ep  = [r['epoka'] for r in hist_m]
        acc = [r['test_acc']*100 for r in hist_m]
        los = [r['test_loss'] for r in hist_m]
        c   = colors[s['model']]
        axes[0].plot(ep, acc, label=s['model'], color=c, linewidth=2)
        axes[1].plot(ep, los, label=s['model'], color=c, linewidth=2)

    axes[0].set_title('Test accuracy'); axes[0].set_ylabel('Accuracy (%)'); axes[0].set_xlabel('Epoka'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_title('Test loss');     axes[1].set_ylabel('Loss');         axes[1].set_xlabel('Epoka'); axes[1].legend(); axes[1].grid(alpha=0.3)

    # Wykres słupkowy — accuracy vs czas
    names     = [s['model'].replace('EfficientNet-','B') for s in summary]
    accs      = [s['best_test_acc']*100 for s in summary]
    times     = [s['total_czas_min'] for s in summary]
    bar_colors = [colors[s['model']] for s in summary]
    x = np.arange(len(names))
    ax3 = axes[2]
    bars = ax3.bar(x, accs, color=bar_colors, alpha=0.8, label='Accuracy (%)')
    ax3.set_ylabel('Test accuracy (%)')
    ax3.set_xticks(x); ax3.set_xticklabels(names)
    ax3.set_title('Accuracy vs czas treningu')
    ax3.grid(axis='y', alpha=0.3)
    ax3b = ax3.twinx()
    ax3b.plot(x, times, 'o--', color='#888', linewidth=2, label='Czas (min)')
    ax3b.set_ylabel('Czas treningu (min)')
    for bar, acc, t in zip(bars, accs, times):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{acc:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'eksperyment3_wykres.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n{'='*60}")
    print(f"PODSUMOWANIE EKSPERYMENTU 3")
    print(f"{'Model':<20} {'Param':>8} {'Acc':>8} {'Czas':>10} {'Śr.epoka':>10}")
    print(f"{'-'*60}")
    for s in summary:
        print(f"{s['model']:<20} {s['parametry_M']:>6.1f}M {s['best_test_acc']*100:>7.2f}% {s['total_czas_min']:>8.1f}min {s['avg_epoka_s']:>8.1f}s")