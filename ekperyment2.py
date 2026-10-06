"""
Eksperyment 2: Benchmark GPU CUDA vs CPU
Użycie:
    python eksperyment2_benchmark.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os, time, csv

TOTAL_EPOCHS   = 38  
script_dir     = os.path.dirname(os.path.abspath(__file__))
data_dir       = os.path.abspath(os.path.join(script_dir, '..', 'dataset_lego'))
img_size       = 260
batch_size     = 64

train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=12, scale=(0.9, 1.1), translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def measure_epoch(device_str, num_workers, use_amp_flag, dataset, num_classes):
    device = torch.device(device_str)
    print(f"\nMierzę epokę na: {device_str.upper()} (AMP: {use_amp_flag}, workers: {num_workers})")

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=6, pin_memory=(device.type == 'cuda'),
        persistent_workers=True,
    )

    m = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(m.classifier[1].in_features, num_classes)
    )
    m = m.to(device)
    m.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(m.classifier.parameters(), lr=0.001, weight_decay=1e-4)
    scaler    = torch.amp.GradScaler(device_str, enabled=use_amp_flag)

    # Rozgrzewka — 5 batchy
    print("  Rozgrzewka...", end=' ', flush=True)
    for i, (inp, lab) in enumerate(loader):
        if i >= 5: break
        inp, lab = inp.to(device), lab.to(device)
        with autocast(device_str, enabled=use_amp_flag):
            loss = criterion(m(inp), lab)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        optimizer.zero_grad(set_to_none=True)
    print("gotowa")

    # Pomiar — pełna epoka
    from tqdm import tqdm
    n = len(dataset)
    t0 = time.time()
    pbar = tqdm(loader, desc=f'  {device_str.upper()} epoka', unit='batch', colour='green')
    for inp, lab in pbar:
        inp, lab = inp.to(device), lab.to(device)
        with autocast(device_str, enabled=use_amp_flag):
            loss = criterion(m(inp), lab)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        optimizer.zero_grad(set_to_none=True)
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    elapsed = time.time() - t0

    img_per_s = n / elapsed
    est_min   = elapsed * TOTAL_EPOCHS / 60

    print(f"  Czas epoki:        {elapsed:.1f} s")
    print(f"  Przepustowość:     {img_per_s:.0f} obrazów/s")
    print(f"  Szacowany trening: {est_min:.0f} min ({est_min/60:.1f} h)")

    return {
        'urzadzenie':          device_str.upper(),
        'czas_epoki_s':        round(elapsed, 1),
        'obrazy_na_s':         round(img_per_s, 0),
        'szacowany_czas_min':  round(est_min, 0),
    }


if __name__ == '__main__':
    dataset     = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transform)
    num_classes = len(dataset.classes)
    n_images    = len(dataset)
    print(f"Klas: {num_classes} | Zdjęć: {n_images}")

    results = []

    if torch.cuda.is_available():
        r = measure_epoch('cuda', num_workers=6, use_amp_flag=True,
                          dataset=dataset, num_classes=num_classes)
        results.append(r)
    else:
        print("Brak GPU — pomiar tylko CPU")

    r = measure_epoch('cpu', num_workers=4, use_amp_flag=False,
                      dataset=dataset, num_classes=num_classes)
    results.append(r)

    print(f"\n{'='*60}")
    print(f"PODSUMOWANIE EKSPERYMENTU 2")
    print(f"{'='*60}")
    print(f"{'Urządzenie':<12} {'Czas epoki':>12} {'Obrazy/s':>12} {'Szac. czas':>16}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['urzadzenie']:<12} {r['czas_epoki_s']:>10.1f} s "
              f"{r['obrazy_na_s']:>10.0f} "
              f"{r['szacowany_czas_min']:>12.0f} min")

    if len(results) == 2:
        speedup = results[0]['obrazy_na_s'] / results[1]['obrazy_na_s']
        time_saved = results[1]['szacowany_czas_min'] - results[0]['szacowany_czas_min']
        print(f"\n  Przyspieszenie GPU vs CPU:  {speedup:.1f}x")
        print(f"  Oszczędność czasu:          {time_saved:.0f} min")

    csv_path = os.path.join(script_dir, 'eksperyment2_wyniki.csv')
    fields   = ['urzadzenie','czas_epoki_s','obrazy_na_s','szacowany_czas_min']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(results)
    print(f"\nCSV: {csv_path}")