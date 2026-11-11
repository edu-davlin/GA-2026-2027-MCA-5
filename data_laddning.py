"""
data_laddning.py - läser in bilderna från data/train och data/test.

Används av trana.py och utvardera.py. Kan också köras själv för att
kontrollera att datan hittas:

    python data_laddning.py                 # senaste datasetet
    python data_laddning.py syntetisk_V1    # ett visst dataset

Varje dataset ligger i en egen mapp i dataset/:
    dataset/<namn>/train/<form>/*.png
    dataset/<namn>/test/<form>/*.png
Mappnamnen (circle, square, ...) blir automatiskt kategorierna.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Storleken bilderna skalas ner till innan de går in i MCA-5.
# 224x224 är onödigt stort för former; 64x64 räcker och tränar snabbare.
STORLEK = 64

# Normalisering: flyttar pixelvärden från 0..1 till -1..1 (stabilare träning)
MEDEL = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

# Förbehandling för TRÄNING: med "data augmentation", alltså små slumpade
# ändringar varje gång en bild visas, så att modellen tål verkligheten bättre.
train_transform = transforms.Compose([
    transforms.Resize((STORLEK, STORLEK)),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(MEDEL, STD),
])

# Förbehandling för TEST och för webbsidan: inga slumpändringar.
# OBS: webbsidan måste använda exakt denna, annars ser modellen
# bilderna annorlunda än under träningen.
test_transform = transforms.Compose([
    transforms.Resize((STORLEK, STORLEK)),
    transforms.ToTensor(),
    transforms.Normalize(MEDEL, STD),
])


def hitta_dataset(namn=None):
    """Returnerar mappen för ett dataset.
    namn: datasetets namn i dataset/ (eller en sökväg). Utan namn väljs
    det senast skapade datasetet i dataset/."""
    if namn:
        mapp = Path(namn)
        if not (mapp / "train").exists():
            mapp = Path("dataset") / namn
        if not (mapp / "train").exists():
            raise FileNotFoundError(f"Hittar inget dataset '{namn}' (letade efter {mapp / 'train'})")
        return mapp
    mappar = [m for m in Path("dataset").glob("*") if (m / "train").exists()]
    if not mappar:
        raise FileNotFoundError("Inga dataset i dataset/. Kör generera_data.py först.")
    return max(mappar, key=lambda m: m.stat().st_ctime)


def skapa_dataloaders(datamapp=None, batch_storlek=64, workers=2):
    """Returnerar (train_loader, test_loader, kategorinamn)."""
    datamapp = hitta_dataset(datamapp)
    train_data = datasets.ImageFolder(datamapp / "train", transform=train_transform)
    test_data = datasets.ImageFolder(datamapp / "test", transform=test_transform)

    if train_data.classes != test_data.classes:
        raise ValueError(
            f"train och test har olika formmappar:\n"
            f"  train: {train_data.classes}\n  test:  {test_data.classes}")

    train_loader = DataLoader(train_data, batch_size=batch_storlek, shuffle=True,
                              num_workers=workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_data, batch_size=batch_storlek, shuffle=False,
                             num_workers=workers, pin_memory=torch.cuda.is_available())
    return train_loader, test_loader, train_data.classes


# if __name__ == "__main__" behövs på Windows när num_workers > 0,
# annars startar programmet om sig själv i en evig loop.
if __name__ == "__main__":
    import sys
    namn = sys.argv[1] if len(sys.argv) > 1 else None
    print("Dataset:", hitta_dataset(namn))
    train_loader, test_loader, kategorier = skapa_dataloaders(namn)
    print("Kategorier:", kategorier)
    print("Träningsbilder:", len(train_loader.dataset))
    print("Testbilder:   ", len(test_loader.dataset))
    bilder, etiketter = next(iter(train_loader))
    print("En batch:", bilder.shape, "(antal, färgkanaler, höjd, bredd)")
