"""
modell.py - själva MCA-5, ett CNN (Convolutional Neural Network).

Uppbyggnad:
  3 faltningsblock som hittar mönster i bilden, från enkla (kanter,
  streck) till mer sammansatta (hörn, kurvor, hela former):

    Conv2d    -> letar efter små mönster med filter på 3x3 pixlar
    BatchNorm -> håller värdena i lagom storlek, stabilare träning
    ReLU      -> gör nätet icke-linjärt (negativa värden blir 0)
    MaxPool   -> halverar bildens storlek och behåller de starkaste signalerna

  Därefter en klassificerare (vanliga "fully connected"-lager) som
  väger ihop allt till en poäng per form. Högst poäng = gissningen.

Bildens storlek genom nätet (med 64x64 in):
  3x64x64 -> 32x32x32 -> 64x16x16 -> 128x8x8 -> 8192 tal -> 256 -> 3 former
"""

from pathlib import Path

import torch
from torch import nn


def faltningsblock(in_kanaler, ut_kanaler):
    return nn.Sequential(
        nn.Conv2d(in_kanaler, ut_kanaler, kernel_size=3, padding=1),
        nn.BatchNorm2d(ut_kanaler),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class MCA5(nn.Module):
    def __init__(self, antal_former, storlek=64):
        super().__init__()
        self.egenskaper = nn.Sequential(
            faltningsblock(3, 32),
            faltningsblock(32, 64),
            faltningsblock(64, 128),
        )
        kvar = storlek // 8  # tre MaxPool som var och en halverar
        self.klassificerare = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * kvar * kvar, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  # stänger av slumpade nervceller under träning -> mindre överanpassning
            nn.Linear(256, antal_former),
        )

    def forward(self, x):
        return self.klassificerare(self.egenskaper(x))


def hitta_modellmapp(namn=None, kräver="modell.pth"):
    """Returnerar mappen för en modell i modeller/.
    namn: mappens namn (eller sökväg). Utan namn väljs den senast ändrade
    mappen som innehåller filen 'kräver'."""
    if namn:
        mapp = Path(namn)
        if not mapp.exists():
            mapp = Path("modeller") / namn
        if not (mapp / kräver).exists():
            raise FileNotFoundError(f"Hittar inte {mapp / kräver}")
        return mapp
    mappar = [m for m in Path("modeller").glob("*") if (m / kräver).exists()]
    if not mappar:
        raise FileNotFoundError(f"Ingen mapp i modeller/ innehåller {kräver}.")
    return max(mappar, key=lambda m: (m / kräver).stat().st_mtime)


if __name__ == "__main__":
    # Snabbtest utan data: skicka in en slumpad "låtsasbild"
    modell = MCA5(antal_former=3)
    låtsasbild = torch.randn(1, 3, 64, 64)
    ut = modell(låtsasbild)
    print("Utdata:", ut.shape, "-> en poäng per form")
    print("Antal parametrar (vikter):", sum(p.numel() for p in modell.parameters()))
