"""
trana.py - tränar MCA-5 och sparar den bästa versionen.

Kör:
    python trana.py                    # 15 epoker, data i "data"
    python trana.py --epoker 30
    python trana.py --data data_syntetisk --namn mca5_syntetisk

En "epok" = modellen har sett varje träningsbild en gång.
Efter varje epok testas modellen på testbilderna (som den aldrig tränar på).
Den version som får bäst testresultat sparas i modeller/<namn>.pth
tillsammans med kategorinamnen, så att webbsidan vet vad svaren betyder.
En logg sparas också i modeller/<namn>_logg.csv (bra för grafer i rapporten).
"""

import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn

from data_laddning import STORLEK, skapa_dataloaders
from modell import MCA5


def kör_en_epok(modell, loader, enhet, förlustfunktion, optimerare=None):
    """Tränar om optimerare ges, annars bara testar. Returnerar (förlust, träffsäkerhet)."""
    tränar = optimerare is not None
    modell.train(tränar)
    total_förlust, rätt, antal = 0.0, 0, 0

    with torch.set_grad_enabled(tränar):
        for bilder, etiketter in loader:
            bilder, etiketter = bilder.to(enhet), etiketter.to(enhet)
            utdata = modell(bilder)
            förlust = förlustfunktion(utdata, etiketter)

            if tränar:
                optimerare.zero_grad()  # nollställ gamla gradienter
                förlust.backward()      # backpropagation: räkna ut hur varje vikt ska ändras
                optimerare.step()       # ändra vikterna lite

            total_förlust += förlust.item() * bilder.size(0)
            rätt += (utdata.argmax(1) == etiketter).sum().item()
            antal += bilder.size(0)

    return total_förlust / antal, rätt / antal


def main():
    parser = argparse.ArgumentParser(description="Träna MCA-5")
    parser.add_argument("--data", default="data")
    parser.add_argument("--epoker", type=int, default=15)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001, help="inlärningshastighet")
    parser.add_argument("--namn", default="mca5")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    enhet = "cuda" if torch.cuda.is_available() else "cpu"
    print("Tränar på:", torch.cuda.get_device_name(0) if enhet == "cuda" else "CPU (långsamt!)")

    train_loader, test_loader, kategorier = skapa_dataloaders(args.data, args.batch, args.workers)
    print(f"Former: {kategorier}")
    print(f"{len(train_loader.dataset)} träningsbilder, {len(test_loader.dataset)} testbilder\n")

    modell = MCA5(len(kategorier), STORLEK).to(enhet)
    förlustfunktion = nn.CrossEntropyLoss()
    optimerare = torch.optim.Adam(modell.parameters(), lr=args.lr)

    Path("modeller").mkdir(exist_ok=True)
    modellfil = Path("modeller") / f"{args.namn}.pth"
    loggfil = Path("modeller") / f"{args.namn}_logg.csv"
    bäst = 0.0

    with open(loggfil, "w", newline="") as f:
        logg = csv.writer(f)
        logg.writerow(["epok", "train_förlust", "train_träff", "test_förlust", "test_träff"])

        for epok in range(1, args.epoker + 1):
            start = time.time()
            tr_f, tr_t = kör_en_epok(modell, train_loader, enhet, förlustfunktion, optimerare)
            te_f, te_t = kör_en_epok(modell, test_loader, enhet, förlustfunktion)
            logg.writerow([epok, f"{tr_f:.4f}", f"{tr_t:.4f}", f"{te_f:.4f}", f"{te_t:.4f}"])

            markering = ""
            if te_t > bäst:
                bäst = te_t
                torch.save({"vikter": modell.state_dict(),
                            "kategorier": kategorier,
                            "storlek": STORLEK}, modellfil)
                markering = "  <- sparad (bäst hittills)"

            print(f"Epok {epok:2d}/{args.epoker} | train {tr_t:6.1%} | test {te_t:6.1%} "
                  f"| {time.time() - start:4.1f} s{markering}")

    print(f"\nKlart! Bästa testträffsäkerhet: {bäst:.1%}")
    print(f"Modell: {modellfil}\nLogg:   {loggfil}")


if __name__ == "__main__":
    main()
