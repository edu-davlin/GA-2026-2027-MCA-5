"""
trana.py - tränar MCA-5 och sparar den bästa versionen.

Kör:
    python trana.py                    # 15 epoker, senaste datasetet
    python trana.py --data syntetisk_V1 --epoker 30
    python trana.py --fortsatt mca5_V1 --data foton_V1   # finjustera en befintlig modell

--fortsatt (finjustering): istället för att börja från noll laddas vikterna
från en tidigare modell, som sedan tränas vidare på (ny) data. Resultatet
sparas som en NY modellmapp, så originalet är orört. Den nya modellen kan
i sin tur finjusteras igen, hur många gånger som helst. I info.txt i
mappen står vilken modell den byggde vidare på.

Varje träning får en egen mapp med datum och tid som namn:
    modeller/2026-11-11_21-05/
        modell.pth     - den tränade modellen
        logg.csv       - träffsäkerhet per epok
utvardera.py och grafer.py lägger sedan sina resultat i samma mapp.
Döp om mappen själv efteråt om du vill (t.ex. till mca5_robust).

En "epok" = modellen har sett varje träningsbild en gång.
Efter varje epok testas modellen på testbilderna (som den aldrig tränar på).
Den version som får bäst testresultat sparas i modell.pth
tillsammans med kategorinamnen, så att webbsidan vet vad svaren betyder.
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from data_laddning import STORLEK, hitta_dataset, skapa_dataloaders
from modell import MCA5, hitta_modellmapp


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
    parser.add_argument("--data", default=None, help="datasetets namn i dataset/ (standard: senaste)")
    parser.add_argument("--epoker", type=int, default=15)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None,
                        help="inlärningshastighet (standard: 0.001, eller 0.0003 vid --fortsatt)")
    parser.add_argument("--fortsatt", default=None, metavar="MODELLMAPP",
                        help="fortsätt träna (finjustera) en befintlig modell, t.ex. mca5_V1")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    enhet = "cuda" if torch.cuda.is_available() else "cpu"
    print("Tränar på:", torch.cuda.get_device_name(0) if enhet == "cuda" else "CPU (långsamt!)")

    dataset = hitta_dataset(args.data)
    print("Dataset:", dataset)
    train_loader, test_loader, kategorier = skapa_dataloaders(dataset, args.batch, args.workers)
    print(f"Former: {kategorier}")
    print(f"{len(train_loader.dataset)} träningsbilder, {len(test_loader.dataset)} testbilder\n")

    modell = MCA5(len(kategorier), STORLEK).to(enhet)

    grund = None
    if args.fortsatt:
        grund = hitta_modellmapp(args.fortsatt)
        sparat = torch.load(grund / "modell.pth", map_location=enhet)
        if sparat["kategorier"] != kategorier:
            raise ValueError(
                f"Formerna matchar inte!\n  {grund.name}: {sparat['kategorier']}\n"
                f"  datan:   {kategorier}\nFinjustering kräver samma former.")
        modell.load_state_dict(sparat["vikter"])
        print(f"Fortsätter från: {grund}\n")

    # Lägre inlärningshastighet vid finjustering, så att det modellen redan
    # kan inte skrivs över för snabbt.
    lr = args.lr if args.lr is not None else (0.0003 if grund else 0.001)
    förlustfunktion = nn.CrossEntropyLoss()
    optimerare = torch.optim.Adam(modell.parameters(), lr=lr)

    namn = datetime.now().strftime("%Y-%m-%d_%H-%M")  # t.ex. 2026-11-11_21-05
    mapp = Path("modeller") / namn
    mapp.mkdir(parents=True, exist_ok=True)
    modellfil = mapp / "modell.pth"
    loggfil = mapp / "logg.csv"

    # Beskrivning av träningen, så att man vet hur versionen skapades
    (mapp / "info.txt").write_text(
        f"Skapad:           {datetime.now():%Y-%m-%d %H:%M}\n"
        f"Byggde vidare på: {grund.name if grund else '- (tränad från noll)'}\n"
        f"Dataset:          {dataset.name}\n"
        f"Former:           {', '.join(kategorier)}\n"
        f"Epoker:           {args.epoker}\n"
        f"Inlärningshast.:  {lr}\n"
        f"Bilder:           {len(train_loader.dataset)} train, {len(test_loader.dataset)} test\n",
        encoding="utf-8")
    bäst = 0.0

    with open(loggfil, "w", newline="", encoding="utf-8-sig") as f:
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
                            "storlek": STORLEK,
                            "dataset": str(dataset)}, modellfil)
                markering = "  <- sparad (bäst hittills)"

            print(f"Epok {epok:2d}/{args.epoker} | train {tr_t:6.1%} | test {te_t:6.1%} "
                  f"| {time.time() - start:4.1f} s{markering}")

    print(f"\nKlart! Bästa testträffsäkerhet: {bäst:.1%}")
    print(f"Sparat i mappen: {mapp}")


if __name__ == "__main__":
    main()
