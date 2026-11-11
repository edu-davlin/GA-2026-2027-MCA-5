"""
utvardera.py - mäter hur bra MCA-5 klarar sig, och hur suddighet och
ljus påverkar den. Det här är kärnan i GA-undersökningen.

Kör (efter trana.py):
    python utvardera.py                       # senast tränade modellen
    python utvardera.py --modell mca5_robust  # en viss modellmapp
    python utvardera.py --data foton_V1       # testa på ett annat dataset

Utan --data används testbilderna från datasetet modellen tränades på.

Gör tre saker:
  1. Träffsäkerhet per form på de vanliga testbilderna.
  2. Förväxlingsmatris: vilka former blandas ihop?
  3. Robusthetstest: samma testbilder med ökande suddighet och med
     för mörkt/för ljust ljus. Resultatet sparas i
     robusthet.csv i modellens mapp, så att ni kan göra grafer i rapporten.

Samma störningsnivåer kan sedan användas på bilderna ni skickar till
ChatGPT och Claude, så att jämförelsen blir rättvis.
"""

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import datasets

from data_laddning import hitta_dataset, test_transform
from modell import MCA5, hitta_modellmapp

# Störningsnivåer. 0 / 1.0 = ingen störning.
SUDDIGHET = [0, 1, 2, 3, 4, 6, 8]                   # Gaussisk oskärpa, radie i pixlar (på 224x224)
LJUS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0]     # 1.0 = original, <1 mörkare, >1 ljusare


def suddig(radie):
    return lambda b: b.filter(ImageFilter.GaussianBlur(radie)) if radie > 0 else b


def ljus(faktor):
    return lambda b: ImageEnhance.Brightness(b).enhance(faktor)


def ladda_modell(fil, enhet):
    sparat = torch.load(fil, map_location=enhet)
    modell = MCA5(len(sparat["kategorier"]), sparat["storlek"]).to(enhet)
    modell.load_state_dict(sparat["vikter"])
    modell.eval()
    return modell, sparat["kategorier"]


@torch.no_grad()
def testa(modell, sökvägar, enhet, störning=None, batch=128):
    """Returnerar lista med (rätt_index, gissat_index)."""
    resultat = []
    for i in range(0, len(sökvägar), batch):
        del_ = sökvägar[i:i + batch]
        tensorer, etiketter = [], []
        for väg, etikett in del_:
            bild = Image.open(väg).convert("RGB")
            if störning:
                bild = störning(bild)
            tensorer.append(test_transform(bild))
            etiketter.append(etikett)
        gissningar = modell(torch.stack(tensorer).to(enhet)).argmax(1).tolist()
        resultat += list(zip(etiketter, gissningar))
    return resultat


def träff(resultat):
    return sum(r == g for r, g in resultat) / len(resultat)


def main():
    parser = argparse.ArgumentParser(description="Utvärdera MCA-5")
    parser.add_argument("--modell", default=None, help="modellmappens namn (standard: senast tränade)")
    parser.add_argument("--data", default=None, help="dataset att testa på (standard: det modellen tränades på)")
    args = parser.parse_args()

    enhet = "cuda" if torch.cuda.is_available() else "cpu"
    mapp = hitta_modellmapp(args.modell)
    print("Modell:", mapp)
    modell, kategorier = ladda_modell(mapp / "modell.pth", enhet)

    if args.data:
        dataset = hitta_dataset(args.data)
    else:
        tränad_på = torch.load(mapp / "modell.pth", map_location="cpu").get("dataset")
        if tränad_på and Path(tränad_på, "test").exists():
            dataset = Path(tränad_på)
        elif Path("data", "test").exists():  # äldre modeller (t.ex. mca5_V1)
            dataset = Path("data")
        else:
            dataset = hitta_dataset()
    print("Testar på dataset:", dataset)
    sökvägar = datasets.ImageFolder(dataset / "test").samples

    # 1. Träffsäkerhet per form
    res = testa(modell, sökvägar, enhet)
    print(f"Total träffsäkerhet: {träff(res):.1%}\n")
    print("Per form:")
    for i, namn in enumerate(kategorier):
        egna = [(r, g) for r, g in res if r == i]
        if egna:
            print(f"  {namn:10s} {träff(egna):6.1%}  ({len(egna)} bilder)")

    # 2. Förväxlingsmatris (rad = rätt svar, kolumn = MCA-5:s gissning)
    print("\nFörväxlingsmatris (rad = rätt form, kolumn = gissning):")
    print(" " * 11 + "".join(f"{n[:6]:>7s}" for n in kategorier))
    for i, namn in enumerate(kategorier):
        rad = [sum(1 for r, g in res if r == i and g == j) for j in range(len(kategorier))]
        print(f"  {namn:9s}" + "".join(f"{x:7d}" for x in rad))

    # 3. Robusthet mot suddighet och ljus
    resultatfil = mapp / "robusthet.csv"
    with open(resultatfil, "w", newline="", encoding="utf-8-sig") as f:
        skriv = csv.writer(f)
        skriv.writerow(["störning", "nivå", "träffsäkerhet"])

        print("\nSuddighet (radie i pixlar):")
        for r in SUDDIGHET:
            t = träff(testa(modell, sökvägar, enhet, suddig(r)))
            skriv.writerow(["suddighet", r, f"{t:.4f}"])
            print(f"  {r:4}  {t:6.1%}  " + "#" * int(t * 40))

        print("\nLjus (1.0 = original):")
        for l in LJUS:
            t = träff(testa(modell, sökvägar, enhet, ljus(l)))
            skriv.writerow(["ljus", l, f"{t:.4f}"])
            print(f"  {l:4}  {t:6.1%}  " + "#" * int(t * 40))

    print("\nSparat:", resultatfil)


if __name__ == "__main__":
    main()
