"""
grafer.py - gör grafer av resultaten till rapporten.

Kör (efter trana.py och utvardera.py):
    python grafer.py                       # senast utvärderade modellen
    python grafer.py --modell mca5_robust  # en viss modellmapp

Läser robusthet.csv, logg.csv, testinfo.txt och info.txt i modellens mapp
och sparar graferna där. Varje graf får en undertitel med modellens namn,
när den testades/tränades och vilket dataset som användes:
    graf_suddighet.png   - träffsäkerhet vid ökande suddighet
    graf_ljus.png        - träffsäkerhet vid olika ljusnivåer
    graf_traning.png     - hur träffsäkerheten utvecklades under träningen
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from modell import hitta_modellmapp

BLÅ = "#2a78d6"
ORANGE = "#eb6834"
TEXT = "#52514e"
RUTNÄT = "#e4e3df"


def läs_csv(fil):
    """Läser en CSV-fil. Testar flera teckenkodningar eftersom Windows
    ibland sparar å, ä, ö på ett annat sätt."""
    for kodning in ("utf-8-sig", "cp1252"):
        try:
            with open(fil, encoding=kodning) as f:
                rader = list(csv.reader(f))
            return rader[1:]  # hoppa över rubrikraden
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Kunde inte läsa {fil}")


UNDERTITEL = ""  # fylls i main() med modell, testdatum och dataset


def läs_info(fil):
    """Läser en info-fil med rader som 'nyckel: värde'."""
    info = {}
    if fil.exists():
        for rad in fil.read_text(encoding="utf-8").splitlines():
            if ":" in rad:
                nyckel, värde = rad.split(":", 1)
                info[nyckel.strip().lower()] = värde.strip()
    return info


def snygga_till(ax, titel, xetikett, undertitel=None):
    undertitel = UNDERTITEL if undertitel is None else undertitel
    ax.set_title(titel, loc="left", fontsize=13, fontweight="bold", pad=24 if undertitel else 12)
    if undertitel:
        ax.text(0, 1.02, undertitel, transform=ax.transAxes, fontsize=8.5, color=TEXT, va="bottom")
    ax.set_xlabel(xetikett, color=TEXT)
    ax.set_ylabel("Träffsäkerhet (%)", color=TEXT)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", color=RUTNÄT, linewidth=0.8)
    ax.set_axisbelow(True)
    for sida in ("top", "right"):
        ax.spines[sida].set_visible(False)
    for sida in ("left", "bottom"):
        ax.spines[sida].set_color(RUTNÄT)
    ax.tick_params(colors=TEXT)


def linjegraf(x, y, titel, xetikett, fil, referens=None):
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    ax.plot(x, y, color=BLÅ, linewidth=2, marker="o", markersize=6)
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.0f}%", (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=TEXT)
    if referens is not None:
        ax.axvline(referens, color=TEXT, linestyle=":", linewidth=1)
        ax.annotate("original", (referens, 3), ha="center", fontsize=8, color=TEXT,
                    xytext=(0, 0), textcoords="offset points")
    snygga_till(ax, titel, xetikett)
    fig.tight_layout()
    fig.savefig(fil)
    plt.close(fig)
    print("Sparade", fil)


def main():
    parser = argparse.ArgumentParser(description="Gör grafer av MCA-5:s resultat")
    parser.add_argument("--modell", default=None,
                        help="modellmappens namn (standard: senast utvärderade)")
    args = parser.parse_args()

    ut = hitta_modellmapp(args.modell, kräver="robusthet.csv")
    print("Modell:", ut)
    loggfil = ut / "logg.csv"

    # Undertitel: vilken modell, när den testades och på vilket dataset
    global UNDERTITEL
    test = läs_info(ut / "testinfo.txt")
    UNDERTITEL = (f"Modell: {ut.name}   ·   Testad: {test.get('testad', 'okänt')}   ·   "
                  f"Dataset: {test.get('dataset', 'okänt')} ({test.get('testbilder', '?')} testbilder)")

    # Robusthet: dela upp raderna i suddighet och ljus
    rader = läs_csv(ut / "robusthet.csv")
    sudd = [(float(n), float(t) * 100) for s, n, t in rader if s.startswith("sudd")]
    ljus = [(float(n), float(t) * 100) for s, n, t in rader if s.startswith("ljus")]

    linjegraf([n for n, _ in sudd], [t for _, t in sudd],
              "MCA-5 vid ökande suddighet", "Suddighet (radie i pixlar, Gaussisk oskärpa)",
              ut / "graf_suddighet.png")
    linjegraf([n for n, _ in ljus], [t for _, t in ljus],
              "MCA-5 vid olika ljusnivåer", "Ljusstyrka (1.0 = original, <1 mörkare, >1 ljusare)",
              ut / "graf_ljus.png", referens=1.0)

    # Träningskurva: train och test i samma graf (samma skala, 0-100 %)
    if loggfil.exists():
        logg = läs_csv(loggfil)
        epoker = [int(r[0]) for r in logg]
        train = [float(r[2]) * 100 for r in logg]
        test = [float(r[4]) * 100 for r in logg]
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
        ax.plot(epoker, train, color=BLÅ, linewidth=2, marker="o", markersize=5, label="Träningsbilder")
        ax.plot(epoker, test, color=ORANGE, linewidth=2, marker="o", markersize=5, label="Testbilder")
        ax.annotate("Träning", (epoker[-1], train[-1]), xytext=(6, -4), textcoords="offset points", color=TEXT, fontsize=9)
        ax.annotate("Test", (epoker[-1], test[-1]), xytext=(6, 2), textcoords="offset points", color=TEXT, fontsize=9)
        ax.legend(frameon=False, loc="lower right")
        träning = läs_info(ut / "info.txt")
        snygga_till(ax, "Träffsäkerhet under träningen", "Epok",
                    f"Modell: {ut.name}   ·   Tränad: {träning.get('skapad', 'okänt')}   ·   "
                    f"Dataset: {träning.get('dataset', 'okänt')}   ·   "
                    f"Byggde vidare på: {träning.get('byggde vidare på', '-')}")
        ax.set_xticks(epoker)
        fig.tight_layout()
        fig.savefig(ut / "graf_traning.png")
        plt.close(fig)
        print("Sparade", ut / "graf_traning.png")
    else:
        print(f"(Ingen {loggfil}, hoppar över träningsgrafen)")


if __name__ == "__main__":
    main()
