"""
generera_data.py - skapar syntetiska träningsbilder åt MCA-5.

Ritar former som ser ungefär handritade ut (lite darriga streck) med
slumpad storlek, position, rotation, linjetjocklek, pappersfärg och brus.
Bilderna sparas som 224x224 PNG i samma mappstruktur som era riktiga
kamerabilder ska ha, så resten av koden fungerar likadant med båda:

    data/train/<form>/00000.png
    data/test/<form>/00000.png

Kör:
    python generera_data.py                 # 1000 train + 200 test per form
    python generera_data.py --train 3000 --test 500
    python generera_data.py --ut data_syntetisk   # annan mapp

Samma --seed ger exakt samma bilder varje gång (bra för att kunna
upprepa experiment i GA:t).
"""

import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

BILDSTORLEK = 224

# Formerna MCA-5 ska känna igen. Mappnamnen blir kategorierna.
FORMER = ["circle", "square", "triangle", "star",
          "hexagon", "pentagon", "line", "zigzag"]


# ---------------------------------------------------------------------------
# Hjälpfunktioner för att få strecken att se handritade ut
# ---------------------------------------------------------------------------

def forta_punkter(punkter, steg=6):
    """Lägger till fler punkter längs varje kant så att vi kan 'darra' dem."""
    ut = []
    for (x1, y1), (x2, y2) in zip(punkter[:-1], punkter[1:]):
        langd = math.hypot(x2 - x1, y2 - y1)
        n = max(2, int(langd / steg))
        for i in range(n):
            t = i / n
            ut.append((x1 + (x2 - x1) * t, y1 + (y2 - y1) * t))
    ut.append(punkter[-1])
    return ut


def darra(punkter, styrka):
    """Flyttar varje punkt lite slumpmässigt, utjämnat så att det ser ut
    som en handrörelse istället för taggigt brus."""
    n = len(punkter)
    brus = np.random.normal(0, styrka, size=(n, 2))
    # Glidande medelvärde = mjukare darrning
    kärna = np.ones(5) / 5
    brus[:, 0] = np.convolve(brus[:, 0], kärna, mode="same")
    brus[:, 1] = np.convolve(brus[:, 1], kärna, mode="same")
    return [(x + dx, y + dy) for (x, y), (dx, dy) in zip(punkter, brus)]


def rotera(punkter, vinkel, cx, cy):
    c, s = math.cos(vinkel), math.sin(vinkel)
    return [(cx + (x - cx) * c - (y - cy) * s,
             cy + (x - cx) * s + (y - cy) * c) for x, y in punkter]


# ---------------------------------------------------------------------------
# En funktion per form. Alla returnerar en lista med punkter som ska
# förbindas med ett streck, centrerade kring (cx, cy) med "radien" r.
# ---------------------------------------------------------------------------

def regelbunden_polygon(cx, cy, r, hörn, startvinkel=-math.pi / 2):
    p = [(cx + r * math.cos(startvinkel + 2 * math.pi * i / hörn),
          cy + r * math.sin(startvinkel + 2 * math.pi * i / hörn))
         for i in range(hörn)]
    return p + [p[0]]  # stäng formen


def form_circle(cx, cy, r):
    # Lite oval ibland, precis som när man ritar för hand
    ry = r * random.uniform(0.85, 1.15)
    n = 60
    p = [(cx + r * math.cos(2 * math.pi * i / n),
          cy + ry * math.sin(2 * math.pi * i / n)) for i in range(n)]
    return p + [p[0]]


def form_square(cx, cy, r):
    return regelbunden_polygon(cx, cy, r, 4, startvinkel=math.pi / 4)


def form_triangle(cx, cy, r):
    return regelbunden_polygon(cx, cy, r, 3)


def form_pentagon(cx, cy, r):
    return regelbunden_polygon(cx, cy, r, 5)


def form_hexagon(cx, cy, r):
    return regelbunden_polygon(cx, cy, r, 6)


def form_star(cx, cy, r):
    inre = r * random.uniform(0.35, 0.5)
    p = []
    for i in range(10):
        radie = r if i % 2 == 0 else inre
        v = -math.pi / 2 + math.pi * i / 5
        p.append((cx + radie * math.cos(v), cy + radie * math.sin(v)))
    return p + [p[0]]


def form_line(cx, cy, r):
    return [(cx - r, cy), (cx + r, cy)]


def form_zigzag(cx, cy, r):
    toppar = random.randint(3, 6)
    höjd = r * random.uniform(0.3, 0.6)
    p = []
    for i in range(toppar * 2 + 1):
        x = cx - r + 2 * r * i / (toppar * 2)
        y = cy + (höjd if i % 2 else -höjd)
        p.append((x, y))
    return p


RITFUNKTIONER = {namn: globals()[f"form_{namn}"] for namn in FORMER}


# ---------------------------------------------------------------------------
# Skapa en hel bild
# ---------------------------------------------------------------------------

def skapa_bild(form):
    # Pappret: vitt till lätt grått/gulaktigt, som riktigt papper i olika ljus
    ljus = random.randint(200, 255)
    bakgrund = tuple(int(np.clip(ljus + random.randint(-15, 10), 0, 255))
                     for _ in range(3))
    bild = Image.new("RGB", (BILDSTORLEK, BILDSTORLEK), bakgrund)
    rita = ImageDraw.Draw(bild)

    # Pennan: mörk färg (svart, blå eller grå)
    penna = random.choice([(20, 20, 20), (30, 30, 90), (60, 60, 60)])
    penna = tuple(int(np.clip(c + random.randint(-15, 15), 0, 255)) for c in penna)

    # Storlek, position och rotation
    r = random.uniform(40, 90)
    marg = r + 10
    cx = random.uniform(marg, BILDSTORLEK - marg)
    cy = random.uniform(marg, BILDSTORLEK - marg)
    vinkel = random.uniform(0, 2 * math.pi)

    punkter = RITFUNKTIONER[form](cx, cy, r)
    punkter = rotera(punkter, vinkel, cx, cy)
    punkter = darra(forta_punkter(punkter), styrka=random.uniform(0.3, 1.8))

    tjocklek = random.randint(2, 7)
    rita.line(punkter, fill=penna, width=tjocklek, joint="curve")

    # Lite kamerabrus och ibland lätt oskärpa
    arr = np.asarray(bild).astype(np.int16)
    arr += np.random.normal(0, random.uniform(0, 8), arr.shape).astype(np.int16)
    bild = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if random.random() < 0.3:
        bild = bild.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.2)))
    return bild


def main():
    parser = argparse.ArgumentParser(description="Skapa syntetisk data åt MCA-5")
    parser.add_argument("--train", type=int, default=1000, help="bilder per form i train")
    parser.add_argument("--test", type=int, default=200, help="bilder per form i test")
    parser.add_argument("--ut", default="data", help="mapp att spara i")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    for del_namn, antal in [("train", args.train), ("test", args.test)]:
        for form in FORMER:
            mapp = Path(args.ut) / del_namn / form
            mapp.mkdir(parents=True, exist_ok=True)
            for i in range(antal):
                skapa_bild(form).save(mapp / f"syn_{i:05d}.png")
            print(f"{del_namn:5s} {form:9s} {antal} bilder -> {mapp}")

    print("Klart!")


if __name__ == "__main__":
    main()
