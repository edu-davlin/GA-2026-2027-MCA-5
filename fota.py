"""
fota.py - fotografera riktiga handritade former med webbkameran och spara
dem direkt i rätt dataset-mapp.

Installera OpenCV först (en gång):
    pip install opencv-python

Kör:
    python fota.py --data foton_1 --del train
    python fota.py --data foton_1 --del test

Ett fönster med kamerabilden öppnas. Håll upp papperet och tryck:

    1 = cirkel (circle)
    2 = fyrkant (square)
    3 = triangel (triangle)
    0 = inget papper / ingen form (ingen) - fota bara bakgrunden
    Z = ångra senaste bilden
    Q = avsluta

PAPPERSIGENKÄNNING (se papper.py):
  Blå ram  = papperet hittades. Det klipps ut och rätas ut automatiskt,
             så bakgrunden försvinner.
  Grön ram = inget papper hittades. Då sparas det som är i den gröna
             rutan istället. Håll papperet inuti den.
  Uppe till höger visas exakt den bild som sparas.

Två bilder sparas per tryck:
  dataset/<data>/<del>/<form>/<id>.png             - 224x224, används för träning
  dataset/<data>/_original/<del>/<form>/<id>.jpg   - hela kamerabilden

Originalen gör att ni kan beskära på ett annat sätt senare (t.ex. med en
bättre pappersigenkänning) utan att fota om allt.

TRAIN OCH TEST - VIKTIGT:
Fotografera test-bilderna på ANDRA papper än train-bilderna (t.ex. lägg
undan var femte papper till test). Annars har MCA-5 redan sett exakt samma
teckning under träningen, och testresultatet blir för bra.
Variera gärna ljus, bakgrund och vinkel mellan sessionerna.
"""

import argparse
from datetime import datetime
from pathlib import Path

import cv2

from papper import BILDSTORLEK, förbehandla, mittruta

FORMER = {ord("1"): "circle", ord("2"): "square", ord("3"): "triangle", ord("0"): "ingen"}
NAMN = {"circle": "cirkel", "square": "fyrkant", "triangle": "triangel", "ingen": "ingen"}


def text(bild, s, pos, storlek=0.7):
    cv2.putText(bild, s, pos, cv2.FONT_HERSHEY_SIMPLEX, storlek, (0, 0, 0), 4)
    cv2.putText(bild, s, pos, cv2.FONT_HERSHEY_SIMPLEX, storlek, (255, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser(description="Fotografera former åt MCA-5")
    parser.add_argument("--data", required=True, help="datasetets namn, t.ex. foton_1")
    parser.add_argument("--del", dest="del_", required=True, choices=["train", "test"])
    parser.add_argument("--kamera", type=int, default=0, help="kamerans nummer (0 = första)")
    parser.add_argument("--ruta", type=float, default=0.8,
                        help="hur stor den gröna rutan är (0.3-1.0 av bildhöjden)")
    args = parser.parse_args()

    bas = Path("dataset") / args.data
    ut = bas / args.del_
    orig = bas / "_original" / args.del_
    # Formmapparna skapas först när första bilden av formen sparas,
    # så att inga tomma mappar stör träningen.

    # Sessions-id i filnamnet visar vilka bilder som togs vid samma tillfälle
    session = datetime.now().strftime("%Y%m%d-%H%M")

    kamera = cv2.VideoCapture(args.kamera, cv2.CAP_DSHOW)  # CAP_DSHOW = snabbare start på Windows
    if not kamera.isOpened():
        kamera = cv2.VideoCapture(args.kamera)
    if not kamera.isOpened():
        raise SystemExit(f"Hittar ingen kamera med nummer {args.kamera}. Testa --kamera 1.")
    kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    antal = {f: len(list((ut / f).glob("*.png"))) if (ut / f).exists() else 0
             for f in FORMER.values()}
    senaste = []   # sparade filer, för ångra
    blixt = 0      # kort vit blink när en bild sparats
    print(f"Sparar i {ut}  |  1=cirkel 2=fyrkant 3=triangel 0=ingen  Z=ångra  Q=avsluta")

    while True:
        ok, bild = kamera.read()
        if not ok:
            print("Kunde inte läsa från kameran.")
            break

        utsnitt, hörn = förbehandla(bild, args.ruta)

        # Visning: blå ram runt papperet, annars grön mittruta
        visning = bild.copy()
        if hörn is not None:
            färg = (255, 255, 255) if blixt else (255, 140, 0)
            cv2.polylines(visning, [hörn.astype(int)], True, färg, 4)
            läge = "PAPPER HITTAT"
        else:
            färg = (255, 255, 255) if blixt else (0, 255, 0)
            x, y, sida = mittruta(bild, args.ruta)
            cv2.rectangle(visning, (x, y), (x + sida, y + sida), färg, 3)
            läge = "inget papper - anvander grona rutan"
        blixt = max(0, blixt - 1)

        # Förhandsvisning av exakt den bild som sparas, uppe till höger
        h, w = visning.shape[:2]
        visning[10:10 + BILDSTORLEK, w - BILDSTORLEK - 10:w - 10] = utsnitt
        cv2.rectangle(visning, (w - BILDSTORLEK - 10, 10), (w - 10, 10 + BILDSTORLEK), (255, 255, 255), 2)

        räknare = "   ".join(f"{NAMN[f]} {antal[f]}" for f in FORMER.values())
        text(visning, f"{args.data}/{args.del_}   {räknare}", (15, 35))
        text(visning, läge, (15, 70))
        text(visning, "1=cirkel 2=fyrkant 3=triangel 0=ingen  Z=angra  Q=avsluta",
             (15, h - 20), 0.6)
        cv2.imshow("MCA-5 fotografering", visning)

        tangent = cv2.waitKey(1) & 0xFF
        if tangent in (ord("q"), ord("Q"), 27):  # Q eller Esc
            break

        if tangent in FORMER:
            form = FORMER[tangent]
            nr = antal[form] + 1
            while (ut / form / f"{session}_{nr:05d}.png").exists():
                nr += 1
            namn = f"{session}_{nr:05d}"
            fil = ut / form / f"{namn}.png"
            orig_fil = orig / form / f"{namn}.jpg"
            fil.parent.mkdir(parents=True, exist_ok=True)
            orig_fil.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(fil), utsnitt)
            cv2.imwrite(str(orig_fil), bild, [cv2.IMWRITE_JPEG_QUALITY, 92])
            antal[form] += 1
            senaste.append((form, fil, orig_fil))
            blixt = 3
            print(f"Sparade {fil}  ({'papper' if hörn is not None else 'grön ruta'})")

        if tangent in (ord("z"), ord("Z")) and senaste:
            form, fil, orig_fil = senaste.pop()
            fil.unlink(missing_ok=True)
            orig_fil.unlink(missing_ok=True)
            antal[form] -= 1
            print(f"Ångrade {fil}")

    kamera.release()
    cv2.destroyAllWindows()
    print("\nKlart! " + ", ".join(f"{NAMN[f]} {antal[f]}" for f in FORMER.values()))


if __name__ == "__main__":
    main()
