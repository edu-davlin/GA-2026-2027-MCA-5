"""
papper.py - hittar papperet i en kamerabild och klipper ut det.

Används av fota.py (när ni fotograferar) och senare av webbsidan, så att
MCA-5 alltid får se bilder som är förbehandlade på EXAKT samma sätt.

Så fungerar det (klassisk bildanalys, ingen AI):
  1. Gör bilden svartvit och lite suddig (tar bort brus).
  2. Delar upp bilden i ljust och mörkt (Otsus metod väljer gränsen själv).
     Ett vitt papper blir då en stor ljus fläck.
  3. Letar upp den största ljusa fläcken med fyra hörn = papperet.
  4. "Rätar ut" papperet (perspektivtransform), som om det fotats rakt
     framifrån, och klipper bort allt runt omkring.
  5. Fyller ut till en kvadrat med papperets färg (så att en cirkel inte
     blir oval) och skalar till 224x224.

Om inget papper hittas används den gröna rutan i mitten istället.
"""

import cv2
import numpy as np

BILDSTORLEK = 224


def sortera_hörn(p):
    """Ordnar fyra punkter som: uppe-vänster, uppe-höger, nere-höger, nere-vänster."""
    p = p.reshape(4, 2).astype(np.float32)
    summa = p.sum(axis=1)
    diff = np.diff(p, axis=1).ravel()
    return np.array([p[np.argmin(summa)], p[np.argmin(diff)],
                     p[np.argmax(summa)], p[np.argmax(diff)]], dtype=np.float32)


def hitta_papper(bild, minsta_andel=0.08):
    """Returnerar papperets fyra hörn (sorterade) eller None om inget hittas.
    minsta_andel: papperet måste täcka minst så stor del av bilden."""
    h, w = bild.shape[:2]
    grå = cv2.cvtColor(bild, cv2.COLOR_BGR2GRAY)
    grå = cv2.GaussianBlur(grå, (7, 7), 0)
    _, ljust = cv2.threshold(grå, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Fyll igen pennstrecken inne i papperet så att det blir en hel fläck
    kärna = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    ljust = cv2.morphologyEx(ljust, cv2.MORPH_CLOSE, kärna)

    konturer, _ = cv2.findContours(ljust, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for k in sorted(konturer, key=cv2.contourArea, reverse=True)[:5]:
        yta = cv2.contourArea(k)
        if yta < minsta_andel * h * w or yta > 0.95 * h * w:
            continue  # för litet, eller hela bilden (då är det inte ett papper)
        omkrets = cv2.arcLength(k, True)
        approx = cv2.approxPolyDP(k, 0.02 * omkrets, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return sortera_hörn(approx)
        # Fyra hörn hittades inte (t.ex. en hand över ett hörn): godta
        # minsta omslutande rektangel om fläcken nästan fyller den.
        rekt = cv2.minAreaRect(k)
        rw, rh = rekt[1]
        if rw * rh > 0 and yta / (rw * rh) > 0.85:
            return sortera_hörn(cv2.boxPoints(rekt))
    return None


def räta_ut(bild, hörn, marginal=0.04):
    """Klipper ut papperet, rätar ut det och gör en kvadratisk 224x224-bild."""
    tl, tr, br, bl = hörn
    bredd = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    höjd = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    mål = np.array([[0, 0], [bredd - 1, 0], [bredd - 1, höjd - 1], [0, höjd - 1]], dtype=np.float32)
    papper = cv2.warpPerspective(bild, cv2.getPerspectiveTransform(hörn, mål), (bredd, höjd))

    # Skär bort en smal kant (papperskanten och lite bakgrund)
    mx, my = int(bredd * marginal), int(höjd * marginal)
    papper = papper[my:höjd - my, mx:bredd - mx]

    # Fyll ut till kvadrat med papperets egen färg, så att formen inte förvrängs
    ph, pw = papper.shape[:2]
    sida = max(ph, pw)
    färg = np.median(np.concatenate([papper[0], papper[-1], papper[:, 0], papper[:, -1]]), axis=0)
    kvadrat = np.full((sida, sida, 3), färg, dtype=np.uint8)
    y0, x0 = (sida - ph) // 2, (sida - pw) // 2
    kvadrat[y0:y0 + ph, x0:x0 + pw] = papper
    return cv2.resize(kvadrat, (BILDSTORLEK, BILDSTORLEK), interpolation=cv2.INTER_AREA)


def mittruta(bild, andel=0.8):
    """Den gröna rutan: (x, y, sida) för en centrerad kvadrat."""
    h, w = bild.shape[:2]
    sida = int(min(h, w) * andel)
    return (w - sida) // 2, (h - sida) // 2, sida


def förbehandla(bild, andel=0.8):
    """Huvudfunktionen. Returnerar (bild_224, hörn_eller_None).
    Hittas papperet klipps det ut, annars används den gröna rutan."""
    hörn = hitta_papper(bild)
    if hörn is not None:
        return räta_ut(bild, hörn), hörn
    x, y, sida = mittruta(bild, andel)
    utsnitt = bild[y:y + sida, x:x + sida]
    return cv2.resize(utsnitt, (BILDSTORLEK, BILDSTORLEK), interpolation=cv2.INTER_AREA), None
