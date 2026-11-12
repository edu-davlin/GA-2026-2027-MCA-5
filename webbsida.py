"""
webbsida.py - webbsidan för MCA-5. Håll upp en ritad form framför kameran
så säger MCA-5 vilken form det är.

Installera Flask först (en gång):
    pip install flask

Starta:
    python webbsida.py
Öppna sedan http://localhost:5000 i webbläsaren.

Sidan går bara att nå från din egen dator (127.0.0.1).

Filer för själva sidan ligger i mappen webb/:
    index.html - sidans innehåll
    style.css  - utseendet
    script.js  - kameran och kontakten med servern

Hur det fungerar:
  1. Webbsidan (webb/script.js) tar en bild från kameran ungefär
     två gånger per sekund och skickar den hit.
  2. papper.py hittar papperet och klipper ut det, precis som i fota.py.
  3. MCA-5 räknar ut hur sannolik varje form är.
  4. Svaret skickas tillbaka och visas på sidan.
Allt körs på din dator. Bilderna sparas inte.
"""

import argparse
import base64
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

from data_laddning import test_transform
from modell import MCA5
from papper import förbehandla

app = Flask(__name__, static_folder="webb", static_url_path="")
enhet = "cuda" if torch.cuda.is_available() else "cpu"
laddade = {}  # modellnamn -> (modell, kategorier), så att de bara laddas en gång


def tillgängliga_modeller():
    """Alla mappar i modeller/ som innehåller en modell.pth, nyast först."""
    mappar = [m for m in Path("modeller").glob("*") if (m / "modell.pth").exists()]
    return [m.name for m in sorted(mappar, key=lambda m: (m / "modell.pth").stat().st_mtime, reverse=True)]


def hämta_modell(namn):
    if namn not in laddade:
        sparat = torch.load(Path("modeller") / namn / "modell.pth", map_location=enhet)
        modell = MCA5(len(sparat["kategorier"]), sparat["storlek"]).to(enhet)
        modell.load_state_dict(sparat["vikter"])
        modell.eval()
        laddade[namn] = (modell, sparat["kategorier"])
        print(f"Laddade modell: {namn} {sparat['kategorier']}")
    return laddade[namn]


@app.get("/")
def startsida():
    return send_from_directory("webb", "index.html")


@app.get("/modeller")
def lista_modeller():
    return jsonify(tillgängliga_modeller())


@app.post("/gissa")
@torch.no_grad()
def gissa():
    data = request.get_json()
    namn = data.get("modell") or (tillgängliga_modeller() or [None])[0]
    if namn is None:
        return jsonify({"fel": "Ingen modell i modeller/. Kör trana.py först."}), 400
    if namn not in tillgängliga_modeller():
        return jsonify({"fel": f"Okänd modell: {namn}"}), 400

    # Avkoda bilden som webbläsaren skickade (data:image/jpeg;base64,...)
    rå = base64.b64decode(data["bild"].split(",", 1)[1])
    bild = cv2.imdecode(np.frombuffer(rå, np.uint8), cv2.IMREAD_COLOR)
    if bild is None:
        return jsonify({"fel": "Kunde inte läsa bilden"}), 400

    # Samma förbehandling som när datan fotograferades
    utsnitt, hörn = förbehandla(bild)
    pil = Image.fromarray(cv2.cvtColor(utsnitt, cv2.COLOR_BGR2RGB))
    tensor = test_transform(pil).unsqueeze(0).to(enhet)

    modell, kategorier = hämta_modell(namn)
    sannolikheter = torch.softmax(modell(tensor), dim=1)[0].tolist()

    h, w = bild.shape[:2]
    _, png = cv2.imencode(".png", utsnitt)
    return jsonify({
        "modell": namn,
        "sannolikheter": dict(zip(kategorier, sannolikheter)),
        "papper": hörn is not None,
        # Hörnen som andelar (0-1) så att sidan kan rita dem oavsett storlek
        "hörn": (hörn / [w, h]).tolist() if hörn is not None else None,
        "utsnitt": "data:image/png;base64," + base64.b64encode(png).decode(),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Starta MCA-5:s webbsida")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    modeller = tillgängliga_modeller()
    if not modeller:
        print("VARNING: inga modeller i modeller/. Kör trana.py först.")
    else:
        print(f"Modeller: {', '.join(modeller)}  (kör på {enhet})")
    print(f"\nÖppna http://localhost:{args.port} i webbläsaren\n")
    app.run(host="127.0.0.1", port=args.port, threaded=True)  # bara din egen dator
