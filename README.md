# MCA-5 – formigenkänning (GA 2026/2027)

MCA-5 är ett eget CNN (Convolutional Neural Network) byggt i PyTorch som känner igen handritade former. I GA:t undersöks hur suddighet och ljus påverkar igenkänningen, och MCA-5 jämförs med ChatGPT och Claude.

To run this AI model you have to meet or exceed the [requirements.txt](requirements.txt) list.

## Former
Avgränsat till tre former: `circle` (cirkel), `square` (fyrkant) och `triangle` (triangel).

## Filer

| Fil | Vad den gör |
|---|---|
| `generera_data.py` | Skapar ett nytt syntetiskt dataset i `dataset/<datum_tid>/` med "handritade" bilder (224x224 PNG): slumpad storlek, rotation, linjetjocklek, pappersfärg, brus och lätt oskärpa. |
| `data_laddning.py` | Läser in bilder från ett dataset i `dataset/`, skalar ner till 64x64 och normaliserar. Lägger på data augmentation (rotation, ljus/kontrast) på träningsbilderna. |
| `modell.py` | Själva MCA-5: 3 faltningsblock (Conv → BatchNorm → ReLU → MaxPool) och en klassificerare med dropout. Ca 2,2 miljoner vikter. |
| `trana.py` | Tränar modellen på GPU:n, testar efter varje epok och skapar en mapp `modeller/<datum_tid>/` (t.ex. `2026-11-11_21-05`) med bästa versionen `modell.pth`, `logg.csv` och `info.txt` (hur modellen skapades). |
| `utvardera.py` | Träffsäkerhet per form, förväxlingsmatris och robusthetstest mot suddighet och ljus. Använder senast tränade modellen (eller `--modell <mapp>`). Sparar `robusthet.csv` i modellens mapp. |
| `grafer.py` | Gör PNG-grafer till rapporten och sparar dem i modellens mapp. Använder senast utvärderade modellen (eller `--modell <mapp>`). |

## Kom igång

Aktivera miljön i VS Code-terminalen:
```bash
ai-miljo\Scripts\activate
```

1. **Skapa data** (syntetisk, tills de riktiga kamerabilderna är klara):
   ```bash
   python generera_data.py
   ```
   Skapar ett nytt dataset i `dataset/<datum_tid>/` med 1000 träningsbilder och 200 testbilder per form. Fler: `python generera_data.py --train 3000 --test 500`. Döp om mappen fritt, t.ex. till `syntetisk_V1`.

2. **Kontrollera att datan hittas:**
   ```bash
   python data_laddning.py
   ```

3. **Träna:**
   ```bash
   python trana.py                       # senaste datasetet
   python trana.py --data syntetisk_V1   # ett visst dataset
   ```
   Ska skriva "Tränar på: NVIDIA GeForce RTX 4070". Står det CPU används inte grafikkortet.

   Varje träning får en egen mapp med datum och tid som namn. Döp om mappen fritt, t.ex. till `mca5_robust`. Allt för en modell hamnar till slut där:
   ```
   modeller/mca5_robust/
     modell.pth  logg.csv  robusthet.csv  graf_suddighet.png  graf_ljus.png  graf_traning.png
   ```

   **Finjustera** (träna vidare på en befintlig modell, t.ex. med riktiga foton):
   ```bash
   python trana.py --fortsatt mca5_V1 --data foton_V1
   ```
   Laddar `mca5_V1` som startpunkt och sparar resultatet i en ny mapp. Originalet ändras inte, och den nya versionen kan finjusteras igen. `info.txt` i varje mapp visar vilken modell den byggde vidare på.

4. **Utvärdera:**
   ```bash
   python utvardera.py                       # testbilderna från datasetet modellen tränades på
   python utvardera.py --data foton_V1       # testa på ett annat dataset
   ```

5. **Gör grafer:**
   ```bash
   python grafer.py
   ```

## Mappstruktur för data
Varje dataset har en egen mapp i `dataset/`:
```
dataset/
  syntetisk_V1/
    info.txt
    train/<form>/*.png
    test/<form>/*.png
  foton_V1/
    train/<form>/*.png
    test/<form>/*.png
```
Mappnamnen under `train`/`test` blir kategorierna. Riktiga kamerabilder läggs i samma struktur. Dela upp train/test efter **fotograferingstillfälle**, inte slumpmässigt, annars hamnar nästan likadana bilder i båda och resultatet ser bättre ut än det är.

`dataset/` och `.pth`-filerna i `modeller/` finns i `.gitignore` eftersom de är för stora för GitHub.

## Idé för undersökningen
Träna en version på syntetisk data och en på riktiga foton (döp om mapparna, t.ex. `mca5_syntetisk`), och jämför hur de klarar samma störningsnivåer. Samma nivåer (se `SUDDIGHET` och `LJUS` i `utvardera.py`) används på bilderna som skickas till ChatGPT och Claude.

## Kvar att göra
- [ ] Samla riktiga kamerabilder
- [ ] Webbsida (Flask + kamera) som använder modellerna i `modeller/`
- [ ] Jämförelse med ChatGPT och Claude
