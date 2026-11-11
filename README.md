# MCA-5 – formigenkänning (GA 2026/2027)

MCA-5 är ett eget CNN (Convolutional Neural Network) byggt i PyTorch som känner igen handritade former. I GA:t undersöks hur suddighet och ljus påverkar igenkänningen, och MCA-5 jämförs med ChatGPT och Claude.

To run this AI model you have to meet or exceed the [requirements.txt](requirements.txt) list.

## Former
`circle`, `square`, `triangle`, `star`, `hexagon`, `pentagon`, `line`, `zigzag`

## Filer

| Fil | Vad den gör |
|---|---|
| `generera_data.py` | Skapar syntetiska "handritade" träningsbilder (224x224 PNG) med slumpad storlek, rotation, linjetjocklek, pappersfärg, brus och lätt oskärpa. |
| `data_laddning.py` | Läser in bilder från `data/train` och `data/test`, skalar ner till 64x64 och normaliserar. Lägger på data augmentation (rotation, ljus/kontrast) på träningsbilderna. |
| `modell.py` | Själva MCA-5: 3 faltningsblock (Conv → BatchNorm → ReLU → MaxPool) och en klassificerare med dropout. Ca 2,2 miljoner vikter. |
| `trana.py` | Tränar modellen på GPU:n, testar efter varje epok och sparar bästa versionen i `modeller/mca5.pth`, plus en logg i `modeller/mca5_logg.csv`. |
| `utvardera.py` | Träffsäkerhet per form, förväxlingsmatris och robusthetstest mot suddighet och ljus. Sparar `resultat/robusthet.csv`. |

## Kom igång

Aktivera miljön i VS Code-terminalen:
```bash
ai-miljo\Scripts\activate
```

1. **Skapa data** (syntetisk, tills de riktiga kamerabilderna är klara):
   ```bash
   python generera_data.py
   ```
   Ger 1000 träningsbilder och 200 testbilder per form. Fler: `python generera_data.py --train 3000 --test 500`

2. **Kontrollera att datan hittas:**
   ```bash
   python data_laddning.py
   ```

3. **Träna:**
   ```bash
   python trana.py
   ```
   Ska skriva "Tränar på: NVIDIA GeForce RTX 4070". Står det CPU används inte grafikkortet.

4. **Utvärdera:**
   ```bash
   python utvardera.py
   ```

## Mappstruktur för data
```
data/
  train/<form>/*.png
  test/<form>/*.png
```
Mappnamnen blir kategorierna. Riktiga kamerabilder läggs i samma struktur. Dela upp train/test efter **fotograferingstillfälle**, inte slumpmässigt, annars hamnar nästan likadana bilder i båda och resultatet ser bättre ut än det är.

`data/` och `modeller/*.pth` finns i `.gitignore` eftersom de är för stora för GitHub.

## Idé för undersökningen
Träna en version på syntetisk data (`--namn mca5_syntetisk`) och en på riktiga foton, och jämför hur de klarar samma störningsnivåer. Samma nivåer (se `SUDDIGHET` och `LJUS` i `utvardera.py`) används på bilderna som skickas till ChatGPT och Claude.

## Kvar att göra
- [ ] Samla riktiga kamerabilder
- [ ] Webbsida (Flask + kamera) som använder `modeller/mca5.pth`
- [ ] Jämförelse med ChatGPT och Claude
