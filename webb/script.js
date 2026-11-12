// script.js - logiken för MCA-5:s webbsida (ESP-stil)
//
// 1. Startar kameran.
// 2. Skickar en bild åt gången till webbsida.py (/gissa).
// 3. Ritar en ram runt papperet med formens namn och en mätare som visar
//    hur säker MCA-5 är, ungefär som ESP i ett spel.

const NAMN = { circle: "Cirkel", square: "Fyrkant", triangle: "Triangel", ingen: "Inget papper" };
const GRÄNS = 0.6;          // under detta visas "?" istället för en form
const MELLANRUM_MS = 250;   // väntetid mellan bilderna som skickas

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const modellval = document.getElementById("modell");
const meddelande = document.getElementById("meddelande");
const fångst = document.createElement("canvas");

let senaste = null;   // senaste svaret från servern
let ram = null;       // ramen som ritas just nu (glider mjukt mot målet)

// ---------- Modeller ----------
async function laddaModeller() {
  const lista = await (await fetch("/modeller")).json();
  modellval.innerHTML = lista.map(m => `<option>${m}</option>`).join("");
  if (!lista.length) visaMeddelande("Inga modeller hittades.<br>Kör <b>trana.py</b> först.");
}

// ---------- Kamera ----------
async function startaKamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    visaMeddelande("Kameran fungerar bara på <b>http://localhost:5000</b>.");
    return;
  }
  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false
    });
    await video.play();
    skickaLoop();
    requestAnimationFrame(rita);
  } catch (e) {
    visaMeddelande("Kunde inte starta kameran.<br>Tillåt kameran i webbläsaren och ladda om sidan.");
  }
}

function visaMeddelande(html) {
  meddelande.innerHTML = html;
  meddelande.style.display = "block";
}

// ---------- Kontakt med servern ----------
async function skickaLoop() {
  while (true) {
    try {
      fångst.width = video.videoWidth;
      fångst.height = video.videoHeight;
      fångst.getContext("2d").drawImage(video, 0, 0);
      const svar = await fetch("/gissa", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ bild: fångst.toDataURL("image/jpeg", 0.85), modell: modellval.value })
      });
      const d = await svar.json();
      if (!d.fel) senaste = d;
    } catch { /* servern svarade inte, försök igen */ }
    await new Promise(r => setTimeout(r, MELLANRUM_MS));
  }
}

// ---------- Ritning ----------

// Videon är "object-fit: cover": räkna om en punkt i kamerabilden (0-1)
// till pixlar på skärmen.
function tillSkärm(x, y) {
  const vw = video.videoWidth, vh = video.videoHeight;
  const skala = Math.max(overlay.width / vw, overlay.height / vh);
  const ox = (overlay.width - vw * skala) / 2, oy = (overlay.height - vh * skala) / 2;
  return [ox + x * vw * skala, oy + y * vh * skala];
}

// Målramen: runt papperet, eller den gröna mittrutan om inget papper hittades
function målram(d) {
  let punkter;
  if (d.papper) {
    punkter = d.hörn;
  } else {
    const vw = video.videoWidth, vh = video.videoHeight, sida = Math.min(vw, vh) * 0.8;
    const x0 = (vw - sida) / 2 / vw, y0 = (vh - sida) / 2 / vh;
    punkter = [[x0, y0], [x0 + sida / vw, y0 + sida / vh]];
  }
  const sp = punkter.map(([x, y]) => tillSkärm(x, y));
  const xs = sp.map(p => p[0]), ys = sp.map(p => p[1]);
  return { x: Math.min(...xs), y: Math.min(...ys), w: Math.max(...xs) - Math.min(...xs), h: Math.max(...ys) - Math.min(...ys) };
}

function mätarfärg(p) {
  if (p >= 0.8) return "#3ddc5a";
  if (p >= GRÄNS) return "#e8c53a";
  return "#e5533a";
}

function rita() {
  overlay.width = window.innerWidth;
  overlay.height = window.innerHeight;
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  if (senaste && video.videoWidth) {
    const par = Object.entries(senaste.sannolikheter).sort((a, b) => b[1] - a[1]);
    const [form, p] = par[0];

    if (form !== "ingen") {
      // Låt ramen glida mjukt mot sin nya plats
      const mål = målram(senaste);
      if (!ram) ram = { ...mål };
      for (const k of ["x", "y", "w", "h"]) ram[k] += (mål[k] - ram[k]) * 0.35;
      ritaESP(ram, p >= GRÄNS ? (NAMN[form] || form) : "?", p);
    } else {
      ram = null;
    }
  }
  requestAnimationFrame(rita);
}

function ritaESP(r, text, p) {
  const orange = "#ff7a2f";

  // Ramen
  ctx.lineWidth = 2;
  ctx.strokeStyle = orange;
  ctx.shadowColor = "rgba(0,0,0,0.8)";
  ctx.shadowBlur = 4;
  ctx.strokeRect(r.x, r.y, r.w, r.h);

  // Mätaren ovanför ramen (som en hälsomätare)
  const mh = 6, my = r.y - mh - 26;
  ctx.shadowBlur = 0;
  ctx.fillStyle = "rgba(0,0,0,0.7)";
  ctx.fillRect(r.x - 1, my - 1, r.w + 2, mh + 2);
  ctx.fillStyle = mätarfärg(p);
  ctx.fillRect(r.x, my, r.w * p, mh);

  // Formens namn och säkerhet mellan mätaren och ramen
  ctx.font = "600 16px 'Segoe UI', Tahoma, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.shadowColor = "rgba(0,0,0,0.9)";
  ctx.shadowBlur = 4;
  ctx.fillStyle = orange;
  ctx.fillText(`${text}  ${Math.round(p * 100)}%`, r.x + r.w / 2, r.y - 5);
  ctx.shadowBlur = 0;
}

laddaModeller().then(startaKamera);
