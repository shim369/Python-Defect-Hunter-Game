// 欠陥ハンター フロント。素の JS + Canvas。採点・指標計算は全てサーバー (Python) 側で行う。
"use strict";
const SCALE = 2;          // 表示倍率 (画像は 128x320 と小さいので拡大表示する)
const SUMMARY_EVERY = 5;  // 何問ごとに中間成績を出すか
const $ = (id) => document.getElementById(id);

const state = { view: "title", lastResult: null, cfg: null, questions: [], idx: 0, results: [], threshold: 0.5 };
let W = 0, H = 0;

// ---------- 多言語 (日本語 / English) ----------
// 文言は i18n.json に置く。選択は localStorage に保存し、初回はブラウザの言語で決める。
let STR = {};
const saved = localStorage.getItem("lang");
let lang = saved || ((navigator.language || "").toLowerCase().startsWith("ja") ? "ja" : "en");
const t = (key, vars = {}) => (STR[lang]?.[key] ?? key).replace(/\{(\w+)\}/g, (_, n) => vars[n]);
function applyStatic() {
  document.documentElement.lang = lang;
  document.title = t("doc_title");
  for (const el of document.querySelectorAll("[data-i18n]")) el.textContent = t(el.dataset.i18n);
  for (const el of document.querySelectorAll("[data-i18n-html]")) el.innerHTML = t(el.dataset.i18nHtml);   // 辞書は自前の固定文字列のみ
  for (const b of document.querySelectorAll(".lang-switch button")) b.setAttribute("aria-pressed", String(b.dataset.lang === lang));
}
// 言語を切り替えたら、いま表示中の画面の動的な文言も作り直す
async function setLang(l) {
  lang = l; localStorage.setItem("lang", l); applyStatic();
  if (state.view === "question") renderQuestionText();
  else if (state.view === "result" && state.lastResult) renderResultText(state.lastResult);
  else if (state.view === "summary") await showSummary();
}         // 内部解像度 (幅, 高さ)

async function api(path, body) {
  const res = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" },
                                          body: JSON.stringify(body) } : undefined);
  if (!res.ok) throw new Error(`${path}: ${res.status} ${await res.text()}`);
  return res.json();
}
function show(name) {
  state.view = name;
  for (const s of document.querySelectorAll("main > section")) s.hidden = (s.id !== `screen-${name}`);
}
function loadImage(url) {
  return new Promise((ok, ng) => { const im = new Image(); im.onload = () => ok(im); im.onerror = ng; im.src = url; });
}
function sizeCanvas(c) { c.width = W; c.height = H; c.style.width = `${W * SCALE}px`; c.style.height = `${H * SCALE}px`; }
const pct = (v) => (v === null || v === undefined ? "—" : (v * 100).toFixed(1) + "%");

// ---------- 描画 ----------
const drawC = $("canvas-draw"), imgC = $("canvas-image");
let drawing = false, last = null, painted = false;
const maskC = document.createElement("canvas");   // 送信用: 黒背景に白で塗る内部解像度のマスク
const maskX = maskC.getContext("2d");

function resetMask() {
  maskX.fillStyle = "#000"; maskX.fillRect(0, 0, W, H);
  drawC.getContext("2d").clearRect(0, 0, W, H);
  painted = false;
}
function pos(e) {
  const r = drawC.getBoundingClientRect();
  return [(e.clientX - r.left) * W / r.width, (e.clientY - r.top) * H / r.height];
}
function stroke(a, b) {
  const r = Number($("brush").value);
  for (const [ctx, color] of [[maskX, "#fff"], [drawC.getContext("2d"), "rgba(255,60,60,0.55)"]]) {
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = r * 2; ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    ctx.beginPath(); ctx.arc(b[0], b[1], r, 0, Math.PI * 2); ctx.fill();
  }
  painted = true;
}
// Pointer Events でマウスもタッチも同じコードで扱う
drawC.addEventListener("pointerdown", (e) => { drawing = true; drawC.setPointerCapture(e.pointerId); last = pos(e); stroke(last, last); });
drawC.addEventListener("pointermove", (e) => { if (!drawing) return; const p = pos(e); stroke(last, p); last = p; });
for (const t of ["pointerup", "pointercancel"]) drawC.addEventListener(t, () => { drawing = false; });

// ---------- ゲーム進行 ----------
async function startGame() {
  state.questions = (await api("/api/round?n=10&defect_ratio=0.6")).questions;
  state.idx = 0; state.results = [];
  await showQuestion();
}
function renderQuestionText() {
  $("q-progress").textContent = `Q${state.idx + 1} / ${state.questions.length}`;
  $("q-bar").style.width = `${(state.idx / state.questions.length) * 100}%`;
}
async function showQuestion() {
  const q = state.questions[state.idx];
  const im = await loadImage(q.image_url);
  sizeCanvas(imgC); sizeCanvas(drawC); maskC.width = W; maskC.height = H;
  imgC.getContext("2d").drawImage(im, 0, 0, W, H);
  resetMask();
  renderQuestionText();
  $("btn-submit").disabled = $("btn-nodefect").disabled = false;
  show("question");
}
async function submit(noDefect) {
  $("btn-submit").disabled = $("btn-nodefect").disabled = true;
  const q = state.questions[state.idx];
  const useNone = noDefect || !painted;
  const r = await api("/api/answer", { id: q.id, no_defect: useNone, mask: useNone ? null : maskC.toDataURL("image/png") });
  r.playerMask = useNone ? null : maskC.toDataURL("image/png");
  r.image_url = q.image_url;
  state.threshold = r.default_threshold;
  state.results.push(r);
  await showResult(r);
}

// 画像にマスクを半透明で重ねて、パネルに描く。getMask(x, y) が true の画素を色付けする。
function paintPanel(canvas, im, color, maskData /* ImageData か null */, test) {
  sizeCanvas(canvas);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(im, 0, 0, W, H);
  if (!maskData) return;
  const base = ctx.getImageData(0, 0, W, H), d = base.data, m = maskData.data;
  for (let i = 0; i < d.length; i += 4) {
    if (test(m[i])) for (let k = 0; k < 3; k++) d[i + k] = d[i + k] * 0.4 + color[k] * 0.6;
  }
  ctx.putImageData(base, 0, 0);
}
function toImageData(im) {
  const c = document.createElement("canvas"); c.width = W; c.height = H;
  const x = c.getContext("2d"); x.drawImage(im, 0, 0, W, H); return x.getImageData(0, 0, W, H);
}
let aiProb = null;   // 現在の結果画面の AI 確率マップ (ImageData)
function renderResultText(r) {
  const outcome = r.player_iou > r.ai_iou ? "win_player" : (r.player_iou < r.ai_iou ? "win_ai" : "tie");
  $("r-title").innerHTML = t(r.gt_has_defect ? "r_defect" : "r_clean") + ` — <span class="${r.player_iou >= r.ai_iou ? "good" : "bad"}">${t(outcome)}</span>`;
  $("cap-player").innerHTML = `<i class="dot human"></i>${t("cap_player", { v: r.player_iou.toFixed(3) })}`;
  $("cap-ai").innerHTML = `<i class="dot ai"></i>${t("cap_ai", { v: r.ai_iou.toFixed(3) })}`;
  const atSummary = state.results.length % SUMMARY_EVERY === 0 || state.idx === state.questions.length - 1;
  $("btn-next").textContent = t(atSummary ? "btn_summary" : "btn_next");
}
async function showResult(r) {
  const [im, gtIm, probIm] = await Promise.all([loadImage(r.image_url), loadImage(r.gt_url), loadImage(r.prob_url)]);
  const plainCanvas = $("p-image"); sizeCanvas(plainCanvas); plainCanvas.getContext("2d").drawImage(im, 0, 0, W, H);
  paintPanel($("p-gt"), im, [60, 220, 90], toImageData(gtIm), (v) => v > 127);
  paintPanel($("p-player"), im, [255, 70, 70], r.playerMask ? toImageData(await loadImage(r.playerMask)) : null, (v) => v > 127);
  aiProb = toImageData(probIm);
  paintPanel($("p-ai"), im, [70, 140, 255], aiProb, (v) => v / 255 >= r.default_threshold);
  state.lastResult = r;
  renderResultText(r);
  show("result");
}

async function showSummary() {
  const final = state.idx === state.questions.length - 1;
  const answers = state.results.map((r) => ({ id: r.id, player_has_defect: r.player_has_defect }));
  const s = await api("/api/summary", { answers, threshold: state.threshold });
  const meanPlayerIou = state.results.reduce((a, r) => a + r.player_iou, 0) / state.results.length;
  $("s-title").textContent = t(final ? "s_final" : "s_mid", { n: s.n });
  const row = (name, x, iou, cls) => `<tr class="${cls}"><td>${name}</td><td>${pct(x.precision)}</td><td>${pct(x.recall)}</td><td>${pct(x.f1)}</td><td>${iou.toFixed(3)}</td><td>${x.tp}/${x.fp}/${x.fn}/${x.tn}</td></tr>`;
  $("s-table").innerHTML = `<tr><th></th><th>Precision</th><th>Recall</th><th>F1</th><th>${t("th_mean_iou")}</th><th>TP/FP/FN/TN</th></tr>`
    + row(t("who_you"), s.player, meanPlayerIou, "row-human") + row("AI", s.ai, s.ai.mean_iou, "row-ai");
  $("thr").value = s.threshold; $("thr-val").textContent = s.threshold.toFixed(2);
  $("btn-continue").textContent = t(final ? "btn_again" : "btn_continue");
  show("summary");
}
let thrTimer = null;
$("thr").addEventListener("input", () => {
  state.threshold = Number($("thr").value); $("thr-val").textContent = state.threshold.toFixed(2);
  clearTimeout(thrTimer); thrTimer = setTimeout(showSummary, 120);   // ドラッグ中のリクエストを間引く
});

$("btn-start").onclick = startGame;
$("btn-clear").onclick = resetMask;
$("btn-nodefect").onclick = () => submit(true);
$("btn-submit").onclick = () => submit(false);
$("btn-next").onclick = async () => {
  const atSummary = state.results.length % SUMMARY_EVERY === 0 || state.idx === state.questions.length - 1;
  if (atSummary) return showSummary();
  state.idx++; await showQuestion();
};
$("btn-continue").onclick = async () => {
  if (state.idx === state.questions.length - 1) return show("title");
  state.idx++; await showQuestion();
};

for (const b of document.querySelectorAll(".lang-switch button")) b.onclick = () => setLang(b.dataset.lang);
(async () => {
  STR = await (await fetch("/static/i18n.json")).json();
  applyStatic();
  state.cfg = await api("/api/config"); W = state.cfg.width; H = state.cfg.height;
})();
