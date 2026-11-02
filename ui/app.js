(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const store = {
    get(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* ignore */ } },
  };

  // ── Theme + voice preferences ──────────────────────────────
  const root = document.documentElement;
  root.dataset.theme = store.get("vv-theme", "dark");
  $("theme-toggle").onclick = () => {
    root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
    store.set("vv-theme", root.dataset.theme);
  };
  let voiceOn = store.get("vv-voice", "1") === "1";
  const voiceBtn = $("voice-toggle");
  function paintVoice() { voiceBtn.setAttribute("aria-pressed", String(voiceOn)); }
  paintVoice();
  function toggleVoice() {
    voiceOn = !voiceOn; store.set("vv-voice", voiceOn ? "1" : "0"); paintVoice();
    if (!voiceOn) speechSynthesis.cancel();
    toast(voiceOn ? "Voice on" : "Voice muted");
  }
  voiceBtn.onclick = toggleVoice;

  function toast(msg) {
    const t = $("toast"); t.textContent = msg; t.hidden = false;
    clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), 1800);
  }

  // ── Speech (offline Windows voices) ────────────────────────
  let voice = null;
  function pickVoice() {
    const vs = speechSynthesis.getVoices();
    voice = vs.find((v) => /en-(IN|US|GB)/i.test(v.lang) && /natural|online/i.test(v.name)) ||
            vs.find((v) => /en-IN/i.test(v.lang)) || vs.find((v) => /en-US/i.test(v.lang)) ||
            vs.find((v) => /^en/i.test(v.lang)) || null;
  }
  if ("speechSynthesis" in window) { pickVoice(); speechSynthesis.onvoiceschanged = pickVoice; }
  function speak(text, force) {
    if (!("speechSynthesis" in window) || (!voiceOn && !force) || !text) return;
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    if (voice) u.voice = voice;
    u.rate = 0.95; u.pitch = 1;
    const lbl = $("speak-lbl");
    u.onstart = () => (lbl.textContent = "Speaking...");
    u.onend = u.onerror = () => (lbl.textContent = "Speak again");
    speechSynthesis.speak(u);
  }

  // ── Startup: wait for models, then reveal the app ──────────
  const splash = $("splash");
  async function waitReady() {
    for (;;) {
      try {
        const s = await (await fetch("/api/status")).json();
        $("splash-msg").textContent = s.message || "Loading...";
        if (s.status === "ready") {
          $("engine-text").textContent = s.engine || "Offline AI";
          $("app").removeAttribute("aria-hidden");
          splash.classList.add("gone");
          setTimeout(() => splash.remove(), 700);
          speak("VocaVision is ready.");
          return;
        }
        if (s.status === "error") {
          $("splash-err").hidden = false;
          $("splash-err").textContent = "Could not start: " + s.error;
          $("splash-msg").textContent = "";
          return;
        }
      } catch { $("splash-msg").textContent = "Starting up..."; }
      await new Promise((r) => setTimeout(r, 700));
    }
  }
  waitReady();

  // Keep-alive so the background engine stops when this window is closed.
  const ping = () => fetch("/api/ping").catch(() => {});
  ping(); setInterval(ping, 4000);
  addEventListener("pagehide", () => navigator.sendBeacon && navigator.sendBeacon("/api/bye"));

  // ── Tabs ───────────────────────────────────────────────────
  let tab = "upload";
  function setTab(name) {
    tab = name;
    for (const t of document.querySelectorAll(".tab")) {
      const on = t.dataset.tab === name;
      t.classList.toggle("active", on); t.setAttribute("aria-selected", String(on));
    }
    $("panel-upload").hidden = name !== "upload";
    $("panel-camera").hidden = name !== "camera";
    if (name === "camera") startCamera(); else stopCamera();
  }
  for (const t of document.querySelectorAll(".tab")) t.onclick = () => setTab(t.dataset.tab);

  // ── Upload / drop / paste ──────────────────────────────────
  const drop = $("drop"), fileInput = $("file");
  let currentUpload = null;
  drop.onclick = () => fileInput.click();
  drop.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } };
  $("choose").onclick = () => fileInput.click();
  fileInput.onchange = () => { if (fileInput.files[0]) loadFile(fileInput.files[0]); fileInput.value = ""; };
  for (const ev of ["dragenter", "dragover"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); });
  for (const ev of ["dragleave", "drop"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); });
  drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) loadFile(f); });
  // Dropping anywhere else must not navigate away from the app.
  addEventListener("dragover", (e) => e.preventDefault());
  addEventListener("drop", (e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f && e.target !== drop) { setTab("upload"); loadFile(f); } });
  addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) { setTab("upload"); loadFile(item.getAsFile()); }
  });

  function readAsDataURL(file) {
    return new Promise((res, rej) => { const r = new FileReader(); r.onload = () => res(r.result); r.onerror = rej; r.readAsDataURL(file); });
  }
  async function loadFile(file) {
    if (!file.type.startsWith("image/")) { showError("Please choose an image file (JPG, PNG, WEBP...)."); return; }
    currentUpload = await readAsDataURL(file);
    $("preview").src = currentUpload; $("preview").hidden = false;
    $("drop-empty").hidden = true; drop.classList.add("has-image");
    $("analyze-upload").disabled = false;
    run(currentUpload);
  }
  $("analyze-upload").onclick = () => currentUpload && run(currentUpload);

  // ── Camera ─────────────────────────────────────────────────
  let stream = null;
  const PHONE_CAM = /camo|iriun|droidcam|epoccam|phone link|cross device|obs|ivcam/i;
  async function fillCameras() {
    const sel = $("cam-select");
    const cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
    const saved = store.get("vv-cam", "");
    sel.innerHTML = "";
    cams.forEach((c, i) => { const o = new Option(c.label || `Camera ${i + 1}`, c.deviceId); sel.add(o); });
    // Prefer the saved choice, else a phone-as-webcam device (Camo etc.), else the default.
    const pick = cams.find((c) => c.deviceId === saved) || cams.find((c) => PHONE_CAM.test(c.label));
    if (pick) sel.value = pick.deviceId;
    return pick ? pick.deviceId : null;
  }
  $("cam-select").onchange = () => { store.set("vv-cam", $("cam-select").value); stopCamera(); startCamera(true); };
  navigator.mediaDevices && (navigator.mediaDevices.ondevicechange = () => { if (tab === "camera") fillCameras(); });

  async function startCamera(useSelected) {
    const msg = $("cam-msg"); msg.classList.remove("hide"); msg.textContent = "Starting camera...";
    $("analyze-cam").disabled = true;
    try {
      if (stream) return finishCam();
      const size = { width: { ideal: 1280 }, height: { ideal: 720 } };
      let id = useSelected ? $("cam-select").value : null;
      if (!id) { try { id = await fillCameras(); } catch { /* labels need permission first */ } }
      stream = await navigator.mediaDevices.getUserMedia({ video: id ? { ...size, deviceId: { exact: id } } : size, audio: false });
      const labelled = await fillCameras();            // labels are available now that access is granted
      const track = stream.getVideoTracks()[0];
      if (track && track.getSettings().deviceId) $("cam-select").value = track.getSettings().deviceId;
      if (!id && labelled && labelled !== track.getSettings().deviceId) {   // auto-switch to Camo if present
        stopCamera(); id = labelled;
        stream = await navigator.mediaDevices.getUserMedia({ video: { ...size, deviceId: { exact: id } }, audio: false });
        $("cam-select").value = id;
      }
      const label = $("cam-select").selectedOptions[0]?.text || "";
      $("video").classList.toggle("nomirror", PHONE_CAM.test(label));   // mirror only built-in webcams
      $("video").srcObject = stream;
      await $("video").play().catch(() => {});
      finishCam();
    } catch (e) {
      msg.textContent = "Camera unavailable. Check that it is connected and not used by another app, then press Restart camera.";
    }
  }
  function finishCam() { $("cam-msg").classList.add("hide"); $("analyze-cam").disabled = false; }
  function stopCamera() {
    if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; $("video").srcObject = null; }
  }
  $("cam-restart").onclick = () => { stopCamera(); startCamera(); };
  function captureFrame() {
    const v = $("video");
    if (!v.videoWidth) return null;
    const c = document.createElement("canvas"); c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext("2d").drawImage(v, 0, 0);               // un-mirrored real view
    $("flash").classList.remove("go"); void $("flash").offsetWidth; $("flash").classList.add("go");
    return c.toDataURL("image/jpeg", 0.92);
  }
  $("analyze-cam").onclick = () => { const f = captureFrame(); if (f) run(f); };

  // ── Analyse ────────────────────────────────────────────────
  let busy = false, last = null;
  const COLOURS = { white: "#f8fafc", black: "#111827", gray: "#9ca3af", grey: "#9ca3af", red: "#ef4444", blue: "#3b82f6",
    green: "#22c55e", yellow: "#facc15", orange: "#f97316", pink: "#f472b6", purple: "#a855f7", brown: "#92400e",
    beige: "#e7d8b8", navy: "#1e3a8a" };

  function show(which) {
    $("empty").hidden = which !== "empty"; $("busy").hidden = which !== "busy";
    $("output").hidden = which !== "output"; $("error").hidden = which !== "error";
  }
  function showError(msg) { $("error").textContent = msg; show("error"); }

  async function run(dataUrl) {
    if (busy) return;
    busy = true; speechSynthesis.cancel(); show("busy"); $("time").hidden = true;
    ["analyze-upload", "analyze-cam"].forEach((id) => ($(id).disabled = true));
    try {
      const res = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image: dataUrl }) });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Something went wrong.");
      render(data); addHistory(dataUrl, data);
    } catch (e) {
      showError(e.message || "Something went wrong."); speak("Sorry, I could not analyse that image.");
    } finally {
      busy = false;
      $("analyze-upload").disabled = !currentUpload; $("analyze-cam").disabled = !stream;
    }
  }

  function parse(desc) {
    const m = desc.match(/^I can see (.+?)\.\s*I am (\d+)% confident\.?$/i);
    if (m) return { what: m[1], conf: +m[2] };
    const m2 = desc.match(/^I can see (.+?)\.?$/i);
    return m2 ? { what: m2[1], conf: null } : { what: null, conf: null };
  }
  const tidy = (s) => s.replace(/^(a|an)\s+/i, "").replace(/\b(a|an)\s+/gi, "").replace(/\s{2,}/g, " ").trim();

  function render(d) {
    last = d;
    $("result-img").src = d.annotated;
    const p = parse(d.description);
    const top = d.top && d.top[0];

    $("kind").textContent = p.what ? "I can see" : "Result";
    $("headline").textContent = p.what ? tidy(p.what) : d.description;
    $("full-desc").textContent = d.description;

    const chips = $("chips"); chips.innerHTML = "";
    if (p.what) {
      const words = tidy(p.what).toLowerCase();
      const colour = Object.keys(COLOURS).find((c) => words.startsWith(c + " ") || words.includes(" " + c + " "));
      if (colour) chips.append(chip(`<i class="swatch" style="background:${COLOURS[colour]}"></i>${colour}`, ""));
      const kind = top ? top.kind : null;
      if (kind) chips.append(chip(kind === "food" ? "Food" : "Clothing", kind === "food" ? "food" : "cloth"));
    }

    const ring = $("ring");
    if (p.conf != null) {
      ring.hidden = false; $("ring-val").textContent = p.conf + "%";
      const fg = $("ring-fg"); fg.style.strokeDashoffset = 113.1;
      requestAnimationFrame(() => requestAnimationFrame(() => (fg.style.strokeDashoffset = 113.1 * (1 - p.conf / 100))));
      fg.style.stroke = p.conf >= 70 ? "var(--cloth)" : p.conf >= 45 ? "var(--accent)" : "var(--food)";
    } else ring.hidden = true;

    const m = $("matches"); m.innerHTML = "";
    $("matches-wrap").hidden = !(d.top && d.top.length);
    for (const r of d.top || []) {
      const row = document.createElement("div"); row.className = "m-row";
      row.innerHTML = `<span class="m-label"></span><span class="m-bar"><i class="${r.kind}"></i></span><span class="m-val">${Math.round(r.p * 100)}%</span>`;
      row.firstChild.textContent = r.label; m.append(row);
      const bar = row.querySelector("i"); requestAnimationFrame(() => requestAnimationFrame(() => (bar.style.width = Math.max(2, r.p * 100) + "%")));
    }

    $("time").hidden = false; $("time").textContent = `${(d.ms / 1000).toFixed(1)} s`;
    show("output"); speak(d.description);
  }
  function chip(html, cls) { const s = document.createElement("span"); s.className = "chip " + cls; s.innerHTML = html; return s; }

  // ── History ────────────────────────────────────────────────
  const hist = [];
  function addHistory(src, data) {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement("canvas"); const w = 300, h = Math.round(300 * img.height / img.width);
      c.width = w; c.height = h; c.getContext("2d").drawImage(img, 0, 0, w, h);
      hist.unshift({ thumb: c.toDataURL("image/jpeg", 0.7), data });
      if (hist.length > 8) hist.pop(); paintHistory();
    };
    img.src = src;
  }
  function paintHistory() {
    const list = $("history"); list.innerHTML = "";
    $("history-wrap").hidden = !hist.length;
    for (const h of hist) {
      const b = document.createElement("button"); b.className = "h-item";
      b.innerHTML = `<img alt=""><span></span>`;
      b.firstChild.src = h.thumb; b.lastChild.textContent = h.data.description;
      b.onclick = () => { render(h.data); window.scrollTo({ top: 0, behavior: "smooth" }); };
      list.append(b);
    }
  }

  // ── Result actions + keyboard ──────────────────────────────
  $("speak").onclick = () => last && speak(last.description, true);
  $("copy").onclick = async () => {
    if (!last) return;
    try { await navigator.clipboard.writeText(last.description); toast("Copied"); } catch { toast("Could not copy"); }
  };
  addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey || /input|textarea/i.test(e.target.tagName)) return;
    const k = e.key.toLowerCase();
    if (k === "u") { setTab("upload"); fileInput.click(); }
    else if (k === "c") setTab("camera");
    else if (k === "r") last && speak(last.description, true);
    else if (k === "m") toggleVoice();
    else if (e.key === " " && tab === "camera" && !/button/i.test(e.target.tagName)) { e.preventDefault(); $("analyze-cam").click(); }
  });
})();
