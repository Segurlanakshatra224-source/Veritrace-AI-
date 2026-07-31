<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>VeriTrace — Multimodal Reasoning Auditor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root{
    --ink-950:#080a10;
    --ink-900:#0e121b;
    --ink-850:#141a25;
    --ink-800:#1a2130;
    --line:#232a3a;
    --line-soft:#1a2130;
    --text-hi:#edeff3;
    --text-mid:#9aa3b8;
    --text-dim:#5c6478;
    --cyan:#3ad9e8;
    --cyan-dim:#1c5a63;
    --amber:#ff7a45;
    --amber-dim:#6b3820;
    --green:#3ecf8e;
    --red:#ff5a5a;
    --mono:'IBM Plex Mono', ui-monospace, monospace;
    --sans:'Inter', -apple-system, sans-serif;
    --radius:10px;
  }

  *{box-sizing:border-box; margin:0; padding:0;}

  body{
    background:
      radial-gradient(1200px 600px at 15% -10%, rgba(58,217,232,0.06), transparent 60%),
      radial-gradient(900px 500px at 100% 10%, rgba(255,122,69,0.05), transparent 55%),
      var(--ink-950);
    color:var(--text-hi);
    font-family:var(--sans);
    min-height:100vh;
    line-height:1.5;
  }

  ::selection{background:rgba(58,217,232,0.25); color:#fff;}

  /* subtle scanline texture over whole page */
  body::before{
    content:"";
    position:fixed; inset:0;
    background-image:repeating-linear-gradient(180deg, rgba(255,255,255,0.012) 0px, rgba(255,255,255,0.012) 1px, transparent 1px, transparent 3px);
    pointer-events:none;
    z-index:1000;
  }

  a{color:inherit;}

  /* ============ HEADER ============ */
  header{
    display:flex; align-items:center; justify-content:space-between;
    padding:20px clamp(20px,4vw,48px);
    border-bottom:1px solid var(--line);
    position:sticky; top:0; z-index:100;
    background:rgba(8,10,16,0.85);
    backdrop-filter:blur(10px);
  }
  .brand{display:flex; align-items:center; gap:12px;}
  .brand-mark{
    width:34px; height:34px; border-radius:8px;
    background:linear-gradient(135deg, var(--cyan), var(--amber));
    display:flex; align-items:center; justify-content:center;
    font-family:var(--mono); font-weight:700; font-size:14px; color:#0a0d13;
    flex-shrink:0;
  }
  .brand-text{display:flex; flex-direction:column; gap:1px;}
  .brand-text .name{font-family:var(--mono); font-weight:600; font-size:14.5px; letter-spacing:0.06em;}
  .brand-text .tag{font-size:10.5px; color:var(--text-dim); letter-spacing:0.09em; text-transform:uppercase;}
  .case-id{
    font-family:var(--mono); font-size:11px; color:var(--text-dim);
    display:flex; align-items:center; gap:8px;
    letter-spacing:0.03em;
  }
  .case-id .dot{width:6px; height:6px; border-radius:50%; background:var(--green); box-shadow:0 0 8px var(--green);}
  .case-id.offline .dot{background:var(--red); box-shadow:0 0 8px var(--red);}

  /* ============ LAYOUT ============ */
  main{
    max-width:980px;
    margin:0 auto;
    padding:clamp(28px,5vw,64px) clamp(20px,4vw,48px) 100px;
  }

  /* ============ HERO / EMPTY STATE ============ */
  .hero{
    text-align:left;
    margin-bottom:36px;
    max-width:640px;
  }
  .hero .eyebrow{
    font-family:var(--mono); font-size:11px; color:var(--cyan);
    letter-spacing:0.14em; text-transform:uppercase; margin-bottom:14px;
    display:flex; align-items:center; gap:8px;
  }
  .hero .eyebrow::before{content:"//"; color:var(--text-dim);}
  .hero h1{
    font-size:clamp(28px,4vw,38px); font-weight:800; letter-spacing:-0.02em;
    line-height:1.15; margin-bottom:14px;
  }
  .hero h1 em{font-style:normal; color:var(--cyan);}
  .hero p{color:var(--text-mid); font-size:15px; max-width:520px;}

  /* ============ COMPOSER ============ */
  .composer{
    background:var(--ink-900);
    border:1px solid var(--line);
    border-radius:var(--radius);
    padding:18px;
    display:flex; flex-direction:column; gap:14px;
  }

  .dropzone{
    border:1.5px dashed var(--line);
    border-radius:8px;
    background:var(--ink-850);
    min-height:150px;
    display:flex; align-items:center; justify-content:center;
    flex-direction:column; gap:8px;
    cursor:pointer;
    transition:border-color .15s ease, background .15s ease;
    position:relative;
    overflow:hidden;
  }
  .dropzone:hover, .dropzone.drag{border-color:var(--cyan); background:#131b26;}
  .dropzone input{display:none;}
  .dropzone .icon{
    width:38px; height:38px; border-radius:50%;
    border:1px solid var(--line);
    display:flex; align-items:center; justify-content:center;
    color:var(--text-mid);
  }
  .dropzone .dz-label{font-size:13.5px; color:var(--text-mid);}
  .dropzone .dz-label b{color:var(--text-hi); font-weight:600;}
  .dropzone .dz-sub{font-family:var(--mono); font-size:10.5px; color:var(--text-dim); letter-spacing:0.03em;}

  .preview-wrap{
    position:relative; width:100%; min-height:150px; max-height:320px;
    border-radius:8px; overflow:hidden; display:none;
  }
  .preview-wrap.show{display:block;}
  .preview-wrap img{width:100%; max-height:320px; object-fit:contain; display:block; background:#000;}
  .preview-wrap .remove-img{
    position:absolute; top:10px; right:10px;
    background:rgba(8,10,16,0.75); border:1px solid var(--line);
    color:var(--text-hi); width:28px; height:28px; border-radius:50%;
    display:flex; align-items:center; justify-content:center; cursor:pointer;
    font-size:14px; backdrop-filter:blur(4px);
  }
  .preview-wrap .remove-img:hover{border-color:var(--red); color:var(--red);}

  /* scanning animation over image during analysis */
  .scan-line{
    position:absolute; left:0; right:0; height:2px;
    background:linear-gradient(90deg, transparent, var(--cyan), transparent);
    box-shadow:0 0 14px 2px var(--cyan);
    top:0; opacity:0; pointer-events:none;
  }
  .scan-line.active{animation:scan 1.8s linear infinite; opacity:1;}
  @keyframes scan{
    0%{top:0%;}
    100%{top:100%;}
  }
  .scan-grid{
    position:absolute; inset:0; opacity:0; pointer-events:none;
    background-image:
      linear-gradient(rgba(58,217,232,0.08) 1px, transparent 1px),
      linear-gradient(90deg, rgba(58,217,232,0.08) 1px, transparent 1px);
    background-size:24px 24px;
    transition:opacity .3s ease;
  }
  .scan-grid.active{opacity:1;}

  .qrow{display:flex; gap:10px; align-items:flex-end;}
  .qrow textarea{
    flex:1; resize:none; min-height:52px; max-height:160px;
    background:var(--ink-850); border:1px solid var(--line); border-radius:8px;
    color:var(--text-hi); font-family:var(--sans); font-size:14.5px;
    padding:14px 16px; outline:none; transition:border-color .15s ease;
  }
  .qrow textarea:focus{border-color:var(--cyan);}
  .qrow textarea::placeholder{color:var(--text-dim);}

  .run-btn{
    background:var(--cyan); color:#06262a; border:none; border-radius:8px;
    font-family:var(--mono); font-weight:600; font-size:12.5px; letter-spacing:0.04em;
    padding:0 22px; height:52px; cursor:pointer; white-space:nowrap;
    display:flex; align-items:center; gap:8px;
    transition:transform .1s ease, box-shadow .15s ease, background .15s ease;
  }
  .run-btn:hover:not(:disabled){box-shadow:0 0 0 3px rgba(58,217,232,0.18);}
  .run-btn:active:not(:disabled){transform:scale(0.98);}
  .run-btn:disabled{background:var(--ink-800); color:var(--text-dim); cursor:not-allowed;}

  .composer-foot{
    display:flex; justify-content:space-between; align-items:center;
    font-family:var(--mono); font-size:10.5px; color:var(--text-dim); letter-spacing:0.03em;
  }
  .presets{display:flex; gap:6px; flex-wrap:wrap;}
  .preset-chip{
    font-family:var(--sans); font-size:12px; color:var(--text-mid);
    background:var(--ink-850); border:1px solid var(--line); border-radius:20px;
    padding:5px 12px; cursor:pointer; transition:.15s ease;
  }
  .preset-chip:hover{border-color:var(--cyan); color:var(--text-hi);}

  /* ============ ERROR BANNER ============ */
  .error-banner{
    display:none; margin-top:16px; padding:14px 16px;
    background:rgba(255,90,90,0.07); border:1px solid rgba(255,90,90,0.35);
    border-radius:8px; font-size:13px; color:#ffb3b3;
    font-family:var(--mono);
    line-height:1.55;
  }
  .error-banner.show{display:block;}
  .error-banner code{background:rgba(255,255,255,0.06); padding:1px 6px; border-radius:4px;}

  /* ============ LOADING STATUS ============ */
  .status-strip{
    display:none; align-items:center; gap:10px; margin-top:16px;
    font-family:var(--mono); font-size:12px; color:var(--cyan); letter-spacing:0.03em;
  }
  .status-strip.show{display:flex;}
  .status-strip .pulse{
    width:8px; height:8px; border-radius:50%; background:var(--cyan);
    animation:pulse 1.1s ease-in-out infinite;
  }
  @keyframes pulse{0%,100%{opacity:1; transform:scale(1);} 50%{opacity:.4; transform:scale(0.7);}}

  /* ============ RESULTS ============ */
  #results{display:none; margin-top:52px;}
  #results.show{display:block; animation:fadeUp .5s ease;}
  @keyframes fadeUp{from{opacity:0; transform:translateY(12px);} to{opacity:1; transform:translateY(0);}}

  .divider{
    display:flex; align-items:center; gap:12px; margin-bottom:28px;
    font-family:var(--mono); font-size:10.5px; color:var(--text-dim); letter-spacing:0.12em; text-transform:uppercase;
  }
  .divider::before, .divider::after{content:""; flex:1; height:1px; background:var(--line);}

  /* exchange bubbles */
  .exchange{display:flex; flex-direction:column; gap:14px; margin-bottom:32px;}
  .bubble-q{
    align-self:flex-end; max-width:78%;
    background:var(--ink-850); border:1px solid var(--line);
    border-radius:12px 12px 2px 12px; padding:12px 16px; font-size:14px; color:var(--text-hi);
  }
  .bubble-q .qimg{max-width:220px; border-radius:8px; display:block; margin-bottom:8px; border:1px solid var(--line);}

  /* verdict strip */
  .verdict{
    display:flex; align-items:center; gap:12px;
    padding:14px 18px; border-radius:8px; margin-bottom:22px;
    border:1px solid var(--line); background:var(--ink-900);
  }
  .verdict .vdot{width:10px; height:10px; border-radius:50%; flex-shrink:0;}
  .verdict.grounded{border-color:rgba(62,207,142,0.35); background:rgba(62,207,142,0.05);}
  .verdict.grounded .vdot{background:var(--green); box-shadow:0 0 10px var(--green);}
  .verdict.flagged{border-color:rgba(255,122,69,0.35); background:rgba(255,122,69,0.05);}
  .verdict.flagged .vdot{background:var(--amber); box-shadow:0 0 10px var(--amber);}
  .verdict .vtext{font-family:var(--mono); font-size:12.5px; letter-spacing:0.02em;}
  .verdict .vtext b{font-weight:600;}
  .verdict .vsub{color:var(--text-dim); font-size:11.5px; margin-left:auto; font-family:var(--mono);}

  /* answer card */
  .card{
    background:var(--ink-900); border:1px solid var(--line); border-radius:var(--radius);
    padding:22px; margin-bottom:20px;
  }
  .card h3{
    font-family:var(--mono); font-size:11.5px; letter-spacing:0.1em; text-transform:uppercase;
    color:var(--text-dim); margin-bottom:14px; display:flex; align-items:center; gap:8px;
  }
  .card h3 .num{color:var(--cyan);}
  .answer-text{font-size:15.5px; line-height:1.65; color:var(--text-hi); white-space:pre-wrap;}

  /* metrics row */
  .metrics{display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; margin-bottom:20px;}
  .metric{
    background:var(--ink-900); border:1px solid var(--line); border-radius:var(--radius);
    padding:16px;
  }
  .metric .mlabel{font-family:var(--mono); font-size:10px; color:var(--text-dim); letter-spacing:0.08em; text-transform:uppercase; margin-bottom:8px;}
  .metric .mval{font-family:var(--mono); font-size:22px; font-weight:600;}
  .metric .mbar{height:4px; background:var(--ink-800); border-radius:2px; margin-top:10px; overflow:hidden;}
  .metric .mbar-fill{height:100%; border-radius:2px; background:var(--cyan); transition:width .8s ease;}

  /* heatmap panels */
  .heat-grid{display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:16px; margin-bottom:20px;}
  @media (max-width:680px){.heat-grid{grid-template-columns:1fr;}}
  .heat-panel{
    background:var(--ink-900); border:1px solid var(--line); border-radius:var(--radius);
    overflow:hidden;
  }
  .heat-panel .hp-head{
    padding:14px 16px; border-bottom:1px solid var(--line);
    display:flex; align-items:center; justify-content:space-between;
  }
  .heat-panel .hp-head .hp-title{font-family:var(--mono); font-size:11.5px; letter-spacing:0.06em; text-transform:uppercase; color:var(--text-mid);}
  .heat-panel .hp-head .hp-swatch{width:28px; height:8px; border-radius:2px; background:linear-gradient(90deg,#0000ff,#00ffff,#ffff00,#ff0000);}
  .heat-panel img{width:100%; display:block; background:#000;}
  .heat-panel .hp-foot{
    padding:12px 16px; font-family:var(--mono); font-size:11px; color:var(--text-dim);
    border-top:1px solid var(--line); display:flex; justify-content:space-between;
  }

  /* reasoning accordion */
  .accordion-item{border-bottom:1px solid var(--line);}
  .accordion-item:last-child{border-bottom:none;}
  .acc-head{
    padding:14px 0; display:flex; align-items:center; justify-content:between; gap:10px;
    cursor:pointer; display:flex; justify-content:space-between;
  }
  .acc-head .at{font-family:var(--mono); font-size:13px; font-weight:600; color:var(--text-hi);}
  .acc-head .chev{color:var(--text-dim); transition:transform .2s ease; font-size:12px;}
  .accordion-item.open .chev{transform:rotate(90deg); color:var(--cyan);}
  .acc-body{max-height:0; overflow:hidden; transition:max-height .25s ease;}
  .accordion-item.open .acc-body{max-height:600px;}
  .acc-body-inner{padding:0 0 16px; font-size:13.5px; color:var(--text-mid); line-height:1.7; white-space:pre-wrap;}

  /* download */
  .download-row{display:flex; gap:10px; flex-wrap:wrap; margin-top:6px;}
  .dl-btn{
    display:flex; align-items:center; gap:8px;
    background:var(--ink-850); border:1px solid var(--line); border-radius:8px;
    padding:11px 18px; font-family:var(--mono); font-size:12px; color:var(--text-hi);
    text-decoration:none; cursor:pointer; transition:.15s ease;
  }
  .dl-btn:hover{border-color:var(--cyan); color:var(--cyan);}
  .dl-btn.primary{background:var(--cyan); color:#06262a; border-color:var(--cyan); font-weight:600;}
  .dl-btn.primary:hover{color:#06262a; box-shadow:0 0 0 3px rgba(58,217,232,0.18);}

  footer{
    text-align:center; padding:28px; font-family:var(--mono); font-size:10.5px; color:var(--text-dim);
    letter-spacing:0.04em; border-top:1px solid var(--line);
  }

  @media (max-width:600px){
    .qrow{flex-direction:column; align-items:stretch;}
    .run-btn{height:44px;}
  }
</style>
</head>
<body>

<header>
  <div class="brand">
    <div class="brand-mark">V</div>
    <div class="brand-text">
      <span class="name">VERITRACE</span>
      <span class="tag">Multimodal Reasoning Auditor</span>
    </div>
  </div>
  <div class="case-id offline" id="caseId">
    <span class="dot"></span>
    <span id="caseIdText">CHECKING BACKEND…</span>
  </div>
</header>

<main>

  <div class="hero">
    <div class="eyebrow">Evidence-grounded VQA</div>
    <h1>Submit an image or video. Ask a question. <em>Get an audited answer.</em></h1>
    <p>VeriTrace runs your image or video through Qwen2-VL, then cross-examines its own answer with attention rollout, integrated gradients, confidence scoring, and hallucination detection — so you see not just what it said, but why. For video, mention a timestamp in your question (e.g. "what happens at 0:12") to focus on that exact moment, or leave it out to auto-detect key scenes.</p>
  </div>

  <form class="composer" id="composerForm">
    <label class="dropzone" id="dropzone" for="fileInput">
      <input type="file" id="fileInput" accept="image/png,image/jpeg,image/jpg,image/bmp,video/mp4,video/avi,video/quicktime,video/x-matroska" />
      <div class="dz-empty" id="dzEmpty">
        <div class="icon">↑</div>
        <div class="dz-label"><b>Click to upload</b> or drag an image/video here</div>
        <div class="dz-sub">JPG · PNG · BMP · MP4 · AVI · MOV · MKV</div>
      </div>
      <div class="preview-wrap" id="previewWrap">
        <img id="previewImg" alt="Uploaded evidence" />
        <video id="previewVideo" muted controls style="display:none;max-width:100%;"></video>
        <div class="scan-grid" id="scanGrid"></div>
        <div class="scan-line" id="scanLine"></div>
        <div class="remove-img" id="removeImg" title="Remove file">✕</div>
      </div>
    </label>

    <div class="qrow">
      <textarea id="questionInput" placeholder="What do you want to know? For video, you can mention a time like 'what happens at 0:12'" rows="1"></textarea>
      <button type="submit" class="run-btn" id="runBtn" disabled>
        <span id="runBtnLabel">RUN ANALYSIS</span>
      </button>
    </div>

    <div class="composer-foot">
      <div class="presets">
        <span class="preset-chip" data-q="Describe what is happening in this image in detail.">Describe scene</span>
        <span class="preset-chip" data-q="How many distinct objects or subjects are visible in this image?">Count objects</span>
        <span class="preset-chip" data-q="Does this image show any signs of digital manipulation or editing?">Check authenticity</span>
        <span class="preset-chip" data-q="What text, if any, is visible in this image?">Read text</span>
      </div>
      <span id="modelTag">qwen2-vl-2b-instruct</span>
    </div>
  </form>

  <div class="status-strip" id="statusStrip">
    <span class="pulse"></span>
    <span id="statusText">Loading vision-language model…</span>
  </div>

  <div class="error-banner" id="errorBanner"></div>

  <section id="results"></section>

</main>

<footer>VERITRACE // EVIDENCE IS THE ONLY ARGUMENT</footer>

<script>
(function(){
  "use strict";

  // ---- Configuration ----
  // Point this at wherever you run server.py (see the backend file for setup).
  const API_BASE = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "http://localhost:8000"
    : "http://localhost:8000"; // change this if your backend runs elsewhere

  // ---- Elements ----
  const fileInput = document.getElementById('fileInput');
  const dropzone = document.getElementById('dropzone');
  const dzEmpty = document.getElementById('dzEmpty');
  const previewWrap = document.getElementById('previewWrap');
  const previewImg = document.getElementById('previewImg');
  const previewVideo = document.getElementById('previewVideo');
  const removeImg = document.getElementById('removeImg');
  const scanLine = document.getElementById('scanLine');
  const scanGrid = document.getElementById('scanGrid');
  const questionInput = document.getElementById('questionInput');
  const runBtn = document.getElementById('runBtn');
  const runBtnLabel = document.getElementById('runBtnLabel');
  const composerForm = document.getElementById('composerForm');
  const statusStrip = document.getElementById('statusStrip');
  const statusText = document.getElementById('statusText');
  const errorBanner = document.getElementById('errorBanner');
  const resultsEl = document.getElementById('results');
  const caseIdEl = document.getElementById('caseId');
  const caseIdText = document.getElementById('caseIdText');

  let selectedFile = null;
  let isVideoFile = false;

  // ---- Case ID / backend health ----
  function genCaseId(){
    const d = new Date();
    const pad = n => String(n).padStart(2,'0');
    return `VT-${d.getFullYear()}${pad(d.getMonth()+1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}`;
  }

  async function checkHealth(){
    try{
      const res = await fetch(`${API_BASE}/api/health`, { method:'GET' });
      if(res.ok){
        caseIdEl.classList.remove('offline');
        caseIdText.textContent = genCaseId();
        return true;
      }
    }catch(e){}
    caseIdEl.classList.add('offline');
    caseIdText.textContent = 'BACKEND OFFLINE';
    return false;
  }
  checkHealth();

  // ---- Dropzone interactions ----
  dropzone.addEventListener('dragover', e=>{ e.preventDefault(); dropzone.classList.add('drag'); });
  dropzone.addEventListener('dragleave', ()=> dropzone.classList.remove('drag'));
  dropzone.addEventListener('drop', e=>{
    e.preventDefault();
    dropzone.classList.remove('drag');
    if(e.dataTransfer.files && e.dataTransfer.files[0]){
      setFile(e.dataTransfer.files[0]);
    }
  });
  fileInput.addEventListener('change', e=>{
    if(e.target.files && e.target.files[0]) setFile(e.target.files[0]);
  });
  removeImg.addEventListener('click', e=>{
    e.preventDefault(); e.stopPropagation();
    selectedFile = null;
    isVideoFile = false;
    fileInput.value = '';
    previewWrap.classList.remove('show');
    previewImg.style.display = 'none';
    previewVideo.style.display = 'none';
    previewVideo.src = '';
    dzEmpty.style.display = 'flex';
    updateRunButton();
  });

  function setFile(file){
    const isImage = file.type.startsWith('image/');
    const isVideo = file.type.startsWith('video/');
    if(!isImage && !isVideo) return;

    selectedFile = file;
    isVideoFile = isVideo;

    if(isVideo){
      previewVideo.src = URL.createObjectURL(file);
      previewVideo.style.display = 'block';
      previewImg.style.display = 'none';
      previewWrap.classList.add('show');
      dzEmpty.style.display = 'none';
    }else{
      const reader = new FileReader();
      reader.onload = ev=>{
        previewImg.src = ev.target.result;
        previewImg.style.display = 'block';
        previewVideo.style.display = 'none';
        previewWrap.classList.add('show');
        dzEmpty.style.display = 'none';
      };
      reader.readAsDataURL(file);
    }
    updateRunButton();
  }

  questionInput.addEventListener('input', ()=>{
    questionInput.style.height = 'auto';
    questionInput.style.height = Math.min(questionInput.scrollHeight, 160) + 'px';
    updateRunButton();
  });

  document.querySelectorAll('.preset-chip').forEach(chip=>{
    chip.addEventListener('click', ()=>{
      questionInput.value = chip.dataset.q;
      questionInput.dispatchEvent(new Event('input'));
      questionInput.focus();
    });
  });

  function updateRunButton(){
    runBtn.disabled = !(selectedFile && questionInput.value.trim().length > 0);
  }

  // ---- Loading status cycling ----
  const statusMessages = [
    "Loading vision-language model…",
    "Running Qwen2-VL inference…",
    "Tracing attention across every layer…",
    "Computing integrated gradients on image patches…",
    "Scoring token-level confidence…",
    "Cross-checking evidence for hallucination…",
    "Compiling explainability report…"
  ];
  let statusInterval = null;
  function startStatusCycle(){
    let i = 0;
    statusText.textContent = statusMessages[0];
    statusStrip.classList.add('show');
    statusInterval = setInterval(()=>{
      i = (i+1) % statusMessages.length;
      statusText.textContent = statusMessages[i];
    }, 1600);
  }
  function stopStatusCycle(){
    clearInterval(statusInterval);
    statusStrip.classList.remove('show');
  }

  // ---- Submit ----
  composerForm.addEventListener('submit', async e=>{
    e.preventDefault();
    if(!selectedFile || !questionInput.value.trim()) return;

    errorBanner.classList.remove('show');
    runBtn.disabled = true;
    runBtnLabel.textContent = 'ANALYZING…';
    scanLine.classList.add('active');
    scanGrid.classList.add('active');
    startStatusCycle();

    const question = questionInput.value.trim();
    const questionImgSrc = previewImg.src;

    const formData = new FormData();
    const endpoint = isVideoFile ? '/api/analyze-video' : '/api/analyze';
    formData.append(isVideoFile ? 'video' : 'image', selectedFile);
    formData.append('question', question);

    try{
      const res = await fetch(`${API_BASE}${endpoint}`, { method:'POST', body: formData });
      if(!res.ok){
        const errText = await res.text().catch(()=> '');
        throw new Error(`Server responded ${res.status}. ${errText}`);
      }
      const data = await res.json();
      if(isVideoFile){
        renderVideoResults(data, question);
      }else{
        renderResults(data, question, questionImgSrc);
      }
    }catch(err){
      showError(err);
    }finally{
      stopStatusCycle();
      scanLine.classList.remove('active');
      scanGrid.classList.remove('active');
      runBtn.disabled = false;
      runBtnLabel.textContent = 'RUN ANALYSIS';
    }
  });

  function showError(err){
    errorBanner.innerHTML =
      `⚠ Could not reach the VeriTrace backend.<br><br>` +
      `${(err && err.message) ? err.message : err}<br><br>` +
      `Make sure the API server is running: <code>python server.py</code> — it must be reachable at <code>${API_BASE}</code>.`;
    errorBanner.classList.add('show');
  }

  // ---- Render results ----
  function fmtPct(v){
    if(v === null || v === undefined) return 'N/A';
    return (v*100).toFixed(1) + '%';
  }

  function renderAccordion(container, sections){
    if(!container) return;
    sections.forEach(([title, body], idx)=>{
      const item = document.createElement('div');
      item.className = 'accordion-item' + (idx === 0 ? ' open' : '');
      item.innerHTML = `
        <div class="acc-head">
          <span class="at">${title}</span>
          <span class="chev">▸</span>
        </div>
        <div class="acc-body"><div class="acc-body-inner">${escapeHtml(body)}</div></div>
      `;
      item.querySelector('.acc-head').addEventListener('click', ()=>{
        item.classList.toggle('open');
      });
      container.appendChild(item);
    });
  }

  // Builds the verdict/metrics/heat-grid/accordion-placeholder/downloads
  // markup for ONE analyzed frame (an image, or a single selected video
  // frame). Returns the HTML plus the accordion sections separately, since
  // the accordion needs live DOM elements + click handlers attached after
  // the HTML is inserted into the page - accordion-container's data-frame-key
  // is how the two get matched back up afterwards.
  function frameResultBlockHTML(data, frameKey, headerLabel){
    const hallucinated = !!data.hallucination;
    const confidence = data.confidence ?? 0;

    const explanation = data.explanation || {};
    const sections = [
      ['Confidence', explanation['Confidence']],
      ['Trust', explanation['Trust']],
      ['Hallucination', explanation['Hallucination']],
      ['Counterfactual', explanation['Counterfactual']],
      ['Attention Rollout', explanation['Attention Rollout']],
      ['Integrated Gradients', explanation['Integrated Gradients']],
      ['Cross Attention', explanation['Cross Attention']],
      ['Evidence Coverage', explanation['Evidence Coverage']],
    ].filter(([,v]) => v);

    const html = `
      <div class="verdict ${hallucinated ? 'flagged' : 'grounded'}">
        <span class="vdot"></span>
        <span class="vtext"><b>${headerLabel ? headerLabel + ' — ' : ''}${hallucinated ? 'Possible Hallucination' : 'Grounded Response'}</b></span>
        <span class="vsub">${data.trust_score !== null && data.trust_score !== undefined ? `Trust ${data.trust_score}/100 · ${data.trust_level || ''}` : 'N/A'}</span>
      </div>

      <div class="card">
        <h3><span class="num">01</span> Response</h3>
        <div class="answer-text">${escapeHtml(data.answer || '')}</div>
      </div>

      <div class="metrics">
        <div class="metric">
          <div class="mlabel">Confidence</div>
          <div class="mval">${fmtPct(confidence)}</div>
          <div class="mbar"><div class="mbar-fill" style="width:${Math.round(confidence*100)}%"></div></div>
        </div>
        <div class="metric">
          <div class="mlabel">Image Attention Mass</div>
          <div class="mval">${data.attention_mass !== null && data.attention_mass !== undefined ? fmtPct(data.attention_mass) : 'N/A'}</div>
          <div class="mbar"><div class="mbar-fill" style="width:${data.attention_mass ? Math.round(data.attention_mass*100) : 0}%; background:var(--amber);"></div></div>
        </div>
        <div class="metric">
          <div class="mlabel">Verdict</div>
          <div class="mval" style="color:${hallucinated ? 'var(--amber)' : 'var(--green)'}">${hallucinated ? 'FLAGGED' : 'CLEAR'}</div>
        </div>
        <div class="metric">
          <div class="mlabel">Explanation Faithfulness</div>
          ${data.counterfactual ? `
            <div class="mval" style="color:${data.counterfactual.is_faithful ? 'var(--green)' : 'var(--amber)'}">${data.counterfactual.is_faithful ? 'FAITHFUL' : 'UNFAITHFUL'}</div>
            <div class="mbar"><div class="mbar-fill" style="width:${Math.round((data.counterfactual.confidence_drop || 0)*100)}%; background:${data.counterfactual.is_faithful ? 'var(--green)' : 'var(--amber)'};"></div></div>
          ` : `<div class="mval" style="color:var(--text-dim);">N/A</div>`}
        </div>
      </div>

      <h3 style="font-family:var(--mono);font-size:11.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--text-dim);margin-bottom:14px;"><span style="color:var(--cyan)">02</span> Visual Evidence</h3>
      <div class="heat-grid">
        <div class="heat-panel">
          <div class="hp-head"><span class="hp-title">Attention Rollout</span><span class="hp-swatch"></span></div>
          ${data.attention_overlay ? `<img src="data:image/png;base64,${data.attention_overlay}" alt="attention rollout overlay" />` : `<div style="padding:40px;text-align:center;color:var(--text-dim);font-family:var(--mono);font-size:12px;">Unavailable for this run</div>`}
          <div class="hp-foot"><span>layer-traced attention</span><span>${data.attention_map_stats ? data.attention_map_stats.grid_size : ''}</span></div>
        </div>
        <div class="heat-panel">
          <div class="hp-head"><span class="hp-title">Integrated Gradients</span><span class="hp-swatch"></span></div>
          ${data.ig_overlay ? `<img src="data:image/png;base64,${data.ig_overlay}" alt="integrated gradients overlay" />` : `<div style="padding:40px;text-align:center;color:var(--text-dim);font-family:var(--mono);font-size:12px;">Unavailable for this run</div>`}
          <div class="hp-foot"><span>gradient attribution</span><span>${data.ig_map_stats ? data.ig_map_stats.grid_size : ''}</span></div>
        </div>
        <div class="heat-panel">
          <div class="hp-head"><span class="hp-title">Cross Attention</span><span class="hp-swatch"></span></div>
          ${data.cross_overlay ? `<img src="data:image/png;base64,${data.cross_overlay}" alt="cross attention overlay" />` : `<div style="padding:40px;text-align:center;color:var(--text-dim);font-family:var(--mono);font-size:12px;">Unavailable for this run</div>`}
          <div class="hp-foot"><span>final-layer attention</span><span></span></div>
        </div>
      </div>

      <div class="card">
        <h3><span class="num">03</span> Reasoning Trace</h3>
        <div class="accordion-container" data-frame-key="${frameKey}"></div>
      </div>

      <div class="card">
        <h3><span class="num">04</span> Download Report</h3>
        <div class="download-row">
          ${data.report_pdf_url ? `<a class="dl-btn primary" href="${API_BASE}${data.report_pdf_url}" download>⬇ PDF REPORT (.pdf)</a>` : ''}
          ${data.report_txt_url ? `<a class="dl-btn" href="${API_BASE}${data.report_txt_url}" download>⬇ TEXT REPORT (.txt)</a>` : ''}
          ${data.report_json_url ? `<a class="dl-btn" href="${API_BASE}${data.report_json_url}" download>⬇ JSON REPORT (.json)</a>` : ''}
        </div>
      </div>
    `;

    return { html, sections };
  }

  function renderResults(data, question, questionImgSrc){
    const { html, sections } = frameResultBlockHTML(data, 'main', null);

    resultsEl.innerHTML = `
      <div class="divider">Case Report</div>

      <div class="exchange">
        <div class="bubble-q">
          <img class="qimg" src="${questionImgSrc}" alt="submitted evidence" />
          ${escapeHtml(question)}
        </div>
      </div>

      ${html}
    `;

    renderAccordion(resultsEl.querySelector('.accordion-container[data-frame-key="main"]'), sections);

    resultsEl.classList.add('show');
    resultsEl.scrollIntoView({ behavior:'smooth', block:'start' });
  }

  function renderVideoResults(data, question){
    const frames = data.frames || [];
    const hasTimestamp = data.requested_time !== null && data.requested_time !== undefined;
    const requestedNote = hasTimestamp
      ? `Showing the frame nearest to <b>${data.requested_time}s</b>, as mentioned in your question.`
      : `No specific timestamp was mentioned, so VeriTrace auto-detected ${frames.length} key moment${frames.length === 1 ? '' : 's'} (scene changes) to analyze in full.`;

    const pending = [];
    let framesHtml = '';

    frames.forEach((frameData, idx)=>{
      const key = `frame-${idx}`;
      const { html, sections } = frameResultBlockHTML(frameData, key, `Frame @ ${frameData.timestamp}s`);
      pending.push([key, sections]);
      framesHtml += `<div class="divider">Moment ${idx + 1} — ${frameData.timestamp}s</div>${html}`;
    });

    if(!frames.length){
      framesHtml = `<div class="card"><div class="answer-text" style="color:var(--text-dim);">No frames could be selected for detailed explainability on this video.</div></div>`;
    }

    resultsEl.innerHTML = `
      <div class="divider">Case Report — Video</div>

      <div class="exchange">
        <div class="bubble-q">${escapeHtml(question)}</div>
      </div>

      <div class="card">
        <h3><span class="num">00</span> Overall Video Answer</h3>
        <div class="answer-text">${escapeHtml(data.overall_answer || '')}</div>
        <div style="margin-top:12px;font-family:var(--mono);font-size:12px;color:var(--text-dim);">${requestedNote}</div>
      </div>

      ${framesHtml}
    `;

    pending.forEach(([key, sections])=>{
      renderAccordion(resultsEl.querySelector(`.accordion-container[data-frame-key="${key}"]`), sections);
    });

    resultsEl.classList.add('show');
    resultsEl.scrollIntoView({ behavior:'smooth', block:'start' });
  }

  function escapeHtml(str){
    if(!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

})();
</script>

</body>
</html>