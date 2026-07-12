#!/usr/bin/env python3
"""
generate_dashboards.py
Reads web/results_demo.json (COMPUTED pipeline output) and emits one standalone,
data-embedded dashboard per structural family, plus an index.html that links them
and the combined atlas. Each family dashboard opens showing computed results — no
manual load needed. Re-run run_pipeline.py on real variants, then re-run this to
regenerate the dashboards on real data.
"""
import json, os

ARMS = ["mpra_transcription", "caqtl_accessibility"]
ARM_META = {
    "mpra_transcription": {"short": "MPRA", "long": "transcription", "unit": "log2 fold-change"},
    "caqtl_accessibility": {"short": "caQTL", "long": "chromatin accessibility", "unit": "accessibility Δ"},
}

FAM_META = {
    "SHP": dict(name="DNA shape", color="#0aa2c0", cls="geometry · continuous", style="continuous",
        tip="MGW · Roll · ProT · HelT · EP",
        method="For each allele, compute per-base DNA-shape features and take the allelic "
               "shape delta over the variant window. Shape captures <b>shape-only variants</b> "
               "that preserve the k-mer/PWM match but bend or open the helix differently.",
        tools=["DNAshapeR", "deepDNAshape"],
        subtitle="Which of the five shape features carries the signal"),
    "SID": dict(name="Duplex destabilization (SIDD)", color="#f43f6f", cls="mechanics · continuous", style="continuous",
        tip="stress-induced destabilization / local melting ΔG",
        method="Score local duplex stability from nearest-neighbour melting energy (ΔG). "
               "Regulatory DNA unwinds easily; a variant that shifts stability can change output "
               "<b>without touching a motif</b> — orthogonal to shape and to G4/i-motif.",
        tools=["SIST / WebSIDD", "nearest-neighbour ΔG"], subtitle="Which destabilization component carries the signal"),
    "ZDA": dict(name="Z-DNA propensity", color="#8b5cf6", cls="non-B structure · discrete", style="discrete",
        tip="alternating purine–pyrimidine Z-forming potential",
        method="Score alternating purine/pyrimidine tracts for Z-DNA-forming potential; a single "
               "base can create or break Z-forming propensity near a TSS.",
        tools=["Z-Hunt / zhunt3", "non-B DB (Z-DNA)"], subtitle=None),
    "RLP": dict(name="R-loop potential", color="#12b76a", cls="non-B structure · discrete", style="discrete",
        tip="RNA:DNA hybrid / R-loop-forming sequence",
        method="Detect G-rich, GC-skewed R-loop-initiating zones (RIZ). GC-skewed promoter and "
               "terminator regions form RNA:DNA hybrids that regulate transcription and are variant-sensitive.",
        tools=["QmRLFS-finder", "RLBase / R-loopBase"], subtitle="Which R-loop component carries the signal"),
    "CTX": dict(name="Cruciform / Triplex", color="#e0559b", cls="non-B structure · discrete", style="discrete",
        tip="inverted repeats (cruciform) + mirror repeats (H-DNA)",
        method="Detect inverted repeats (cruciform) and mirror repeats (triplex / H-DNA); the event "
               "is a repeat symmetry created or broken by the variant — two distinct non-B axes.",
        tools=["non-B DB (IR, MR)", "EMBOSS palindrome", "Triplex-Inspector"],
        subtitle="Cruciform (IR) vs triplex (MR) sub-tracks"),
    "NUC": dict(name="Nucleosome positioning", color="#6366f1", cls="mechanics · continuous", style="continuous",
        tip="predicted occupancy / positioning delta",
        method="Predict nucleosome occupancy from sequence mechanics and take the allelic occupancy "
               "delta. Regulatory variants frequently act by changing local <b>accessibility</b> — "
               "especially strong in the caQTL arm.",
        tools=["NuPoP", "sequence bendability model"], subtitle=None),
}
VERBS = {"continuous": ["gain", "loss", "shift"], "discrete": ["create", "disrupt", "modulate"]}
REF_VERB = {"continuous": "shift", "discrete": "none"}

def build_fam_obj(code, bundle):
    meta = FAM_META[code]; style = meta["style"]
    arms = {}
    for arm in ARMS:
        a = bundle["arms"].get(arm)
        if not a: continue
        block = a["families"][code]
        none_med = a["measured"]["category_effects"]["none"]["median_abs_effect"]
        arms[arm] = {"n_variants": a["n_variants"], "none_med": none_med, "block": block}
    return {
        "code": code, "name": meta["name"], "color": meta["color"], "cls": meta["cls"],
        "style": style, "verbs": VERBS[style], "ref_verb": REF_VERB[style],
        "tip": meta["tip"], "method": meta["method"], "tools": meta["tools"],
        "subtitle": meta["subtitle"], "arm_meta": ARM_META, "arms": arms,
    }

# ---------------------------------------------------------------- HTML templates
HEAD = """<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0"><title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
<style>
:root{--panel:#fff;--panel-2:#f2f6fd;--line:#d7e0f0;--ink:#101c33;--muted:#59688a;--faint:#93a1bf;
--accent:#0aa2c0;--accent-dim:#0d7d8f;--good:#12b76a;--warn:#f59e0b;--danger:#f43f6f;--tf:#8b5cf6;--r:10px;--fam:__COLOR__;}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:#eef3fb;color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;line-height:1.5;
background-image:radial-gradient(1100px 560px at 86% -12%,#cdf2e6cc,transparent 62%),radial-gradient(1000px 520px at -12% 4%,#d3e4ffcc,transparent 62%),radial-gradient(900px 520px at 50% 128%,#ece1ffcc,transparent 60%);background-attachment:fixed}
body::before{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;background-image:radial-gradient(#b9c6de55 1px,transparent 1.4px);background-size:26px 26px;mask:linear-gradient(180deg,#000 60%,transparent)}
.mono{font-family:"IBM Plex Mono",monospace}.wrap{max-width:1060px;margin:0 auto;padding:0 22px;position:relative;z-index:1}
a{color:var(--accent-dim);text-decoration:none}a:hover{color:var(--accent)}
header{padding:26px 0 6px}.back{font-size:13px;font-family:"IBM Plex Mono"}
.brand{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-top:8px}
.dot{width:16px;height:16px;border-radius:5px;background:var(--fam)}
.brand h1{font-family:"Space Grotesk";font-weight:700;font-size:27px;letter-spacing:-.02em;margin:0}
.brand .code{font-family:"IBM Plex Mono";font-size:12px;border:1px solid var(--fam);color:var(--fam);padding:2px 8px;border-radius:999px;background:color-mix(in srgb,var(--fam) 8%,transparent)}
.cls{color:var(--muted);font-family:"IBM Plex Mono";font-size:12px;margin-top:2px}
.lede{color:var(--muted);max-width:720px;margin:10px 0 0;font-size:14.5px}
.prov-badge{font-family:"IBM Plex Mono";font-size:10px;font-weight:600;padding:1px 7px;border-radius:999px;background:#0aa2c01a;color:#0d7d8f;border:1px solid #0aa2c055;vertical-align:middle}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:18px}
.armtog{display:inline-flex;background:#fff;border:1px solid var(--line);border-radius:11px;padding:4px;box-shadow:0 14px 34px -30px #1b2b57}
.armtog button{font-family:"Space Grotesk";font-weight:600;font-size:13.5px;cursor:pointer;border:none;border-radius:8px;padding:9px 15px;background:transparent;color:var(--muted);transition:.15s}
.armtog button.on{color:#fff;background:var(--fam)}
.provtxt{margin-left:auto;font-family:"IBM Plex Mono";font-size:11.5px;color:var(--faint)}
.note{display:flex;gap:10px;align-items:flex-start;background:linear-gradient(180deg,#f5fcff,#eef7fd);border:1px dashed #0aa2c088;border-radius:var(--r);padding:11px 14px;margin:14px 0;font-size:12.5px;color:#0d5b6a}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:20px 0}
.card{background:#fff;border:1px solid var(--line);border-radius:var(--r);padding:13px 15px;box-shadow:0 14px 34px -28px #1b2b57}
.card .k{font-size:12px;color:var(--muted)}.card .v{font-family:"Space Grotesk";font-weight:700;font-size:23px;margin-top:3px;line-height:1.1}
.card .u{font-size:11px;color:var(--faint);margin-top:2px}.card.hi .v{color:var(--fam)}
section{margin:26px 0}.sec-h{display:flex;align-items:baseline;gap:10px;margin-bottom:11px}
.sec-h h2{font-family:"Space Grotesk";font-weight:600;font-size:17px;margin:0}.sec-h .sub{color:var(--muted);font-size:12.5px;margin-left:auto}
.panel{background:#fff;border:1px solid var(--line);border-radius:var(--r);padding:16px;box-shadow:0 18px 44px -34px #1b2b57}
.bar-row{display:grid;grid-template-columns:110px 1fr 54px;align-items:center;gap:10px;margin:8px 0;font-size:13px}
.bar-row .rk{color:var(--muted)}.bar{height:18px;background:var(--panel-2);border:1px solid var(--line);border-radius:5px;overflow:hidden}
.bar>span{display:block;height:100%;border-radius:4px}.bar-row .cn{font-family:"IBM Plex Mono";text-align:right}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600;font-size:12px}td.num,th.num{font-family:"IBM Plex Mono";text-align:right}
.sig{font-family:"IBM Plex Mono";font-size:11px;padding:2px 7px;border-radius:6px}.sig.yes{background:#12b76a1c;color:#0a7d47}.sig.no{background:#93a1bf1c;color:var(--faint)}
.verdict{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.vbox{background:var(--panel-2);border:1px solid var(--line);border-radius:9px;padding:13px}
.vbox .lab{font-size:11.5px;color:var(--muted)}.vbox .big{font-family:"Space Grotesk";font-weight:700;font-size:20px;margin:5px 0 2px}
.vbox .sub{font-size:11px;color:var(--faint);font-family:"IBM Plex Mono"}
.ci{position:relative;height:30px;margin:10px 0 3px;background:linear-gradient(90deg,#f2f6fd,#eaf0fb);border:1px solid var(--line);border-radius:7px}
.ci .zero{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--faint)}.ci .zero::after{content:"0";position:absolute;top:100%;left:-3px;font:10px "IBM Plex Mono";color:var(--faint)}
.ci .band{position:absolute;top:8px;height:14px;background:var(--fam);border-radius:5px;opacity:.85}.ci .mean{position:absolute;top:4px;width:3px;height:22px;background:var(--ink);border-radius:2px}
.readme{font-size:12.5px;color:var(--muted);margin-top:12px;padding-top:12px;border-top:1px dashed var(--line)}.readme b{color:var(--ink)}
.tools{margin-top:8px;display:flex;gap:6px;flex-wrap:wrap}.tools span{font-family:"IBM Plex Mono";font-size:11px;background:#0aa2c012;border:1px solid #0aa2c033;color:var(--accent-dim);padding:1px 7px;border-radius:999px}
footer{color:var(--faint);font-size:12px;text-align:center;padding:30px 0 22px;border-top:1px solid var(--line);margin-top:30px}
@media(max-width:640px){.brand h1{font-size:22px}}
</style></head><body><div class="wrap">"""

BODY = """
<header>
  <div class="back"><a href="index.html">← RegulatoryScope index</a> · <a href="RegulatoryScope_Combined.html">combined atlas</a></div>
  <div class="brand"><span class="dot"></span><h1>__NAME__</h1><span class="code mono">__CODE__</span>
    <span class="prov-badge">COMPUTED · demo cohort</span></div>
  <div class="cls">__CLS__ · __TIP__</div>
  <p class="lede">__METHOD__</p>
  <div class="controls">
    <div class="armtog" id="armtog">
      <button class="on" data-arm="mpra_transcription">MPRA · transcription</button>
      <button data-arm="caqtl_accessibility">caQTL · accessibility</button>
    </div>
    <span class="provtxt" id="provtxt"></span>
  </div>
</header>
<div class="note"><span>✓</span><div><b>Computed by the RegulatoryScope pipeline</b>
  (features.py → decompose.py) on a synthetic demo cohort — real algorithms and real statistics,
  illustrative data. Re-run <span class="mono">run_pipeline.py</span> on your MPRA/caQTL variants and
  regenerate to make these numbers real.</div></div>
<div class="cards" id="cards"></div>
<section><div class="sec-h"><h2>Event breakdown</h2><span class="sub">how each variant touches the structure</span></div>
  <div class="panel" id="events"></div></section>
<section><div class="sec-h"><h2>Effect size by event</h2><span class="sub">|effect| vs structure-neutral variants</span></div>
  <div class="panel"><table id="cat"></table><p class="readme" id="catnote"></p></div></section>
<section id="subsec" style="display:none"><div class="sec-h"><h2>__SUBTITLE__</h2><span class="sub">incremental R² of each sub-feature</span></div>
  <div class="panel" id="subs"></div></section>
<section><div class="sec-h"><h2>Is the signal real?</h2><span class="sub">rigor checks against confounds and chance</span></div>
  <div class="panel" id="rigor"></div></section>
<section><div class="sec-h"><h2>Method &amp; upgrade path</h2><span class="sub">how it is computed and the gold-standard tool</span></div>
  <div class="panel"><div class="readme" style="border:none;margin:0;padding:0">__METHOD__
    <div class="tools">__TOOLS__</div></div></div></section>
<footer>RegulatoryScope · __NAME__ (__CODE__) · computed in your browser from embedded pipeline output</footer>
</div>
<script>
const FAM = __FAM_JSON__;
let ARM = "mpra_transcription";
const f3=x=>x==null?"—":x.toFixed(3), f4=x=>x==null?"—":x.toFixed(4);
const pct=(x,d=1)=>x==null?"—":(100*x).toFixed(d)+"%";
const pf=p=>p==null?"—":(p<1e-4?p.toExponential(1):p.toFixed(4));
const sci=x=>x==null?"—":(Math.abs(x)<1e-4?x.toExponential(1):x.toFixed(4));
function armd(){return FAM.arms[ARM];}
function render(){
  const d=armd(); if(!d){return;}
  const b=d.block, m=FAM.arm_meta[ARM];
  const altering=FAM.verbs.filter(v=>v!==FAM.ref_verb).reduce((s,v)=>s+(b.events[v]||0),0);
  const r=b.rigor;
  document.getElementById("provtxt").textContent="embedded · demo run · "+m.short;
  document.getElementById("cards").innerHTML=[
    ["variants scored",d.n_variants.toLocaleString(),m.short+" · active"],
    ["structure-altering",altering.toLocaleString(),FAM.verbs.filter(v=>v!==FAM.ref_verb).join(" / ")],
    ["incremental ΔR²",sci(b.dR2),"over base + TF",true],
    ["partial ρ | GC",f3(r.partial),"p = "+pf(r.pp)],
    ["bootstrap 95% CI","["+sci(r.boot_ci[0])+", "+sci(r.boot_ci[1])+"]",(r.boot_ci[0]>0?"excludes 0":"touches 0")],
    ["GC-matched",r.gc,"p = "+pf(r.gcp)]
  ].map(c=>`<div class="card${c[3]?' hi':''}"><div class="k">${c[0]}</div><div class="v">${c[1]}</div><div class="u">${c[2]}</div></div>`).join("");

  // events
  const maxC=Math.max(...FAM.verbs.map(v=>b.events[v]||0),1);
  document.getElementById("events").innerHTML=FAM.verbs.map(v=>{
    const c=b.events[v]||0, op=(v==="disrupt"||v==="loss")?1:(v==="create"||v==="gain")?0.62:0.82;
    return `<div class="bar-row"><span class="rk">${v}</span><div class="bar"><span style="width:${100*c/maxC}%;background:var(--fam);opacity:${op}"></span></div><span class="cn">${c}</span></div>`;
  }).join("");

  // effect table
  let rows=`<thead><tr><th>Event</th><th class="num">n</th><th class="num">median |effect|</th><th class="num">mean signed</th><th class="num">p vs ref</th><th>vs ref</th></tr></thead><tbody>`;
  FAM.verbs.forEach(v=>{const c=b.effect[v]; if(!c)return; const sig=(c.p!=null&&c.p<0.05);
    rows+=`<tr><td><b>${v}</b></td><td class="num">${c.n}</td><td class="num">${f4(c.med)}</td>
    <td class="num" style="color:${c.signed>=0?'var(--good)':'var(--danger)'}">${c.signed>=0?'+':''}${f4(c.signed)}</td>
    <td class="num">${pf(c.p)}</td><td>${c.p==null?'<span class="sig no">n too small</span>':sig?'<span class="sig yes">p &lt; 0.05</span>':'<span class="sig no">n.s.</span>'}</td></tr>`;});
  rows+=`<tr style="border-top:2px solid var(--line)"><td style="color:var(--muted)">reference (${FAM.ref_verb})</td><td class="num">${b.events[FAM.ref_verb]||"—"}</td><td class="num">${f4(d.none_med)}</td><td class="num">—</td><td class="num">—</td><td>—</td></tr></tbody>`;
  document.getElementById("cat").innerHTML=rows;
  document.getElementById("catnote").innerHTML=`Effect on the ${m.short} scale (${m.unit}). Reference = structure-neutral <b>${FAM.ref_verb}</b> variants. `+
    (FAM.style==="continuous"?"gain/loss = upper/lower tercile of the allelic delta.":"create/disrupt/modulate = structure gained / lost / altered by the single-base change.");

  // subfeatures
  if(b.subfeatures){
    document.getElementById("subsec").style.display="";
    const ents=Object.entries(b.subfeatures).sort((a,c)=>Math.abs(c[1].dR2)-Math.abs(a[1].dR2));
    const mx=Math.max(...ents.map(e=>Math.abs(e[1].dR2)),1e-9);
    document.getElementById("subs").innerHTML=ents.map(([k,v])=>
      `<div class="bar-row"><span class="rk">${k}</span><div class="bar"><span style="width:${100*Math.abs(v.dR2)/mx}%;background:var(--fam)"></span></div><span class="cn">${sci(v.dR2)}</span></div>`).join("")+
      `<p class="readme">Per-sub-feature incremental R² over base + TF. The largest bar is the component doing the work; swap in DNAshapeR/non-B DB per-base values to sharpen these.</p>`;
  } else { document.getElementById("subsec").style.display="none"; }

  // rigor
  const ciPos=r.boot_ci[0]>0, lo=Math.min(0,r.boot_ci[0]), hi=r.boot_ci[1]*1.1||1, span=(hi-lo)||1;
  const zx=100*(0-lo)/span, bl=100*(r.boot_ci[0]-lo)/span, bw=100*(r.boot_ci[1]-r.boot_ci[0])/span, mx2=100*(r.boot_mean-lo)/span;
  const gcHold=r.gc==="holds";
  document.getElementById("rigor").innerHTML=`<div class="verdict">
    <div class="vbox"><div class="lab">partial correlation | GC</div><div class="big" style="color:${r.pp!=null&&r.pp<0.05?'var(--good)':'var(--faint)'}">ρ = ${f3(r.partial)}</div><div class="sub">p = ${pf(r.pp)}</div>
      <div class="readme" style="margin-top:10px;padding-top:10px">Feature score still tracks effect after regressing out GC.</div></div>
    <div class="vbox"><div class="lab">bootstrap ΔR²</div><div class="ci"><div class="zero" style="left:${zx}%"></div><div class="band" style="left:${bl}%;width:${bw}%"></div><div class="mean" style="left:${mx2}%"></div></div><div class="sub">mean ${sci(r.boot_mean)} · [${sci(r.boot_ci[0])}, ${sci(r.boot_ci[1])}]</div>
      <div class="readme" style="margin-top:10px;padding-top:10px">${ciPos?"Interval <b>above zero</b> — reproducible.":"Interval <b>touches zero</b> — suggestive."}</div></div>
    <div class="vbox"><div class="lab">direction test</div><div class="big">${pct(r.dir)}</div><div class="sub">binom p = ${pf(r.dirp)}</div>
      <div class="readme" style="margin-top:10px;padding-top:10px">${r.dir>0.55?"Effect sign follows the predicted direction more than chance.":"Sign near chance — magnitude effect more than directional."}</div></div>
    <div class="vbox"><div class="lab">GC-matched control</div><div class="big" style="color:${gcHold?'var(--good)':'var(--faint)'}">${r.gc}</div><div class="sub">p = ${pf(r.gcp)}</div>
      <div class="readme" style="margin-top:10px;padding-top:10px">${gcHold?"Survives GC matching — the toughest confound control.":"Does not clearly separate once GC is matched — read cautiously."}</div></div>
  </div>`;
  document.querySelectorAll("#armtog button").forEach(x=>x.classList.toggle("on",x.dataset.arm===ARM));
}
document.getElementById("armtog").addEventListener("click",e=>{const b=e.target.closest("button");if(!b)return;ARM=b.dataset.arm;render();});
render();
</script></body></html>"""

def render_dashboard(code, bundle):
    fam = build_fam_obj(code, bundle); meta = FAM_META[code]
    html = HEAD.replace("__TITLE__", f"RegulatoryScope · {meta['name']}").replace("__COLOR__", meta["color"])
    body = BODY
    tools_html = "".join(f"<span>{t}</span>" for t in meta["tools"])
    repl = {"__NAME__": meta["name"], "__CODE__": code, "__CLS__": meta["cls"], "__TIP__": meta["tip"],
            "__METHOD__": meta["method"], "__TOOLS__": tools_html,
            "__SUBTITLE__": meta["subtitle"] or "Sub-features",
            "__FAM_JSON__": json.dumps(fam)}
    for k, v in repl.items():
        body = body.replace(k, v)
    return html + body

# ---------------------------------------------------------------- index page
INDEX = """<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0"><title>RegulatoryScope · index</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
<style>
:root{--line:#d7e0f0;--ink:#101c33;--muted:#59688a;--faint:#93a1bf;--accent:#0aa2c0;--accent-dim:#0d7d8f;--r:12px;}
*{box-sizing:border-box}body{margin:0;background:#eef3fb;color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;line-height:1.5;
background-image:radial-gradient(1100px 560px at 86% -12%,#cdf2e6cc,transparent 62%),radial-gradient(1000px 520px at -12% 4%,#d3e4ffcc,transparent 62%),radial-gradient(900px 520px at 50% 128%,#ece1ffcc,transparent 60%);background-attachment:fixed}
body::before{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;background-image:radial-gradient(#b9c6de55 1px,transparent 1.4px);background-size:26px 26px;mask:linear-gradient(180deg,#000 60%,transparent)}
.wrap{max-width:1060px;margin:0 auto;padding:0 22px;position:relative;z-index:1}.mono{font-family:"IBM Plex Mono",monospace}
a{text-decoration:none;color:inherit}
header{padding:34px 0 6px}h1{font-family:"Space Grotesk";font-weight:700;font-size:30px;letter-spacing:-.02em;margin:0;
background:linear-gradient(90deg,#0e2b6b,#0aa2c0 55%,#12b76a);-webkit-background-clip:text;background-clip:text;color:transparent}
.lede{color:var(--muted);max-width:760px;margin:10px 0 0;font-size:15px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:14px;margin:24px 0}
.tile{display:block;background:#fff;border:1px solid var(--line);border-radius:var(--r);padding:16px;box-shadow:0 16px 40px -32px #1b2b57;transition:.15s;border-top:4px solid var(--c)}
.tile:hover{transform:translateY(-3px);box-shadow:0 24px 50px -30px #1b2b57}
.tile .h{display:flex;align-items:center;gap:9px}.tile .sw{width:14px;height:14px;border-radius:4px;background:var(--c)}
.tile h3{font-family:"Space Grotesk";font-size:16px;margin:0}.tile .code{font-family:"IBM Plex Mono";font-size:11px;color:var(--muted);margin-left:auto}
.tile .cls{font-family:"IBM Plex Mono";font-size:11px;color:var(--faint);margin:6px 0 10px}
.tile .stat{display:flex;justify-content:space-between;font-size:12px;color:var(--muted);border-top:1px dashed var(--line);padding-top:9px}
.tile .stat b{font-family:"Space Grotesk";color:var(--c)}
.big-tile{grid-column:1/-1;background:linear-gradient(180deg,#fff,#eef5fe);border-top:4px solid #0aa2c0}
.big-tile h3{font-size:19px}.big-tile p{color:var(--muted);font-size:13.5px;margin:6px 0 0}
.sec{font-family:"Space Grotesk";font-weight:600;font-size:14px;color:var(--muted);margin:26px 0 2px;text-transform:uppercase;letter-spacing:.05em}
.note{background:#f5fcff;border:1px dashed #0aa2c088;border-radius:var(--r);padding:12px 15px;margin:16px 0;font-size:13px;color:#0d5b6a}
footer{color:var(--faint);font-size:12px;text-align:center;padding:34px 0 22px;border-top:1px solid var(--line);margin-top:30px}
</style></head><body><div class="wrap">
<header><h1>RegulatoryScope</h1>
<p class="lede">A model reads DNA to predict regulatory activity; in-silico mutagenesis asks what a single-base change
does; then we decompose that effect across the full DNA-structure feature space. Each family below is a standalone
dashboard computed by the pipeline. Two arms throughout: MPRA (transcription) and caQTL (accessibility).</p></header>
<div class="note">✓ Family dashboards show <b>COMPUTED</b> output (real algorithms + statistics) on a synthetic demo cohort.
Re-run <span class="mono">run_pipeline.py</span> on your variants and <span class="mono">generate_dashboards.py</span> to refresh them on real data.</div>
__BIG__
<div class="sec">Extended structural families</div>
<div class="grid">__TILES__</div>
<footer>RegulatoryScope · index · open any tile · combined atlas holds the measured G4/i-Motif run</footer>
</div></body></html>"""

def render_index(bundle, codes):
    tiles = ""
    for code in codes:
        meta = FAM_META[code]; a = bundle["arms"]["mpra_transcription"]["families"][code]
        c = bundle["arms"]["caqtl_accessibility"]["families"][code]
        tiles += f"""<a class="tile" style="--c:{meta['color']}" href="dash_{code}.html">
  <div class="h"><span class="sw"></span><h3>{meta['name']}</h3><span class="code">{code}</span></div>
  <div class="cls">{meta['cls']}</div>
  <div class="stat"><span>MPRA ΔR²</span><b>{_sci(a['dR2'])}</b></div>
  <div class="stat" style="border:none;padding-top:4px"><span>caQTL ΔR²</span><b>{_sci(c['dR2'])}</b></div></a>"""
    sdm = """<a class="tile big-tile" href="RegulatoryScope_SDM_Studio.html" style="border-top-color:#0aa2c0">
  <div class="h"><span class="sw" style="background:linear-gradient(90deg,#0aa2c0,#12b76a)"></span><h3>SDM Studio — site-directed mutagenesis impact explorer</h3></div>
  <p>Enumerate every single-base mutation, predict each one's impact across all structural families + TF motifs,
  and read the per-mutation breakdown with cited references. Exportable table, filters, and per-SDM detail.</p></a>"""
    big = """<a class="tile big-tile" href="RegulatoryScope_Combined.html">
  <div class="h"><span class="sw" style="background:linear-gradient(90deg,#f59e0b,#2e6ff2)"></span><h3>Combined atlas — DNA-reading model + all 8 families</h3></div>
  <p>The integrative view: measured G4 (PGS) + i-Motif (PIS), the nested decomposition ladder, in-silico mutagenesis,
  and the six extended families in one place. Load a <span class="mono">results.json</span> to make every family computed.</p></a>"""
    return INDEX.replace("__BIG__", sdm + big).replace("__TILES__", tiles)

def _sci(x):
    if x is None: return "—"
    return f"{x:.2e}" if abs(x) < 1e-4 else f"{x:.4f}"

def main():
    bundle = json.load(open("web/results_demo.json"))
    codes = ["SHP", "SID", "ZDA", "RLP", "CTX", "NUC"]
    os.makedirs("web", exist_ok=True)
    for code in codes:
        with open(f"web/dash_{code}.html", "w") as fh:
            fh.write(render_dashboard(code, bundle))
        print("wrote web/dash_%s.html" % code)
    with open("web/index.html", "w") as fh:
        fh.write(render_index(bundle, codes))
    print("wrote web/index.html")

if __name__ == "__main__":
    main()
