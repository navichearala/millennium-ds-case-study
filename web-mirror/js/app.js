/* ------------------------------------------------------------------
   Candidate Resume Search Platform — browser mirror.

   This is a faithful port of the Streamlit application in the case-study
   submission: identical scoring maths, identical filter semantics and the
   same four views, rendered client-side so it previews live without a
   Python server. The data is the real pipeline output.
   ------------------------------------------------------------------ */
import { SERIES, barChart, donutChart, heatmap, histogram, radar, scatter } from "./charts.js";

const DATA = window.PLATFORM_DATA;
const CANDIDATES = DATA.candidates;
const RAW = DATA.raw_text;

/* ------------------------------------------------------------ taxonomies */
const SECTORS = [
  "Technology", "Media & Telecom", "Healthcare", "Financial Services", "Energy",
  "Industrials", "Consumer", "Credit", "Macro / Rates & FX", "Materials",
  "Multi-Sector / Generalist",
];
const REGIONS = ["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East & Africa"];
const STRATEGIES = ["Fundamental", "Systematic / Quantitative", "Hybrid"];
const SENIORITY = ["Analyst", "Senior Analyst", "Associate", "Lead Analyst", "Portfolio Manager"];
const SIDES = ["Buy-Side", "Sell-Side", "Private Markets", "Corporate", "Academic"];
const DEGREES = ["Doctorate / Medical", "Master's / MBA", "Bachelor's", "Unknown"];

const PRESETS = {
  "— No requisition (browse all) —": {},
  "US Healthcare Fundamental Analyst (5–10 yrs)": {
    region: "North America", strategy: "Fundamental", sectors: ["Healthcare"],
    minYears: 5, maxYears: 10, seniority: ["Analyst", "Senior Analyst", "Associate"],
  },
  "US TMT Fundamental L/S Analyst (4–12 yrs)": {
    region: "North America", strategy: "Fundamental",
    sectors: ["Technology", "Media & Telecom"], minYears: 4, maxYears: 12,
    seniority: ["Analyst", "Senior Analyst", "Associate"],
  },
  "Europe Systematic / Quant Researcher (2–8 yrs)": {
    region: "Europe", strategy: "Systematic / Quantitative",
    sectors: ["Macro / Rates & FX", "Credit"], minYears: 2, maxYears: 8,
  },
  "APAC Healthcare Research Analyst (6–15 yrs)": {
    region: "Asia-Pacific", strategy: "Fundamental", sectors: ["Healthcare"],
    minYears: 6, maxYears: 15,
  },
  "Global Credit / Macro Analyst (3–10 yrs)": {
    sectors: ["Credit", "Macro / Rates & FX"], minYears: 3, maxYears: 10,
  },
};

const DEFAULT_WEIGHTS = { sector: 35, region: 20, strategy: 20, experience: 15, credentials: 10 };

/* ------------------------------------------------------------- row model */
const ROWS = CANDIDATES.map((c) => ({
  id: c.candidate_id,
  name: (c.honorific ? c.honorific + " " : "") + c.full_name,
  region: c.region,
  location: [c.location_city, c.location_country].filter(Boolean).join(", ") || "Not stated",
  employer: c.current_employer || "No current employer stated",
  title: c.current_title || "No current title stated",
  employed: !!c.is_currently_employed,
  strategy: c.primary_strategy_type,
  side: c.primary_market_side,
  firmType: c.primary_firm_type,
  seniority: c.seniority_level,
  sectors: c.sectors_covered || [],
  markets: c.geographic_markets_covered || [],
  years: Number(c.computed_years_experience || 0),
  selfYears: c.self_reported_years_experience,
  coverage: c.max_coverage_universe,
  degreeTier: c.highest_degree_tier || "Unknown",
  degree: c.highest_degree,
  cfa: !!c.has_cfa,
  md: !!c.has_medical_degree,
  languages: c.languages_spoken || [],
  tools: c.tools_and_platforms || [],
  employers: c.employer_list || [],
  quality: Number(c.data_quality_score || 0),
  flags: c.data_quality_flags || [],
  gaps: c.career_gaps || [],
  confidence: c.extraction_confidence,
  agency: c.source_agency || "",
  sourceFile: c.source_file,
  searchable: c.searchable_text || "",
  record: c,
}));

/* -------------------------------------------------------------- app state */
const state = {
  preset: Object.keys(PRESETS)[0],
  region: "Any",
  strategy: "Any",
  sectors: [],
  sectorLogic: "any",
  minYears: 0,
  maxYears: 20,
  enforceYears: false,
  seniority: [],
  side: [],
  degree: [],
  cfa: false,
  md: false,
  employed: false,
  quality: 0,
  keyword: "",
  weights: { ...DEFAULT_WEIGHTS },
  view: "cards",
  scope: "pool",
  origin: "all",
  tab: "results",
  selected: null,
  detailTab: "career",
  compare: [],
};

/* ---------------------------------------------------------------- scoring */
function scoreCandidate(row) {
  const req = state;
  const parts = {};

  // Sector: proportion of requested sectors actually covered. A generalist mandate
  // counts as credible-but-unproven rather than a miss.
  if (req.sectors.length) {
    const overlap = req.sectors.filter((s) => row.sectors.includes(s)).length;
    parts.sector =
      overlap === 0 && row.sectors.includes("Multi-Sector / Generalist")
        ? 0.55
        : overlap / req.sectors.length;
  } else {
    parts.sector = 1;
  }

  // Region: exact match, with partial credit for researching the market from elsewhere.
  if (req.region !== "Any") {
    if (row.region === req.region) parts.region = 1;
    else {
      const hints = {
        "North America": ["united states", "us", "north america"],
        Europe: ["europe", "united kingdom", "emea", "france", "germany"],
        "Asia-Pacific": ["asia", "china", "india", "japan", "hong kong"],
        "Latin America": ["latam", "latin america", "brazil"],
        "Middle East & Africa": ["middle east", "africa", "emea"],
      }[req.region] || [];
      const blob = row.markets.join(" ").toLowerCase();
      parts.region = hints.some((h) => blob.includes(h)) ? 0.5 : 0;
    }
  } else {
    parts.region = 1;
  }

  // Strategy: hybrid profiles get partial credit for either mandate.
  if (req.strategy !== "Any") {
    parts.strategy = row.strategy === req.strategy ? 1 : row.strategy === "Hybrid" ? 0.7 : 0;
  } else {
    parts.strategy = 1;
  }

  // Experience: full credit inside the band, linear decay outside it — one year
  // outside a band is not a non-match.
  const lo = req.minYears;
  const hi = req.maxYears;
  if (row.years >= lo && row.years <= hi) parts.experience = 1;
  else {
    const distance = row.years < lo ? lo - row.years : row.years - hi;
    parts.experience = Math.max(0, 1 - distance / 5);
  }

  // Credentials: a small bonus pool, never a gate.
  let cred = ["Master's / MBA", "Doctorate / Medical"].includes(row.degreeTier) ? 0.35 : 0.15;
  if (row.cfa) cred += 0.25;
  if (row.md && req.sectors.includes("Healthcare")) cred += 0.15;
  const wantedSeniority = req.seniority.length ? req.seniority : PRESETS[req.preset]?.seniority || [];
  if (!wantedSeniority.length || wantedSeniority.includes(row.seniority)) cred += 0.25;
  parts.credentials = Math.min(cred, 1);

  const totalW = Object.values(req.weights).reduce((a, b) => a + b, 0) || 1;
  const contributions = {};
  let score = 0;
  for (const key of Object.keys(parts)) {
    const c = (parts[key] * (req.weights[key] || 0)) / totalW * 100;
    contributions[key] = Math.round(c * 10) / 10;
    score += c;
  }
  return { score: Math.round(score * 10) / 10, contributions };
}

function scoringOn() {
  return state.sectors.length > 0 || state.region !== "Any" || state.strategy !== "Any";
}

/* --------------------------------------------------------------- filtering */
function filteredRows() {
  const kw = state.keyword
    .split(",")
    .map((t) => t.trim().toLowerCase())
    .filter(Boolean);

  let rows = ROWS.filter((r) => {
    if (state.region !== "Any" && r.region !== state.region) return false;
    if (state.strategy !== "Any" && !(r.strategy === state.strategy || r.strategy === "Hybrid"))
      return false;
    if (state.sectors.length) {
      const hit =
        state.sectorLogic === "all"
          ? state.sectors.every((s) => r.sectors.includes(s))
          : state.sectors.some((s) => r.sectors.includes(s));
      if (!hit) return false;
    }
    if (state.enforceYears && (r.years < state.minYears || r.years > state.maxYears)) return false;
    if (state.seniority.length && !state.seniority.includes(r.seniority)) return false;
    if (state.side.length && !state.side.includes(r.side)) return false;
    if (state.degree.length && !state.degree.includes(r.degreeTier)) return false;
    if (state.cfa && !r.cfa) return false;
    if (state.md && !r.md) return false;
    if (state.employed && !r.employed) return false;
    if (r.quality < state.quality) return false;
    if (kw.length && !kw.every((t) => r.searchable.includes(t))) return false;
    return true;
  }).map((r) => ({ ...r, ...scoreCandidate(r) }));

  rows.sort((a, b) => (scoringOn() ? b.score - a.score || b.years - a.years : b.years - a.years));
  return rows;
}

/* ----------------------------------------------------------------- helpers */
const $ = (sel) => document.querySelector(sel);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function chipGroup(container, options, selected, onToggle) {
  container.innerHTML = "";
  options.forEach((opt) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.textContent = opt;
    b.setAttribute("aria-pressed", selected.includes(opt) ? "true" : "false");
    b.addEventListener("click", () => onToggle(opt));
    container.appendChild(b);
  });
}

function segGroup(container, value, onPick) {
  container.querySelectorAll("button").forEach((b) => {
    b.setAttribute("aria-pressed", b.dataset.value === value ? "true" : "false");
    b.onclick = () => onPick(b.dataset.value);
  });
}

function downloadCSV(filename, rows) {
  const csv = rows
    .map((r) => r.map((v) => `"${String(v ?? "").replace(/"/g, '""')}"`).join(","))
    .join("\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function chartCard(title, note, svg, legend) {
  const card = document.createElement("div");
  card.className = "chart-card";
  const h = document.createElement("div");
  h.className = "chart-title";
  h.textContent = title;
  const p = document.createElement("p");
  p.className = "chart-note";
  p.textContent = note;
  card.append(h, p, svg);
  if (legend?.length) {
    const l = document.createElement("div");
    l.className = "legend";
    legend.forEach(({ label, color }) => {
      const s = document.createElement("span");
      const i = document.createElement("i");
      i.style.background = color;
      s.append(i, document.createTextNode(label));
      l.appendChild(s);
    });
    card.appendChild(l);
  }
  return card;
}

/* -------------------------------------------------------------- rendering */
function renderControls() {
  const presetSel = $("#preset");
  if (!presetSel.options.length) {
    Object.keys(PRESETS).forEach((k) => presetSel.add(new Option(k, k)));
    ["Any", ...REGIONS].forEach((k) => $("#region").add(new Option(k, k)));
    ["Any", ...STRATEGIES].forEach((k) => $("#strategy").add(new Option(k, k)));

    const wrap = $("#weights");
    Object.keys(DEFAULT_WEIGHTS).forEach((key) => {
      const field = document.createElement("div");
      field.className = "field";
      field.innerHTML = `<label for="w-${key}">${key[0].toUpperCase() + key.slice(1)}</label>
        <div class="range-row"><span class="range-val" id="wv-${key}"></span>
        <input id="w-${key}" type="range" min="0" max="50" step="5"></div>`;
      wrap.appendChild(field);
      const input = field.querySelector("input");
      input.value = state.weights[key];
      input.addEventListener("input", () => {
        state.weights[key] = Number(input.value);
        render();
      });
    });
  }

  presetSel.value = state.preset;
  presetSel.onchange = () => applyPreset(presetSel.value);
  $("#region").value = state.region;
  $("#region").onchange = (e) => {
    state.region = e.target.value;
    render();
  };
  $("#strategy").value = state.strategy;
  $("#strategy").onchange = (e) => {
    state.strategy = e.target.value;
    render();
  };

  chipGroup($("#sectors"), SECTORS, state.sectors, (v) => {
    toggle(state.sectors, v);
    render();
  });
  chipGroup($("#seniority"), SENIORITY, state.seniority, (v) => {
    toggle(state.seniority, v);
    render();
  });
  chipGroup($("#side"), SIDES, state.side, (v) => {
    toggle(state.side, v);
    render();
  });
  chipGroup($("#degree"), DEGREES, state.degree, (v) => {
    toggle(state.degree, v);
    render();
  });

  segGroup($("#sectorLogic"), state.sectorLogic, (v) => {
    state.sectorLogic = v;
    render();
  });
  segGroup($("#viewMode"), state.view, (v) => {
    state.view = v;
    render();
  });
  segGroup($("#scopeMode"), state.scope, (v) => {
    state.scope = v;
    render();
  });
  segGroup($("#originMode"), state.origin, (v) => {
    state.origin = v;
    render();
  });

  const bind = (id, key, label, fmt) => {
    const input = $(id);
    input.value = state[key];
    $(label).textContent = fmt(state[key]);
    input.oninput = () => {
      state[key] = Number(input.value);
      if (key === "minYears" && state.minYears > state.maxYears) state.maxYears = state.minYears;
      if (key === "maxYears" && state.maxYears < state.minYears) state.minYears = state.maxYears;
      render();
    };
  };
  bind("#yearsMin", "minYears", "#yearsMinVal", (v) => v.toFixed(1));
  bind("#yearsMax", "maxYears", "#yearsMaxVal", (v) => v.toFixed(1));
  bind("#quality", "quality", "#qualityVal", (v) => v.toFixed(2));

  $("#enforceYears").checked = state.enforceYears;
  $("#enforceYears").onchange = (e) => {
    state.enforceYears = e.target.checked;
    render();
  };
  ["cfa", "md", "employed"].forEach((k) => {
    const box = $("#" + k);
    box.checked = state[k];
    box.onchange = (e) => {
      state[k] = e.target.checked;
      render();
    };
  });
  const kwInput = $("#keyword");
  if (document.activeElement !== kwInput) kwInput.value = state.keyword;
  kwInput.oninput = (e) => {
    state.keyword = e.target.value.toLowerCase();
    render();
  };

  Object.keys(DEFAULT_WEIGHTS).forEach((key) => {
    $("#w-" + key).value = state.weights[key];
    $("#wv-" + key).textContent = state.weights[key];
  });

  $("#reset").onclick = () => {
    applyPreset(Object.keys(PRESETS)[0]);
  };
}

function toggle(arr, value) {
  const i = arr.indexOf(value);
  if (i === -1) arr.push(value);
  else arr.splice(i, 1);
}

function applyPreset(name) {
  const p = PRESETS[name] || {};
  Object.assign(state, {
    preset: name,
    region: p.region || "Any",
    strategy: p.strategy || "Any",
    sectors: [...(p.sectors || [])],
    minYears: p.minYears ?? 0,
    maxYears: p.maxYears ?? 20,
    enforceYears: false,
    seniority: [],
    side: [],
    degree: [],
    cfa: false,
    md: false,
    employed: false,
    quality: 0,
    keyword: "",
    selected: null,
    compare: [],
  });
  render();
}

function pills(row) {
  const out = [];
  row.sectors.slice(0, 5).forEach((s) => out.push(`<span class="pill">${esc(s)}</span>`));
  out.push(
    `<span class="pill ${row.strategy === "Unclear" ? "warn" : "good"}">${esc(row.strategy)}</span>`
  );
  out.push(`<span class="pill">${esc(row.side)}</span>`);
  if (row.cfa) out.push('<span class="pill good">CFA</span>');
  if (row.md) out.push('<span class="pill good">MD / MBBS</span>');
  if (!row.employed) out.push('<span class="pill warn">Not currently employed</span>');
  if (row.flags.length)
    out.push(`<span class="pill warn">${row.flags.length} data flags</span>`);
  if (row.agency) out.push(`<span class="pill warn">via ${esc(row.agency)}</span>`);
  return out.join("");
}

function renderResults(rows) {
  const host = $("#results");
  const note = $("#resultsNote");
  const scored = scoringOn();

  note.textContent = scored
    ? ""
    : "No requisition criteria set, so candidates are ranked by experience and the right-hand figure shows data quality. Choose a preset mandate, region, approach or sector to rank by match score.";

  if (!rows.length) {
    host.innerHTML = `<div class="empty"><strong>No candidates match these criteria</strong>
      <p>Try widening the experience band, switching sector match to “Any of these”, clearing the
      keyword search, or turning off the hard experience filter.</p></div>`;
    $("#detail").innerHTML = "";
    return;
  }

  if (state.view === "table") {
    const head = [
      "Candidate", ...(scored ? ["Match"] : []), "Region", "Location", "Current firm", "Title",
      "Approach", "Side", "Seniority", "Yrs", "Coverage", "Degree", "CFA", "Quality", "Flags",
    ];
    host.innerHTML = `<div class="table-wrap"><table><thead><tr>${head
      .map((h) => `<th class="${["Match", "Yrs", "Coverage", "Quality", "Flags"].includes(h) ? "num" : ""}">${h}</th>`)
      .join("")}</tr></thead><tbody>${rows
      .map(
        (r) => `<tr data-id="${r.id}">
        <td><strong>${esc(r.name)}</strong></td>
        ${scored ? `<td class="num">${r.score.toFixed(1)}</td>` : ""}
        <td>${esc(r.region)}</td><td>${esc(r.location)}</td><td>${esc(r.employer)}</td>
        <td class="wrap">${esc(r.title)}</td><td>${esc(r.strategy)}</td><td>${esc(r.side)}</td>
        <td>${esc(r.seniority)}</td><td class="num">${r.years.toFixed(1)}</td>
        <td class="num">${r.coverage ?? "—"}</td><td>${esc(r.degreeTier)}</td>
        <td>${r.cfa ? "Yes" : "—"}</td><td class="num">${r.quality.toFixed(2)}</td>
        <td class="num">${r.flags.length}</td></tr>`
      )
      .join("")}</tbody></table></div>`;
    host.querySelectorAll("tbody tr").forEach((tr) => {
      tr.style.cursor = "pointer";
      tr.onclick = () => {
        state.selected = tr.dataset.id;
        render();
      };
    });
  } else {
    host.innerHTML = `<div class="cards">${rows
      .map(
        (r) => `<button class="card" data-id="${r.id}" aria-current="${state.selected === r.id}">
          <span>
            <span class="card-name">${esc(r.name)}</span>
            <span class="card-role" style="display:block">${esc(r.title)} · ${esc(r.employer)}</span>
            <span class="card-meta" style="display:block">${esc(r.location)} · ${esc(r.region)} ·
              ${r.years.toFixed(1)} yrs · ${esc(r.seniority)} · ${esc(r.degreeTier)}</span>
            <span class="card-pills">${pills(r)}</span>
          </span>
          <span class="card-score">
            <span class="score-value">${scored ? r.score.toFixed(0) : r.quality.toFixed(2)}</span>
            <span class="score-label">${scored ? "match score" : "data quality"}</span>
          </span>
        </button>`
      )
      .join("")}</div>`;
    host.querySelectorAll(".card").forEach((card) => {
      card.onclick = () => {
        state.selected = card.dataset.id;
        render();
      };
    });
  }

  const chosen = rows.find((r) => r.id === state.selected) || rows[0];
  state.selected = chosen.id;
  renderDetail(chosen, scored);
}

function renderDetail(row, scored) {
  const rec = row.record;
  const host = $("#detail");
  const tabs = [
    ["career", "Career"],
    ["education", "Education & skills"],
    ["quality", "Data quality"],
    ["source", "Source resume"],
  ];

  let body = "";
  if (state.detailTab === "career") {
    body = `<div class="table-wrap" style="max-height:none">
      <table><thead><tr><th>Employer</th><th>Title</th><th>Period</th><th>Firm type</th>
      <th>Side</th><th>Style</th><th>Sectors</th><th class="num">Coverage</th><th>Intern</th></tr></thead>
      <tbody>${rec.roles
        .map((r) => {
          const period = r.start_date
            ? `${r.start_date} → ${r.end_date || (r.is_current ? "present" : "?")}`
            : r.duration_stated || "dates not stated";
          return `<tr><td><strong>${esc(r.employer)}</strong></td><td class="wrap">${esc(r.title || "—")}</td>
          <td style="white-space:nowrap">${esc(period)}</td><td>${esc(r.firm_type)}</td>
          <td>${esc(r.market_side)}</td><td>${esc(r.strategy_type)}</td>
          <td class="wrap">${esc((r.sectors || []).join(", ") || "—")}</td>
          <td class="num">${r.coverage_universe_size ?? "—"}</td>
          <td>${r.is_internship ? "yes" : ""}</td></tr>`;
        })
        .join("")}</tbody></table></div>`;
    const highlights = rec.roles.filter((r) => (r.highlights || []).length);
    if (highlights.length) {
      body += `<h4 class="block-title">Role highlights</h4>`;
      body += highlights
        .map(
          (r) =>
            `<h4 class="block-title" style="text-transform:none;letter-spacing:0;color:var(--fg-2);margin-bottom:var(--space-2)">${esc(
              r.employer
            )} — ${esc(r.title || "role")}</h4><ul class="clean">${r.highlights
              .map((h) => `<li>${esc(h)}</li>`)
              .join("")}</ul>`
        )
        .join("");
    }
  } else if (state.detailTab === "education") {
    body = `<div class="table-wrap" style="max-height:none"><table>
      <thead><tr><th>Institution</th><th>Degree</th><th>Field</th><th class="num">Completed</th>
      <th>Grade</th><th>Honours</th></tr></thead><tbody>${(rec.education || [])
        .map(
          (e) => `<tr><td><strong>${esc(e.institution)}</strong></td><td>${esc(e.degree || "—")}</td>
        <td class="wrap">${esc(e.field_of_study || "—")}</td><td class="num">${e.end_year ?? "—"}</td>
        <td>${esc(e.gpa || "—")}</td><td class="wrap">${esc(e.honors || "—")}</td></tr>`
        )
        .join("") || `<tr><td colspan="6">No education section could be extracted.</td></tr>`}
      </tbody></table></div>`;
    const listOr = (arr, fallback) => esc((arr || []).join(", ") || fallback);
    body += `<h4 class="block-title">Credentials and skills</h4><dl class="kv">
      <dt>Certifications</dt><dd>${listOr(rec.certifications, "None stated")}</dd>
      <dt>CFA status</dt><dd>${esc(rec.cfa_status || "Not stated")}</dd>
      <dt>Programming</dt><dd>${listOr(rec.programming_languages, "None stated")}</dd>
      <dt>Tools &amp; platforms</dt><dd>${listOr(rec.tools_and_platforms, "None stated")}</dd>
      <dt>Languages</dt><dd>${listOr(rec.languages_spoken, "None stated")}</dd>
      <dt>Contact</dt><dd>${esc(rec.email || "no email")} · ${esc(rec.phone || "no phone")}</dd></dl>`;
    if ((rec.sector_specialisation_detail || []).length) {
      body += `<h4 class="block-title">Sector detail (verbatim)</h4><ul class="clean">${rec.sector_specialisation_detail
        .map((d) => `<li>${esc(d)}</li>`)
        .join("")}</ul>`;
    }
    if ((rec.publications || []).length) {
      body += `<h4 class="block-title">Publications</h4><ul class="clean">${rec.publications
        .map((p) => `<li>${esc(p)}</li>`)
        .join("")}</ul>`;
    }
  } else if (state.detailTab === "quality") {
    const rules = rec.data_quality_flags.filter((f) => !f.startsWith("[model]"));
    const notes = rec.data_quality_flags
      .filter((f) => f.startsWith("[model]"))
      .map((f) => f.replace("[model] ", ""));
    body = `<dl class="kv"><dt>Extraction confidence</dt><dd>${esc(rec.extraction_confidence)}</dd>
      <dt>Quality score</dt><dd>${rec.data_quality_score.toFixed(2)}</dd>
      <dt>Source document</dt><dd>${esc(rec.source_file)}${
      rec.source_agency ? ` · agency-formatted: <strong>${esc(rec.source_agency)}</strong>` : ""
    }</dd></dl>`;
    if (rec.career_gaps.length) {
      body += `<h4 class="block-title">Career continuity</h4>${rec.career_gaps
        .map((g) => `<div class="finding rule"><span class="finding-origin">gap</span><span>${esc(g)}</span></div>`)
        .join("")}`;
    }
    if (rules.length) {
      body += `<h4 class="block-title">Automated validation findings</h4>${rules
        .map((f) => `<div class="finding rule"><span class="finding-origin">rule</span><span>${esc(f)}</span></div>`)
        .join("")}`;
    }
    if (notes.length) {
      body += `<h4 class="block-title">Ambiguities reported by the extraction model</h4>${notes
        .map((f) => `<div class="finding model"><span class="finding-origin">model</span><span>${esc(f)}</span></div>`)
        .join("")}`;
    }
    if (!rules.length && !notes.length && !rec.career_gaps.length) {
      body += `<div class="finding ok"><span class="finding-origin">clean</span><span>No data-quality issues detected.</span></div>`;
    }
  } else {
    body = `<p class="chart-note" style="margin-top:0">Extracted text from <code>${esc(
      rec.source_file
    )}</code> — the exact input the model received, after cleaning.</p>
      <pre class="source">${esc(RAW[row.id] || "Source text not available.")}</pre>`;
  }

  host.innerHTML = `<div class="detail">
    <div class="detail-head">
      <h3 class="detail-name">${esc(row.name)}</h3>
      <p class="detail-role">${esc(row.title)} · ${esc(row.employer)} · ${esc(row.location)}</p>
    </div>
    <div class="detail-metrics">
      <div><div class="kpi-label">Experience (computed)</div><div class="kpi-value">${row.years.toFixed(1)} yrs</div></div>
      <div><div class="kpi-label">Employers</div><div class="kpi-value">${row.employers.length}</div></div>
      <div><div class="kpi-label">Max coverage</div><div class="kpi-value">${row.coverage ?? "n/a"}</div></div>
      <div><div class="kpi-label">Data quality</div><div class="kpi-value">${row.quality.toFixed(2)}</div></div>
      ${scored ? `<div><div class="kpi-label">Match score</div><div class="kpi-value">${row.score.toFixed(1)}</div></div>` : ""}
    </div>
    ${
      scored
        ? `<div style="padding:var(--space-5) var(--space-6);border-bottom:1px solid var(--line)">
             <h4 class="block-title">Why this match score</h4>
             <div class="score-chart" id="scoreChart"></div></div>`
        : ""
    }
    <div class="detail-body">
      <div class="subtabs" role="tablist">${tabs
        .map(
          ([k, label]) =>
            `<button class="subtab" role="tab" data-k="${k}" aria-selected="${
              state.detailTab === k
            }">${label}</button>`
        )
        .join("")}</div>
      <div>${body}</div>
    </div></div>`;

  host.querySelectorAll(".subtab").forEach((b) => {
    b.onclick = () => {
      state.detailTab = b.dataset.k;
      render();
    };
  });

  if (scored) {
    const order = ["sector", "region", "strategy", "experience", "credentials"];
    $("#scoreChart").appendChild(
      barChart(
        order.map((k) => ({
          label: k[0].toUpperCase() + k.slice(1),
          value: row.contributions[k],
          display: row.contributions[k].toFixed(1),
        })),
        { width: 520, labelW: 110, barH: 18, gap: 7 }
      )
    );
  }
}

/* ---------------------------------------------------------------- compare */
function renderCompare(rows) {
  const picker = $("#comparePicker");
  if (!state.compare.length) state.compare = rows.slice(0, 3).map((r) => r.id);

  chipGroup(picker, ROWS.map((r) => r.name), ROWS.filter((r) => state.compare.includes(r.id)).map((r) => r.name), (name) => {
    const row = ROWS.find((r) => r.name === name);
    if (state.compare.includes(row.id)) state.compare = state.compare.filter((id) => id !== row.id);
    else if (state.compare.length < 4) state.compare.push(row.id);
    render();
  });

  const host = $("#compareBody");
  const picked = state.compare
    .map((id) => rows.find((r) => r.id === id) || ROWS.find((r) => r.id === id))
    .filter(Boolean)
    .map((r) => (r.score === undefined ? { ...r, ...scoreCandidate(r) } : r));

  if (picked.length < 2) {
    host.innerHTML = `<div class="empty"><strong>Select at least two candidates</strong>
      <p>Pick two to four candidates above to compare them on the dimensions a hiring manager
      asks about.</p></div>`;
    return;
  }

  const inShortlist = new Set(rows.map((r) => r.id));
  const fields = [
    ["Match score", (r) => (scoringOn() ? r.score.toFixed(1) : "not scored")],
    ["In current shortlist", (r) => (inShortlist.has(r.id) ? "yes" : "no — excluded by filters")],
    ["Region", (r) => r.region],
    ["Location", (r) => r.location],
    ["Current firm", (r) => r.employer],
    ["Title", (r) => r.title],
    ["Approach", (r) => r.strategy],
    ["Market side", (r) => r.side],
    ["Firm type", (r) => r.firmType],
    ["Seniority", (r) => r.seniority],
    ["Years experience", (r) => r.years.toFixed(1)],
    ["Sectors", (r) => r.sectors.join(", ")],
    ["Max coverage universe", (r) => r.coverage ?? "n/a"],
    ["Highest degree", (r) => r.degree || r.degreeTier],
    ["CFA", (r) => (r.cfa ? "Yes" : "No")],
    ["Languages", (r) => r.languages.join(", ") || "Not stated"],
    ["Employers", (r) => r.employers.length],
    ["Data quality", (r) => r.quality.toFixed(2)],
    ["Findings", (r) => r.flags.length],
  ];

  host.innerHTML = `<div class="table-wrap" style="max-height:none"><table>
    <thead><tr><th></th>${picked.map((r) => `<th>${esc(r.name)}</th>`).join("")}</tr></thead>
    <tbody>${fields
      .map(
        ([label, fn]) =>
          `<tr><td style="color:var(--fg-3);white-space:nowrap">${label}</td>${picked
            .map((r) => `<td class="wrap">${esc(fn(r))}</td>`)
            .join("")}</tr>`
      )
      .join("")}</tbody></table></div>`;

  const axes = ["Experience", "Sector breadth", "Coverage scale", "Credentials", "Data quality"];
  const series = picked.map((r, i) => ({
    label: r.name,
    color: SERIES[i % SERIES.length],
    values: [
      Math.min(r.years / 15, 1) * 100,
      Math.min(r.sectors.length / 5, 1) * 100,
      Math.min((r.coverage || 0) / 75, 1) * 100,
      Math.min(
        (["Master's / MBA", "Doctorate / Medical"].includes(r.degreeTier) ? 35 : 15) +
          (r.cfa ? 35 : 0) +
          (r.md ? 30 : 0),
        100
      ),
      r.quality * 100,
    ],
  }));

  const grid = document.createElement("div");
  grid.className = "chart-grid";
  grid.style.marginTop = "var(--space-6)";
  grid.appendChild(
    chartCard(
      "Normalised profile comparison",
      "Axes normalised for comparability: experience against a 15-year scale, sector breadth against 5 sectors, coverage against a 75-name universe.",
      radar(series, axes, { size: 360 }),
      series.map((s) => ({ label: s.label, color: s.color }))
    )
  );
  host.appendChild(grid);
}

/* --------------------------------------------------------------- insights */
function renderInsights(rows) {
  const data = state.scope === "filtered" && rows.length ? rows : ROWS;
  const host = $("#charts");
  host.innerHTML = "";

  const regionsPresent = REGIONS.filter((r) => data.some((d) => d.region === r));
  const sectorsPresent = SECTORS.filter((s) => data.some((d) => d.sectors.includes(s)));
  const matrix = sectorsPresent.map((s) =>
    regionsPresent.map((r) => data.filter((d) => d.region === r && d.sectors.includes(s)).length)
  );
  host.appendChild(
    chartCard(
      "Sector coverage by region",
      "Where the pool is deep and where it is empty. Empty cells are sourcing gaps, not noise.",
      heatmap(matrix, sectorsPresent, regionsPresent, { width: 470 })
    )
  );

  const stratRows = STRATEGIES.filter((s) => data.some((d) => d.strategy === s)).map((s, i) => ({
    label: s,
    value: data.filter((d) => d.strategy === s).length,
    color: SERIES[i % SERIES.length],
  }));
  host.appendChild(
    chartCard(
      "Fundamental vs systematic mix",
      "The pool skews heavily fundamental, which matters if open mandates are quantitative.",
      donutChart(stratRows, { size: 250 }),
      stratRows.map((r) => ({ label: `${r.label} (${r.value})`, color: r.color }))
    )
  );

  host.appendChild(
    chartCard(
      "Experience distribution",
      "Computed from extracted role dates, with overlapping roles counted once.",
      histogram(data.map((d) => d.years), { bins: 8 })
    )
  );

  const firmCounts = {};
  data.forEach((d) => (firmCounts[d.firmType] = (firmCounts[d.firmType] || 0) + 1));
  host.appendChild(
    chartCard(
      "Current firm type",
      "Buy-side, sell-side, banking and consulting backgrounds present in the pool.",
      barChart(
        Object.entries(firmCounts)
          .sort((a, b) => b[1] - a[1])
          .map(([label, value]) => ({ label: label.replace("Investment Bank - ", "IB - "), value })),
        { width: 470, labelW: 185 }
      )
    )
  );

  const pts = data
    .filter((d) => d.coverage)
    .map((d, i) => ({ x: d.years, y: d.coverage, label: d.name, color: SERIES[i % SERIES.length] }));
  host.appendChild(
    chartCard(
      "Coverage universe vs experience",
      "Names under coverage where the resume states it. Hover a point for the candidate.",
      scatter(pts, { xLabel: "years of experience", yLabel: "names covered" })
    )
  );

  const toolCounts = {};
  data.forEach((d) => d.tools.forEach((t) => (toolCounts[t] = (toolCounts[t] || 0) + 1)));
  host.appendChild(
    chartCard(
      "Most common tools and platforms",
      "Extracted verbatim, then counted. Useful for screening on platform familiarity.",
      barChart(
        Object.entries(toolCounts)
          .sort((a, b) => b[1] - a[1])
          .slice(0, 10)
          .map(([label, value]) => ({ label, value })),
        { width: 470, labelW: 160 }
      )
    )
  );

  const gaps = $("#gaps");
  if (!state.sectors.length) {
    gaps.innerHTML = `<div class="empty"><strong>No sectors selected</strong>
      <p>Choose one or more sectors in the requisition to see where the pipeline is thin.</p></div>`;
  } else {
    gaps.innerHTML = `<div class="table-wrap" style="max-height:none"><table>
      <thead><tr><th>Sector</th><th class="num">In pool</th><th class="num">In target region</th>
      <th class="num">Matching full requisition</th></tr></thead><tbody>${state.sectors
        .map((s) => {
          const inPool = ROWS.filter((r) => r.sectors.includes(s)).length;
          const inRegion = ROWS.filter(
            (r) => (state.region === "Any" || r.region === state.region) && r.sectors.includes(s)
          ).length;
          const inReq = rows.filter((r) => r.sectors.includes(s)).length;
          const cls = inReq === 0 ? ' style="color:var(--danger);font-weight:600"' : "";
          return `<tr><td><strong>${esc(s)}</strong></td><td class="num">${inPool}</td>
            <td class="num">${inRegion}</td><td class="num"${cls}>${inReq}</td></tr>`;
        })
        .join("")}</tbody></table></div>`;
  }
}

/* ---------------------------------------------------------------- quality */
function qualityFindings() {
  const out = [];
  CANDIDATES.forEach((c) => {
    c.data_quality_flags.forEach((f) =>
      out.push({
        candidate: c.full_name,
        source: c.source_file,
        origin: f.startsWith("[model]") ? "model" : "rule",
        finding: f.replace("[model] ", ""),
      })
    );
    (c.career_gaps || []).forEach((g) =>
      out.push({
        candidate: c.full_name,
        source: c.source_file,
        origin: "rule",
        finding: `Career continuity: ${g}`,
      })
    );
  });
  return out;
}

function renderQuality() {
  const findings = qualityFindings();
  const shown = state.origin === "all" ? findings : findings.filter((f) => f.origin === state.origin);

  const charts = $("#qualityCharts");
  charts.innerHTML = "";
  charts.appendChild(
    chartCard(
      "Data-quality score by candidate",
      "1.00 is a clean record. Each validation finding reduces the score, so a recruiter can filter by confidence.",
      barChart(
        [...ROWS]
          .sort((a, b) => a.quality - b.quality)
          .map((r) => ({
            label: r.name,
            value: r.quality,
            display: r.quality.toFixed(2),
            color: r.quality >= 0.8 ? SERIES[2] : r.quality >= 0.55 ? SERIES[3] : "oklch(55% 0.16 22)",
          })),
        { width: 470, labelW: 165 }
      ),
      [
        { label: "clean (≥ 0.80)", color: SERIES[2] },
        { label: "review (0.55–0.79)", color: SERIES[3] },
        { label: "verify before use (< 0.55)", color: "oklch(55% 0.16 22)" },
      ]
    )
  );
  const byOrigin = [
    { label: "Validation rules", value: findings.filter((f) => f.origin === "rule").length, color: SERIES[0] },
    { label: "Model-reported", value: findings.filter((f) => f.origin === "model").length, color: SERIES[1] },
  ];
  charts.appendChild(
    chartCard(
      "Findings by origin",
      "Rules compare the extraction against itself and the source dates; model-reported notes are ambiguities the extractor flagged rather than resolved.",
      donutChart(byOrigin, { size: 250, centerLabel: "findings" }),
      byOrigin.map((r) => ({ label: `${r.label} (${r.value})`, color: r.color }))
    )
  );

  $("#qualityTable").innerHTML = `<div class="table-wrap"><table>
    <thead><tr><th>Candidate</th><th>Source</th><th>Origin</th><th>Finding</th></tr></thead>
    <tbody>${shown
      .map(
        (f) => `<tr><td style="white-space:nowrap"><strong>${esc(f.candidate)}</strong></td>
        <td style="color:var(--fg-3)">${esc(f.source)}</td>
        <td><span class="pill ${f.origin === "rule" ? "warn" : ""}">${
          f.origin === "rule" ? "validation rule" : "extraction model"
        }</span></td>
        <td class="wrap">${esc(f.finding)}</td></tr>`
      )
      .join("")}</tbody></table></div>`;
}

/* ------------------------------------------------------------------ render */
function render() {
  renderControls();
  const rows = filteredRows();
  const scored = scoringOn();

  $("#kpiPool").textContent = ROWS.length;
  $("#kpiMatch").textContent = rows.length;
  $("#kpiYears").textContent = rows.length
    ? (rows.reduce((s, r) => s + r.years, 0) / rows.length).toFixed(1) + " yrs"
    : "—";
  $("#kpiTop").textContent = rows.length && scored ? Math.max(...rows.map((r) => r.score)).toFixed(0) : "—";
  $("#kpiFlags").textContent = ROWS.filter((r) => r.flags.length).length;

  $("#mandateTag").innerHTML =
    state.preset === Object.keys(PRESETS)[0]
      ? ""
      : `<span class="mandate-tag">Scoring against mandate: ${esc(state.preset)}</span>`;

  document.querySelectorAll(".tab").forEach((t) => {
    const on = t.dataset.panel === state.tab;
    t.setAttribute("aria-selected", on ? "true" : "false");
    $("#panel-" + t.dataset.panel).hidden = !on;
  });

  if (state.tab === "results") renderResults(rows);
  else if (state.tab === "compare") renderCompare(rows);
  else if (state.tab === "insights") renderInsights(rows);
  else renderQuality();
}

/* -------------------------------------------------------------------- init */
document.querySelectorAll(".tab").forEach((t) => {
  t.onclick = () => {
    state.tab = t.dataset.panel;
    render();
    document.querySelector(".main").scrollTop = 0;
  };
});

$("#exportShortlist").onclick = () => {
  const rows = filteredRows();
  const header = [
    "candidate", "match_score", "region", "location", "current_employer", "current_title",
    "strategy", "market_side", "seniority", "years_experience", "max_coverage", "highest_degree",
    "has_cfa", "data_quality_score", "n_findings", "source_file",
  ];
  downloadCSV("shortlist.csv", [
    header,
    ...rows.map((r) => [
      r.name, scoringOn() ? r.score : "", r.region, r.location, r.employer, r.title, r.strategy,
      r.side, r.seniority, r.years, r.coverage ?? "", r.degree ?? r.degreeTier, r.cfa,
      r.quality, r.flags.length, r.sourceFile,
    ]),
  ]);
};

$("#exportQuality").onclick = () => {
  const findings = qualityFindings();
  downloadCSV("data_quality_report.csv", [
    ["candidate", "source_file", "origin", "finding"],
    ...findings.map((f) => [f.candidate, f.source, f.origin, f.finding]),
  ]);
};

// A requisition is deep-linkable (?req=<n>) so a recruiter can send a colleague the
// exact mandate view rather than a description of which filters to set.
const reqIndex = Number(new URLSearchParams(location.search).get("req"));
const keys = Object.keys(PRESETS);
applyPreset(keys[Number.isInteger(reqIndex) && reqIndex >= 0 && reqIndex < keys.length ? reqIndex : 0]);
