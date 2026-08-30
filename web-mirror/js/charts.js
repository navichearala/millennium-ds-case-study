/* ------------------------------------------------------------------
   Hand-rolled SVG charts.

   No charting library and no CDN: the whole bundle is self-contained, so
   the preview renders identically offline and nothing can be blocked at
   load time. Every chart is plain SVG with tokens read from CSS, so the
   palette stays in one place.
   ------------------------------------------------------------------ */
const SVG_NS = "http://www.w3.org/2000/svg";

export const SERIES = [
  "oklch(48% 0.12 235)",
  "oklch(58% 0.115 232)",
  "oklch(52% 0.11 158)",
  "oklch(62% 0.13 62)",
  "oklch(46% 0.1 300)",
  "oklch(60% 0.09 200)",
];

function el(name, attrs = {}, text) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== undefined && v !== null) node.setAttribute(k, String(v));
  }
  if (text !== undefined) node.textContent = String(text);
  return node;
}

function svgRoot(w, h) {
  // Explicit width/height plus `max-width:100%` in CSS means the chart renders at its
  // natural size and only shrinks on narrow viewports. Without this, an SVG with only a
  // viewBox stretches to fill its container and scales the label text with it, which is
  // why chart typography balloons in wide cards.
  return el("svg", {
    viewBox: `0 0 ${w} ${h}`,
    width: w,
    height: h,
    role: "img",
    preserveAspectRatio: "xMidYMid meet",
  });
}

const LABEL = { "font-size": 11, fill: "oklch(46% 0.02 250)", "font-family": "inherit" };
const VALUE = {
  "font-size": 11,
  fill: "oklch(26% 0.03 250)",
  "font-family": "inherit",
  "font-weight": 600,
};

/* -------------------------------------------------------- bar (horizontal) */
export function barChart(rows, { width = 460, barH = 22, gap = 8, labelW = 150, color } = {}) {
  const max = Math.max(1, ...rows.map((r) => r.value));
  const h = rows.length * (barH + gap) + 8;
  const svg = svgRoot(width, h);
  const plotW = width - labelW - 42;

  rows.forEach((row, i) => {
    const y = i * (barH + gap) + 4;
    svg.appendChild(
      el("text", { x: labelW - 8, y: y + barH / 2 + 4, "text-anchor": "end", ...LABEL }, row.label)
    );
    svg.appendChild(
      el("rect", {
        x: labelW,
        y,
        width: plotW,
        height: barH,
        rx: 3,
        fill: "oklch(96% 0.005 250)",
      })
    );
    const w = Math.max(2, (row.value / max) * plotW);
    const bar = el("rect", {
      x: labelW,
      y,
      width: w,
      height: barH,
      rx: 3,
      fill: color || row.color || SERIES[0],
    });
    svg.appendChild(bar);
    svg.appendChild(
      el(
        "text",
        { x: labelW + w + 6, y: y + barH / 2 + 4, ...VALUE },
        row.display ?? row.value
      )
    );
  });
  return svg;
}

/* ------------------------------------------------------------------ donut */
export function donutChart(rows, { size = 260, thickness = 34, centerLabel } = {}) {
  const total = rows.reduce((s, r) => s + r.value, 0) || 1;
  const svg = svgRoot(size, size);
  const cx = size / 2;
  const cy = size / 2;
  const r = (size - thickness) / 2 - 6;
  let angle = -Math.PI / 2;

  rows.forEach((row, i) => {
    const sweep = (row.value / total) * Math.PI * 2;
    const end = angle + sweep;
    const large = sweep > Math.PI ? 1 : 0;
    const x1 = cx + r * Math.cos(angle);
    const y1 = cy + r * Math.sin(angle);
    const x2 = cx + r * Math.cos(end);
    const y2 = cy + r * Math.sin(end);
    // A full-circle single segment cannot be drawn as an arc; use a ring.
    if (rows.length === 1 || row.value === total) {
      svg.appendChild(
        el("circle", {
          cx,
          cy,
          r,
          fill: "none",
          stroke: row.color || SERIES[i % SERIES.length],
          "stroke-width": thickness,
        })
      );
    } else {
      svg.appendChild(
        el("path", {
          d: `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`,
          fill: "none",
          stroke: row.color || SERIES[i % SERIES.length],
          "stroke-width": thickness,
          "stroke-linecap": "butt",
        })
      );
    }
    const mid = angle + sweep / 2;
    const pct = Math.round((row.value / total) * 100);
    if (pct >= 8) {
      svg.appendChild(
        el(
          "text",
          {
            x: cx + r * Math.cos(mid),
            y: cy + r * Math.sin(mid) + 4,
            "text-anchor": "middle",
            "font-size": 11,
            "font-weight": 600,
            fill: "oklch(99% 0 0)",
            "font-family": "inherit",
          },
          `${pct}%`
        )
      );
    }
    angle = end;
  });

  svg.appendChild(
    el(
      "text",
      {
        x: cx,
        y: cy - 2,
        "text-anchor": "middle",
        "font-size": 22,
        "font-weight": 650,
        fill: "oklch(24% 0.045 250)",
        "font-family": "inherit",
      },
      total
    )
  );
  const noun = centerLabel || (total === 1 ? "candidate" : "candidates");
  svg.appendChild(el("text", { x: cx, y: cy + 16, "text-anchor": "middle", ...LABEL }, noun));
  return svg;
}

/* ---------------------------------------------------------------- heatmap */
export function heatmap(matrix, rowLabels, colLabels, { width = 470, cell = 30, labelW = 150 } = {}) {
  const colW = Math.max(52, (width - labelW) / Math.max(colLabels.length, 1));
  const h = rowLabels.length * cell + 34;
  const svg = svgRoot(labelW + colW * colLabels.length, h);
  const max = Math.max(1, ...matrix.flat());

  colLabels.forEach((c, j) => {
    svg.appendChild(
      el(
        "text",
        { x: labelW + colW * j + colW / 2, y: 12, "text-anchor": "middle", ...LABEL },
        c
      )
    );
  });

  rowLabels.forEach((rLabel, i) => {
    const y = 22 + i * cell;
    svg.appendChild(
      el("text", { x: labelW - 8, y: y + cell / 2 + 4, "text-anchor": "end", ...LABEL }, rLabel)
    );
    colLabels.forEach((_, j) => {
      const v = matrix[i][j];
      const t = v / max;
      svg.appendChild(
        el("rect", {
          x: labelW + colW * j + 1,
          y: y + 1,
          width: colW - 2,
          height: cell - 2,
          rx: 3,
          fill: v === 0 ? "oklch(97% 0.004 250)" : `oklch(${94 - t * 46}% ${0.02 + t * 0.1} 235)`,
        })
      );
      svg.appendChild(
        el(
          "text",
          {
            x: labelW + colW * j + colW / 2,
            y: y + cell / 2 + 4,
            "text-anchor": "middle",
            "font-size": 11,
            "font-weight": v === 0 ? 400 : 600,
            fill: v === 0 ? "oklch(72% 0.01 250)" : t > 0.55 ? "oklch(99% 0 0)" : "oklch(30% 0.05 250)",
            "font-family": "inherit",
          },
          v
        )
      );
    });
  });
  return svg;
}

/* -------------------------------------------------------------- histogram */
export function histogram(values, { width = 460, height = 210, bins = 8, color } = {}) {
  const svg = svgRoot(width, height);
  if (!values.length) return svg;
  const min = Math.floor(Math.min(...values));
  const max = Math.ceil(Math.max(...values)) || 1;
  const span = Math.max(max - min, 1);
  const step = span / bins;
  const counts = new Array(bins).fill(0);
  values.forEach((v) => {
    const idx = Math.min(bins - 1, Math.floor((v - min) / step));
    counts[idx] += 1;
  });

  const padL = 30;
  const padB = 30;
  const plotW = width - padL - 10;
  const plotH = height - padB - 12;
  const maxCount = Math.max(...counts, 1);

  for (let g = 0; g <= maxCount; g += Math.max(1, Math.ceil(maxCount / 4))) {
    const y = 12 + plotH - (g / maxCount) * plotH;
    svg.appendChild(
      el("line", { x1: padL, y1: y, x2: padL + plotW, y2: y, stroke: "oklch(93% 0.006 250)" })
    );
    svg.appendChild(el("text", { x: padL - 6, y: y + 4, "text-anchor": "end", ...LABEL }, g));
  }

  const bw = plotW / bins;
  counts.forEach((c, i) => {
    const bh = (c / maxCount) * plotH;
    svg.appendChild(
      el("rect", {
        x: padL + i * bw + 2,
        y: 12 + plotH - bh,
        width: bw - 4,
        height: Math.max(bh, c ? 2 : 0),
        rx: 3,
        fill: color || SERIES[1],
      })
    );
  });

  [0, bins / 2, bins].forEach((i) => {
    const v = min + step * i;
    svg.appendChild(
      el(
        "text",
        { x: padL + bw * i, y: height - 8, "text-anchor": i === 0 ? "start" : i === bins ? "end" : "middle", ...LABEL },
        `${v.toFixed(0)}y`
      )
    );
  });
  return svg;
}

/* ---------------------------------------------------------------- scatter */
export function scatter(points, { width = 460, height = 230, xLabel = "", yLabel = "" } = {}) {
  const svg = svgRoot(width, height);
  if (!points.length) return svg;
  const padL = 40;
  const padB = 32;
  const plotW = width - padL - 14;
  const plotH = height - padB - 14;
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const xMax = Math.max(...xs) * 1.1 || 1;
  const yMax = Math.max(...ys) * 1.12 || 1;

  for (let i = 0; i <= 3; i++) {
    const y = 14 + plotH - (i / 3) * plotH;
    svg.appendChild(
      el("line", { x1: padL, y1: y, x2: padL + plotW, y2: y, stroke: "oklch(93% 0.006 250)" })
    );
    svg.appendChild(
      el("text", { x: padL - 6, y: y + 4, "text-anchor": "end", ...LABEL }, Math.round((i / 3) * yMax))
    );
  }

  points.forEach((p) => {
    const cx = padL + (p.x / xMax) * plotW;
    const cy = 14 + plotH - (p.y / yMax) * plotH;
    const g = el("g");
    g.appendChild(
      el("circle", { cx, cy, r: 7, fill: p.color || SERIES[0], "fill-opacity": 0.82 })
    );
    g.appendChild(el("title", {}, `${p.label}: ${p.x} yrs, ${p.y} names`));
    svg.appendChild(g);
  });

  svg.appendChild(
    el("text", { x: padL + plotW / 2, y: height - 6, "text-anchor": "middle", ...LABEL }, xLabel)
  );
  svg.appendChild(
    el(
      "text",
      { x: 12, y: 14 + plotH / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${14 + plotH / 2})`, ...LABEL },
      yLabel
    )
  );
  return svg;
}

/* ------------------------------------------------------------------ radar */
export function radar(series, axes, { size = 340 } = {}) {
  const svg = svgRoot(size, size);
  const cx = size / 2;
  const cy = size / 2;
  const r = size / 2 - 54;
  const n = axes.length;
  const point = (i, value) => {
    const a = -Math.PI / 2 + (i / n) * Math.PI * 2;
    const rad = (value / 100) * r;
    return [cx + rad * Math.cos(a), cy + rad * Math.sin(a)];
  };

  [25, 50, 75, 100].forEach((ring) => {
    const pts = axes.map((_, i) => point(i, ring).join(",")).join(" ");
    svg.appendChild(
      el("polygon", {
        points: pts,
        fill: "none",
        stroke: ring === 100 ? "oklch(84% 0.01 250)" : "oklch(93% 0.006 250)",
      })
    );
  });

  axes.forEach((axis, i) => {
    const [x, y] = point(i, 100);
    svg.appendChild(el("line", { x1: cx, y1: cy, x2: x, y2: y, stroke: "oklch(93% 0.006 250)" }));
    const [lx, ly] = point(i, 124);
    const anchor = Math.abs(lx - cx) < 12 ? "middle" : lx > cx ? "start" : "end";
    svg.appendChild(el("text", { x: lx, y: ly + 3, "text-anchor": anchor, ...LABEL }, axis));
  });

  series.forEach((s, si) => {
    const color = s.color || SERIES[si % SERIES.length];
    const pts = s.values.map((v, i) => point(i, v).join(",")).join(" ");
    svg.appendChild(
      el("polygon", {
        points: pts,
        fill: color,
        "fill-opacity": 0.14,
        stroke: color,
        "stroke-width": 2,
      })
    );
    s.values.forEach((v, i) => {
      const [x, y] = point(i, v);
      svg.appendChild(el("circle", { cx: x, cy: y, r: 3, fill: color }));
    });
  });
  return svg;
}
