const caches = {
  quartic: new Map(),
  freeEnergy: new Map(),
  densityPotential: new Map(),
  cooperInstability: new Map(),
  channelCompare: new Map(),
  densityRpa: new Map(),
  rpaFreeEnergy: new Map(),
  exchange: new Map(),
  bcs: new Map(),
};

const FIXED_DOMAINS = {
  quartic: {
    x: [-3, 6],
    y: [-10, 20],
  },
  freeEnergy: {
    x: [0, 10],
    y: [-1.5, 0.2],
  },
  densityPotential: {
    x: [-2, 10],
    y: [-8, 4],
  },
  cooperInstability: {
    x: [0.1, 5],
    y: [-12, 12],
  },
  channelDirect: {
    x: [-2, 2],
    y: [-8, 4],
  },
  channelCooper: {
    x: [-2, 2],
    y: [-8, 4],
  },
  densityRpa: {
    x: [-1.5, 1.5],
    y: [-6, 18],
  },
  rpaFreeEnergy: {
    x: [0.1, 10],
    y: [-3.5, 1.6],
  },
  exchange: {
    x: [0, 2],
    y: [-8, 20],
  },
  bcs: {
    x: [0, 0.6],
    y: [-4, 8],
  },
};

function memoize(bucket, keyParts, build) {
  const key = JSON.stringify(keyParts);
  const store = caches[bucket];
  if (store.has(key)) {
    return store.get(key);
  }
  const value = build();
  store.set(key, value);
  return value;
}

function linspace(start, stop, count) {
  if (count <= 1) return [start];
  const step = (stop - start) / (count - 1);
  return Array.from({ length: count }, (_, i) => start + i * step);
}

function finitePoints(xs, ys, series, color, dash = "solid") {
  const points = [];
  for (let i = 0; i < xs.length; i += 1) {
    const y = ys[i];
    if (Number.isFinite(y)) {
      points.push({ x: xs[i], y, series, color, dash });
    }
  }
  return points;
}

function constantSeries(xs, value, series, color, dash = "solid") {
  return xs.map((x) => ({ x, y: value, series, color, dash }));
}

function simplePlot(Plot, data, {
  width = 760,
  height = 520,
  xLabel,
  yLabel,
  marks = [],
  xDomain,
  yDomain,
} = {}) {
  const plotMarks = [
    Plot.line(data, {
      x: "x",
      y: "y",
      stroke: "color",
      strokeDash: "dash",
      strokeWidth: 2.5,
    }),
    Plot.frame(),
    ...marks,
  ];

  return Plot.plot({
    width,
    height,
    marginLeft: 72,
    marginRight: 24,
    marginBottom: 56,
    marginTop: 24,
    x: {
      label: xLabel,
      domain: xDomain,
      grid: true,
    },
    y: {
      label: yLabel,
      domain: yDomain,
      grid: true,
    },
    marks: plotMarks,
  });
}

function quarticPotential(x, a, b) {
  return -a * x ** 2 + b * x ** 4;
}

function quarticExact(y, a, b) {
  return y ** 2 + 0.5 * Math.log(Math.max(a - 2 * Math.sqrt(b) * y, 1e-9));
}

function calculateY0(a, b) {
  if (b <= 0 || a ** 2 <= 4 * b) return Number.NaN;
  const y0 = (a - Math.sqrt(a ** 2 - 4 * b)) / (4 * Math.sqrt(b));
  const curvature = 2 * (1 - 4 * y0 ** 2);
  return curvature > 0 ? y0 : Number.NaN;
}

export function quarticFigure({ Plot, a, b }) {
  return memoize("quartic", [a, b], () => {
    const ySingularity = a / (2 * Math.sqrt(b));
    const yMax = Math.max(ySingularity - 0.05, -2.5);
    const ys = linspace(-3, yMax, 420);

    const quartic = finitePoints(ys, ys.map((y) => quarticPotential(y, a, b)), "Quartic Potential", "#2b6cb0");
    const exact = finitePoints(ys, ys.map((y) => quarticExact(y, a, b)), "Full Action", "#1a365d");
    const original = finitePoints(ys, ys.map((y) => y ** 2), "Original Gaussian", "#6b7280", "dotted");

    const data = [...quartic, ...exact, ...original];
    const marks = [
      Plot.ruleX([ySingularity], { stroke: "#111827", strokeDash: [5, 4], strokeOpacity: 0.45 }),
      Plot.dot([{ x: 0, y: 0 }], { x: "x", y: "y", r: 4.5, fill: "#111827" }),
    ];

    let note = null;
    if (a ** 2 >= 4 * b) {
      const y0 = (a - Math.sqrt(a ** 2 - 4 * b)) / (4 * Math.sqrt(b));
      const phi0 = quarticExact(y0, a, b);
      const phiPP0 = 2 * (1 - 4 * y0 ** 2);
      const approx = finitePoints(
        ys,
        ys.map((y) => phi0 + 0.5 * phiPP0 * (y - y0) ** 2),
        "Quadratic Approximation",
        "#c53030",
        "dashed",
      );
      data.push(...approx);
      marks.push(Plot.dot([{ x: y0, y: phi0 }], { x: "x", y: "y", r: 5, fill: "#c53030" }));
    } else {
      note = "No real saddle point exists in this parameter region, so the quadratic approximation is omitted.";
    }

    const legend = [
      { label: "Quartic potential", color: "#2b6cb0", kind: "line" },
      { label: "Full action", color: "#1a365d", kind: "line" },
      { label: "Original Gaussian", color: "#6b7280", dash: "dotted", kind: "line" },
      { label: "Quadratic approximation", color: "#c53030", dash: "dashed", kind: "line" },
      { label: "Original minimum", color: "#111827", kind: "point" },
      { label: "Singularity", color: "#111827", dash: "dashed", kind: "line" },
    ];

    return {
      plot: simplePlot(Plot, data, {
        xLabel: "y",
        yLabel: "Action value Φ(y)",
        xDomain: FIXED_DOMAINS.quartic.x,
        yDomain: FIXED_DOMAINS.quartic.y,
        marks,
      }),
      legend,
      note,
    };
  });
}

export function freeEnergyFigure({ Plot, a, bMax }) {
  return memoize("freeEnergy", [a, bMax], () => {
    const bs = linspace(0.001, bMax, 220);
    const total = [];
    const meanField = [];
    const fluctuation = [];

    for (const b of bs) {
      const y0 = calculateY0(a, b);
      if (!Number.isFinite(y0)) {
        total.push(Number.NaN);
        meanField.push(Number.NaN);
        fluctuation.push(Number.NaN);
        continue;
      }
      const phi0 = y0 ** 2 + 0.5 * Math.log(a - 2 * Math.sqrt(b) * y0);
      const curvature = 2 * (1 - 4 * y0 ** 2);
      const fluct = 0.5 * Math.log(curvature) - 0.5 * Math.log(2 * Math.PI);
      meanField.push(phi0);
      fluctuation.push(fluct);
      total.push(phi0 + fluct);
    }

    const firstValid = total.findIndex(Number.isFinite);
    const note = total.some((value) => !Number.isFinite(value))
      ? "Parameter values where the saddle-point expansion breaks down are skipped."
      : null;

    if (firstValid === -1) {
      return {
        plot: Plot.plot({
          width: 760,
          height: 360,
          marks: [
            Plot.text([{ x: 0.5, y: 0.5, text: "No valid saddle-point region in this range." }], {
              x: "x",
              y: "y",
              text: "text",
              frameAnchor: "middle",
              fontSize: 18,
            }),
          ],
        }),
        legend: [],
        note,
      };
    }

    const totalNorm = total.map((value) => (Number.isFinite(value) ? value - total[firstValid] : Number.NaN));
    const meanNorm = meanField.map((value) => (Number.isFinite(value) ? value - meanField[firstValid] : Number.NaN));
    const fluctNorm = fluctuation.map((value) => (Number.isFinite(value) ? value - fluctuation[firstValid] : Number.NaN));

    const data = [
      ...finitePoints(bs, totalNorm, "Total free energy change", "#111827"),
      ...finitePoints(bs, meanNorm, "Mean-field contribution", "#2563eb", "dashed"),
      ...finitePoints(bs, fluctNorm, "Fluctuation contribution", "#dc2626", "dashdot"),
    ];

    return {
      plot: simplePlot(Plot, data, {
        xLabel: "Interaction parameter b",
        yLabel: "Change in free energy",
        xDomain: FIXED_DOMAINS.freeEnergy.x,
        yDomain: FIXED_DOMAINS.freeEnergy.y,
      }),
      legend: [
        { label: "Total free energy change", color: "#111827", kind: "line" },
        { label: "Mean-field part", color: "#2563eb", dash: "dashed", kind: "line" },
        { label: "Fluctuation part", color: "#dc2626", dash: "dashdot", kind: "line" },
      ],
      note,
    };
  });
}

export function densityPotentialFigure({ Plot, epsilon, vMax }) {
  return memoize("densityPotential", [epsilon, vMax], () => {
    const vs = linspace(-0.95 * epsilon, vMax, 220);
    const exact = finitePoints(
      vs,
      vs.map((v) => (epsilon + v > 0 ? Math.log(epsilon + v) : Number.NaN)),
      "Exact loop term",
      "#2563eb",
    );
    const rpa = finitePoints(
      vs,
      vs.map((v) => Math.log(epsilon) + v / epsilon - (v ** 2) / (2 * epsilon ** 2)),
      "RPA approximation",
      "#dc2626",
      "dashed",
    );

    return {
      plot: simplePlot(Plot, [...exact, ...rpa], {
        width: 720,
        height: 460,
        xLabel: "Field fluctuation V",
        yLabel: "Effective potential",
        xDomain: FIXED_DOMAINS.densityPotential.x,
        yDomain: FIXED_DOMAINS.densityPotential.y,
        marks: [
          Plot.dot([{ x: 0, y: Math.log(epsilon) }], { x: "x", y: "y", r: 4.5, fill: "#111827" }),
        ],
      }),
      legend: [
        { label: "Exact loop term", color: "#2563eb", kind: "line" },
        { label: "RPA approximation", color: "#dc2626", dash: "dashed", kind: "line" },
        { label: "Expansion point", color: "#111827", kind: "point" },
      ],
      note: null,
    };
  });
}

function cooperRCoeff(T, g, epsilon) {
  if (T < 1e-4) return 1 / g - 1 / epsilon;
  const val = epsilon / (2 * T);
  const chi = (1 / epsilon) * (1 / Math.tanh(val));
  return 1 / g - chi;
}

export function cooperInstabilityFigure({ Plot, g, epsilon }) {
  return memoize("cooperInstability", [g, epsilon], () => {
    const Ts = linspace(0.1, 5, 240);
    const data = finitePoints(Ts, Ts.map((T) => cooperRCoeff(T, g, epsilon)), "r(T)", "#166534");
    return {
      plot: simplePlot(Plot, data, {
        width: 720,
        height: 460,
        xLabel: "Temperature T",
        yLabel: "Coefficient 1/g - χ(T)",
        xDomain: FIXED_DOMAINS.cooperInstability.x,
        yDomain: FIXED_DOMAINS.cooperInstability.y,
        marks: [Plot.ruleY([0], { stroke: "#6b7280", strokeDash: [5, 4] })],
      }),
      legend: [
        { label: "r(T)", color: "#166534", kind: "line" },
        { label: "Instability threshold", color: "#6b7280", dash: "dashed", kind: "line" },
      ],
      note: null,
    };
  });
}

function directAction(V, epsilon, g) {
  return epsilon + V > 0 ? -(V ** 2) / (4 * g) + Math.log(epsilon + V) : Number.NaN;
}

function cooperAction(Delta, epsilon, g) {
  return epsilon ** 2 - 4 * Delta ** 2 > 0
    ? (Delta ** 2) / g + 0.5 * Math.log(epsilon ** 2 - 4 * Delta ** 2)
    : Number.NaN;
}

export function channelCompareFigure({ Plot, epsilon, gDirect, gCooper1, gCooper2 }) {
  return memoize("channelCompare", [epsilon, gDirect, gCooper1, gCooper2], () => {
    const directXs = linspace(-0.9 * epsilon, 2.0, 220);
    const cooperXs = linspace(-epsilon / 2, epsilon / 2, 220);

    const leftData = finitePoints(
      directXs,
      directXs.map((x) => directAction(x, epsilon, gDirect)),
      "Direct channel",
      "#2563eb",
    );
    const rightData = [
      ...finitePoints(
        cooperXs,
        cooperXs.map((x) => cooperAction(x, epsilon, gCooper1)),
        "Cooper channel 1",
        "#dc2626",
      ),
      ...finitePoints(
        cooperXs,
        cooperXs.map((x) => cooperAction(x, epsilon, gCooper2)),
        "Cooper channel 2",
        "#d97706",
        "dashed",
      ),
    ];

    return {
      leftPlot: simplePlot(Plot, leftData, {
        width: 380,
        height: 420,
        xLabel: "Potential V",
        yLabel: "Effective action",
        xDomain: FIXED_DOMAINS.channelDirect.x,
        yDomain: FIXED_DOMAINS.channelDirect.y,
      }),
      rightPlot: simplePlot(Plot, rightData, {
        width: 380,
        height: 420,
        xLabel: "Gap parameter Δ",
        yLabel: "Effective action",
        xDomain: FIXED_DOMAINS.channelCooper.x,
        yDomain: FIXED_DOMAINS.channelCooper.y,
      }),
      legend: [
        { label: `Direct channel (g=${gDirect.toFixed(1)})`, color: "#2563eb", kind: "line" },
        { label: `Cooper channel (g=${gCooper1.toFixed(1)})`, color: "#dc2626", kind: "line" },
        { label: `Cooper channel (g=${gCooper2.toFixed(1)})`, color: "#d97706", dash: "dashed", kind: "line" },
      ],
      note: null,
    };
  });
}

function fermi(E, beta) {
  return 1 / (1 + Math.exp(beta * E));
}

export function densityRpaFigure({ Plot, beta, cgAbs, xi }) {
  return memoize("densityRpa", [beta, cgAbs, xi], () => {
    const Cg = -cgAbs;
    const vs = linspace(-1.5, 1.5, 220);
    const exact = finitePoints(
      vs,
      vs.map((V) => -(V ** 2) / (2 * Cg) * beta - Math.log(1 + Math.exp(-beta * (xi + V)))),
      "Full action",
      "#6d28d9",
    );

    const nF = fermi(xi, beta);
    const s0 = -Math.log(1 + Math.exp(-beta * xi));
    const coeff1 = beta * nF;
    const curvature = -beta / Cg - beta ** 2 * nF * (1 - nF);
    const approx = finitePoints(
      vs,
      vs.map((V) => s0 + coeff1 * V + 0.5 * curvature * V ** 2),
      "RPA expansion",
      "#ea580c",
      "dashed",
    );
    const y0 = -(0 ** 2) / (2 * Cg) * beta - Math.log(1 + Math.exp(-beta * xi));

    return {
      plot: simplePlot(Plot, [...exact, ...approx], {
        width: 720,
        height: 460,
        xLabel: "Hartree potential V",
        yLabel: "Effective action",
        xDomain: FIXED_DOMAINS.densityRpa.x,
        yDomain: FIXED_DOMAINS.densityRpa.y,
        marks: [Plot.dot([{ x: 0, y: y0 }], { x: "x", y: "y", r: 4.5, fill: "#111827" })],
      }),
      legend: [
        { label: "Full action", color: "#6d28d9", kind: "line" },
        { label: "Quadratic expansion", color: "#ea580c", dash: "dashed", kind: "line" },
        { label: "Expansion point", color: "#111827", kind: "point" },
      ],
      note: null,
    };
  });
}

export function rpaFreeEnergyFigure({ Plot, beta, xi, gMax }) {
  return memoize("rpaFreeEnergy", [beta, xi, gMax], () => {
    const gs = linspace(0.1, gMax, 220);
    const nF = fermi(xi, beta);
    const rho0 = beta * nF;
    const s0 = -Math.log(1 + Math.exp(-beta * xi));
    const Pi = beta * nF * (1 - nF);

    const total = [];
    const hartree = [];
    const corr = [];

    for (const g of gs) {
      const A = beta / g + Pi;
      const h = (rho0 ** 2) / (2 * A);
      const c = 0.5 * Math.log(A) - 0.5 * Math.log(2 * Math.PI);
      hartree.push(h + s0);
      corr.push(c + s0);
      total.push(s0 + h + c);
    }

    const data = [
      ...finitePoints(gs, total, "Total RPA free energy", "#111827"),
      ...finitePoints(gs, hartree, "Mean field + Hartree", "#2563eb", "dashed"),
      ...finitePoints(gs, corr, "Fluctuation contribution", "#dc2626", "dashdot"),
      ...constantSeries([gs[0], gs[gs.length - 1]], s0, "Non-interacting limit", "#6b7280", "dotted"),
    ];

    return {
      plot: simplePlot(Plot, data, {
        xLabel: "Interaction strength g",
        yLabel: "Free energy F",
        xDomain: FIXED_DOMAINS.rpaFreeEnergy.x,
        yDomain: FIXED_DOMAINS.rpaFreeEnergy.y,
      }),
      legend: [
        { label: "Total RPA free energy", color: "#111827", kind: "line" },
        { label: "Mean field + Hartree", color: "#2563eb", dash: "dashed", kind: "line" },
        { label: "Fluctuation contribution", color: "#dc2626", dash: "dashdot", kind: "line" },
        { label: "Non-interacting limit", color: "#6b7280", dash: "dotted", kind: "line" },
      ],
      note: null,
    };
  });
}

export function exchangeFigure({ Plot, beta, Cg, xi1, xi2 }) {
  return memoize("exchange", [beta, Cg, xi1, xi2], () => {
    const Ms = linspace(0, 2, 240);
    const exact = [];
    const approx = [];

    const n1 = fermi(xi1, beta);
    const n2 = fermi(xi2, beta);
    const chi0 = Math.abs(xi1 - xi2) < 1e-5 ? beta * n1 * (1 - n1) : (n1 - n2) / (xi1 - xi2);
    const coeff = beta * (1 / Cg - chi0);

    for (const M of Ms) {
      const avgXi = (xi1 + xi2) / 2;
      const deltaXi = (xi1 - xi2) / 2;
      const root = Math.sqrt(deltaXi ** 2 + M ** 2);
      const ePlus = avgXi + root;
      const eMinus = avgXi - root;
      const exactVal = beta * (M ** 2) / Cg - Math.log(1 + Math.exp(-beta * ePlus)) - Math.log(1 + Math.exp(-beta * eMinus));
      const s0 = -Math.log(1 + Math.exp(-beta * xi1)) - Math.log(1 + Math.exp(-beta * xi2));
      exact.push(exactVal);
      approx.push(s0 + coeff * M ** 2);
    }

    const data = [
      ...finitePoints(Ms, exact, "Exact action", "#2563eb"),
      ...finitePoints(Ms, approx, "Stoner expansion", "#dc2626", "dashed"),
    ];

    return {
      plot: simplePlot(Plot, data, {
        xLabel: "Magnetization magnitude |M|",
        yLabel: "Effective action",
        xDomain: FIXED_DOMAINS.exchange.x,
        yDomain: FIXED_DOMAINS.exchange.y,
      }),
      legend: [
        { label: "Exact action", color: "#2563eb", kind: "line" },
        { label: "Quadratic Stoner expansion", color: "#dc2626", dash: "dashed", kind: "line" },
      ],
      note: null,
    };
  });
}

export function bcsFigure({ Plot, beta, Cg, xi }) {
  return memoize("bcs", [beta, Cg, xi], () => {
    const deltas = linspace(0, 0.6, 240);
    const exact = [];
    const approx = [];

    for (const delta of deltas) {
      const eqp = Math.sqrt(xi ** 2 + delta ** 2);
      exact.push(beta * (delta ** 2) / Cg - 2 * Math.log(Math.cosh(beta * eqp / 2) / Math.cosh(beta * xi / 2)));
      const chiPair = Math.tanh(beta * xi / 2) / (2 * xi);
      approx.push(beta * (1 / Cg - chiPair) * delta ** 2);
    }

    const minIndex = exact.reduce((best, value, index, arr) => (value < arr[best] ? index : best), 0);
    const minDelta = deltas[minIndex];
    const status = approx[Math.min(10, approx.length - 1)] < 0 ? "Unstable (superconducting)" : "Stable (normal state)";

    return {
      plot: simplePlot(Plot, [
        ...finitePoints(deltas, exact, "Full BCS action", "#166534"),
        ...finitePoints(deltas, approx, "Ginzburg-Landau approximation", "#d97706", "dashed"),
      ], {
        xLabel: "Order parameter |Δ|",
        yLabel: "Effective action",
        xDomain: FIXED_DOMAINS.bcs.x,
        yDomain: FIXED_DOMAINS.bcs.y,
        marks: [
          Plot.ruleY([0], { stroke: "#9ca3af", strokeDash: [4, 4] }),
          Plot.dot([{ x: minDelta, y: exact[minIndex] }], { x: "x", y: "y", r: 5, fill: "#166534" }),
        ],
      }),
      legend: [
        { label: "Full BCS action", color: "#166534", kind: "line" },
        { label: "Ginzburg-Landau approximation", color: "#d97706", dash: "dashed", kind: "line" },
        { label: `BCS gap Δ₀=${minDelta.toFixed(2)}`, color: "#166534", kind: "point" },
        { label: "Zero reference", color: "#9ca3af", dash: "dashed", kind: "line" },
      ],
      note: `State inferred from the quadratic coefficient: ${status}.`,
    };
  });
}
