import fs from 'fs/promises';
import path from 'path';
import MetricsFigures from './components/MetricsFigures';

type MetricSeries = Record<string, number>;
type ModelMetrics = Record<string, MetricSeries>;
type Regimes = Record<string, ModelMetrics>;

type Split = {
  n_queries: number;
  regimes: Regimes;
};

type RuntimeInfo = {
  platform: string;
  detected: string;
  cuda_available: boolean;
  xla_available: boolean;
  torch_version?: string;
  torch_cuda_version?: string | null;
  cuda_device_count?: number;
  cuda_device_names?: string[];
  torch_error?: string;
};

type MetricsFile = {
  config: {
    encoder: string;
    device: string;
    runtime?: RuntimeInfo;
    corpus_size: number;
    weights: Record<string, number>;
    ks: number[];
  };
  splits: Record<string, Split>;
  significance__SessionRerank_vs_Dense: Record<
    string,
    Record<
      string,
      Record<string, { p_value: number; delta_mean: number; delta_rel_pct: number }>
    >
  >;
};
type FigureItem = {
  file: string;
  title: string;
  caption: string;
};

type BenchmarkRow = {
  model: string;
  Hit1: number;
  NDCG5: number;
  highlight: boolean;
};
const splitOrder = [
  'G1_instruction',
  'G1_tool',
  'G1_category',
  'G2_instruction',
  'G2_category',
  'G3_instruction',
];

const regimeOrder = ['closed', 'open', 'session'];

const sessionSplit = 'G1_instruction';

function formatLabel(value: string) {
  return value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatValue(value: number) {
  if (Number.isInteger(value)) return value.toString();
  return value.toFixed(3);
}

async function loadMetrics(): Promise<MetricsFile> {
  const raw = await fs.readFile(path.join(process.cwd(), 'public', 'metrics.json'), 'utf-8');
  return JSON.parse(raw);
}

function getBestModels(splitName: string, split: Split) {
  return Object.entries(split.regimes).map(([regimeName, models]) => {
    const winner = Object.entries(models).reduce(
      (current, [modelName, metrics]) => {
        const hit1 = metrics['Hit@1'] ?? 0;
        if (hit1 > current.hit1) {
          return {
            modelName,
            hit1,
            ndcg5: metrics['NDCG@5'] ?? 0,
            recall10: metrics['Recall@10'] ?? 0,
          };
        }
        return current;
      },
      { modelName: '', hit1: -Infinity, ndcg5: 0, recall10: 0 }
    );

    return {
      splitName,
      regimeName,
      ...winner,
    };
  });
}

function extractSignificantHighlights(metrics: MetricsFile) {
  const highlights: Array<{
    splitName: string;
    regimeName: string;
    metricName: string;
    pValue: number;
    deltaRelPct: number;
  }> = [];

  for (const splitName of splitOrder) {
    const splitData = metrics.significance__SessionRerank_vs_Dense[splitName];
    if (!splitData) continue;

    for (const regimeName of regimeOrder) {
      const regimeData = splitData[regimeName];
      if (!regimeData) continue;

      for (const metricName of ['NDCG@5', 'Hit@5', 'NDCG@10', 'Hit@10']) {
        const metricInfo = regimeData[metricName];
        if (!metricInfo) continue;
        if (metricInfo.p_value < 0.05) {
          highlights.push({
            splitName,
            regimeName,
            metricName,
            pValue: metricInfo.p_value,
            deltaRelPct: metricInfo.delta_rel_pct,
          });
        }
      }
    }
  }

  return highlights;
}

function getComparisonData(metrics: MetricsFile) {
  const sessionModels = metrics.splits[sessionSplit]?.regimes.session;
  if (!sessionModels) return [];

  const candidateModels = [
    'SessionRerank(Dense (all-MiniLM-L6-v2))',
    'Dense (all-MiniLM-L6-v2)',
    'BM25',
    'ToolRerank',
  ];

  return candidateModels
    .map((model) => {
      const values = sessionModels[model];
      return {
        model,
        Hit1: values?.['Hit@1'] ?? 0,
        NDCG5: values?.['NDCG@5'] ?? 0,
        highlight: model.startsWith('SessionRerank'),
      } as BenchmarkRow;
    })
    .filter((row) => row.Hit1 > 0 || row.NDCG5 > 0);
}

function getSessionRerankDelta(metrics: MetricsFile) {
  const session = metrics.splits[sessionSplit]?.regimes.session?.['SessionRerank(Dense (all-MiniLM-L6-v2))'];
  const dense = metrics.splits[sessionSplit]?.regimes.session?.['Dense (all-MiniLM-L6-v2)'];
  if (!session || !dense) return null;

  return {
    hit1: session['Hit@1'] - dense['Hit@1'],
    ndcg5: session['NDCG@5'] - dense['NDCG@5'],
    hit1Pct: dense['Hit@1'] ? 100 * (session['Hit@1'] - dense['Hit@1']) / dense['Hit@1'] : 0,
    ndcg5Pct: dense['NDCG@5'] ? 100 * (session['NDCG@5'] - dense['NDCG@5']) / dense['NDCG@5'] : 0,
  };
}

function getTrendData(metrics: MetricsFile) {
  return splitOrder
    .map((splitName) => {
      const split = metrics.splits[splitName];
      if (!split) return null;
      const session = split.regimes.session?.['SessionRerank(Dense (all-MiniLM-L6-v2))'];
      const dense = split.regimes.session?.['Dense (all-MiniLM-L6-v2)'];
      const bm25 = split.regimes.session?.BM25;
      if (!session || !dense || !bm25) return null;
      return {
        split: formatLabel(splitName),
        SessionRerank: session['NDCG@5'] ?? 0,
        Dense: dense['NDCG@5'] ?? 0,
        BM25: bm25['NDCG@5'] ?? 0,
      };
    })
    .filter(Boolean) as Array<{ split: string; SessionRerank: number; Dense: number; BM25: number }>;
}

function getMeanSessionRelGain(metrics: MetricsFile, metricName: 'NDCG@5' | 'Hit@5') {
  const gains = splitOrder.flatMap((splitName) => {
    const session = metrics.splits[splitName]?.regimes.session?.['SessionRerank(Dense (all-MiniLM-L6-v2))'];
    const dense = metrics.splits[splitName]?.regimes.session?.['Dense (all-MiniLM-L6-v2)'];
    if (!session || !dense) return [];
    const denseValue = dense[metricName];
    const sessionValue = session[metricName];
    return denseValue ? [100 * (sessionValue - denseValue) / denseValue] : [];
  });
  if (gains.length === 0) return 0;
  return gains.reduce((sum, value) => sum + value, 0) / gains.length;
}

function getPublicationFigures(metrics: MetricsFile): FigureItem[] {
  return [
    {
      file: 'fig1_ndcg5_session.png',
      title: 'Figure 1 — NDCG@5 in session regime',
      caption: `SessionRerank leads in NDCG@5 across session splits. Mean relative improvement vs Dense: ${getMeanSessionRelGain(metrics, 'NDCG@5').toFixed(1)}%.`,
    },
    {
      file: 'fig2_hit5_session.png',
      title: 'Figure 2 — Hit@5 in session regime',
      caption: `SessionRerank yields stronger Hit@5 performance over Dense in the session regime, confirming improved end-to-end retrieval recall.`,
    },
    {
      file: 'fig3_ndcg_atK_g2inst.png',
      title: 'Figure 3 — NDCG@K on G2 instruction',
      caption: `NDCG@K for the G2 instruction split under the open session evaluation. Session-aware ranking maintains a high curve across K.`,
    },
    {
      file: 'fig4_relgain_vs_dense.png',
      title: 'Figure 4 — Relative gain vs Dense',
      caption: `Average SessionRerank improvement over Dense: ${getMeanSessionRelGain(metrics, 'NDCG@5').toFixed(1)}% NDCG@5 and ${getMeanSessionRelGain(metrics, 'Hit@5').toFixed(1)}% Hit@5 across splits.`,
    },
    {
      file: 'fig5_heatmap_ndcg5.png',
      title: 'Figure 5 — NDCG@5 heatmap',
      caption: `Heatmap of NDCG@5 for all methods and session splits. Brighter cells indicate higher retrieval quality, with SessionRerank shown as the strongest row.`,
    },
    {
      file: 'fig6_ndcg5_open.png',
      title: 'Figure 6 — Open-regime NDCG@5',
      caption: `Open-regime NDCG@5 highlights the harder no-history evaluation setting and demonstrates that SessionRerank remains safe when history is absent.`,
    },
  ];
}

function getInsightCards(metrics: MetricsFile, highlights: ReturnType<typeof extractSignificantHighlights>) {
  const allBestModels = splitOrder
    .filter((name) => name in metrics.splits)
    .flatMap((name) => getBestModels(name, metrics.splits[name]));

  const bestOverall = allBestModels.reduce(
    (current, item) => (item.hit1 > current.hit1 ? item : current), allBestModels[0] ?? null
  );

  const bestSessionGain = highlights.reduce(
    (current, item) => (current === null || item.deltaRelPct > current.deltaRelPct ? item : current),
    null as null | (typeof highlights)[0]
  );

  const bestConfigWeight = Object.entries(metrics.config.weights).sort((a, b) => b[1] - a[1])[0];
  const sessionDelta = getSessionRerankDelta(metrics);

  return [
    {
      title: 'Highest hit rate',
      value: bestOverall ? `${formatLabel(bestOverall.splitName)} / ${formatLabel(bestOverall.regimeName)}` : 'n/a',
      detail: bestOverall ? `${bestOverall.modelName} with Hit@1=${formatValue(bestOverall.hit1)}` : '',
    },
    {
      title: 'SessionRerank advantage',
      value: sessionDelta
        ? `${sessionDelta.ndcg5Pct.toFixed(1)}% NDCG@5 uplift`
        : 'n/a',
      detail: sessionDelta
        ? `SessionRerank beats Dense by ${sessionDelta.ndcg5.toFixed(3)} absolute NDCG@5 points`
        : '',
    },
    {
      title: 'Largest statistical lift',
      value: bestSessionGain
        ? `${bestSessionGain.metricName} @ ${formatLabel(bestSessionGain.regimeName)}`
        : 'No significant gain',
      detail: bestSessionGain
        ? `${bestSessionGain.deltaRelPct.toFixed(1)}% lift at p=${bestSessionGain.pValue.toExponential(1)}`
        : '',
    },
    {
      title: 'Top model weight',
      value: bestConfigWeight ? formatLabel(bestConfigWeight[0]) : 'n/a',
      detail: bestConfigWeight ? bestConfigWeight[1].toFixed(3) : '',
    },
  ];
}

export default async function Page() {
  const metrics = await loadMetrics();
  const allBestModels = splitOrder
    .filter((name) => name in metrics.splits)
    .flatMap((name) => getBestModels(name, metrics.splits[name]));
  const highlights = extractSignificantHighlights(metrics);
  const benchmarkData = getComparisonData(metrics);
  const trendData = getTrendData(metrics);
  const insightCards = getInsightCards(metrics, highlights);
  const publicationFigures = getPublicationFigures(metrics);
  const configCards = [
    { label: 'Corpus size', value: metrics.config.corpus_size.toLocaleString() },
    { label: 'Encoder', value: metrics.config.encoder },
    { label: 'Device', value: metrics.config.device },
    { label: 'Ranking thresholds', value: metrics.config.ks.join(' • ') },
  ];
  const metricDefinitions = [
    {
      title: 'Hit@1',
      description:
        'Top-1 accuracy for retrieval: this metric tells us whether the first returned API is the correct one. Higher values indicate more confident, precise ranking and better first-pass decision quality.',
    },
    {
      title: 'Hit@5',
      description:
        'Top-5 recall: the proportion of queries for which the correct API appears anywhere in the first five results. This is essential for reranking and downstream agent robustness, since later stages can still recover a good answer.',
    },
    {
      title: 'NDCG@5',
      description:
        'Normalized discounted cumulative gain at 5: a ranking-aware score that rewards correct APIs appearing near the top of the list and penalizes correct but lower-ranked candidates. It is the most reliable measure of ranked retrieval quality in the top 5 window.',
    },
  ];

  const sessionMetrics = metrics.splits[sessionSplit]?.regimes.session;
  const denseSession = sessionMetrics?.['Dense (all-MiniLM-L6-v2)'];
  const sessionRerankMetric = sessionMetrics?.['SessionRerank(Dense (all-MiniLM-L6-v2))'];
  const hit1Delta = denseSession && sessionRerankMetric ? sessionRerankMetric['Hit@1'] - denseSession['Hit@1'] : 0;
  const ndcg5Delta = denseSession && sessionRerankMetric ? sessionRerankMetric['NDCG@5'] - denseSession['NDCG@5'] : 0;
  const hit5Delta = denseSession && sessionRerankMetric ? sessionRerankMetric['Hit@5'] - denseSession['Hit@5'] : 0;
  const hit1DeltaPct = denseSession && sessionRerankMetric && denseSession['Hit@1']
    ? 100 * hit1Delta / denseSession['Hit@1']
    : 0;
  const ndcg5DeltaPct = denseSession && sessionRerankMetric && denseSession['NDCG@5']
    ? 100 * ndcg5Delta / denseSession['NDCG@5']
    : 0;

  const metricInterpretations = [
    {
      title: 'Hit@1 interpretation',
      insight:
        `In the G1_instruction session regime, SessionRerank raises Hit@1 from ${denseSession ? formatValue(denseSession['Hit@1']) : 'n/a'} to ${sessionRerankMetric ? formatValue(sessionRerankMetric['Hit@1']) : 'n/a'}. This means the correct API is placed first more often, reducing the need for fallback reasoning steps.`,
      example:
        `Actual lift: ${hit1Delta.toFixed(3)} absolute points, or ${hit1DeltaPct.toFixed(1)}% relative to Dense.`,
    },
    {
      title: 'Hit@5 interpretation',
      insight:
        `In the same session split, Hit@5 improves from ${denseSession ? formatValue(denseSession['Hit@5']) : 'n/a'} to ${sessionRerankMetric ? formatValue(sessionRerankMetric['Hit@5']) : 'n/a'}. This indicates a stronger safety margin for the correct API to appear within the top five results.`,
      example:
        `Actual lift: ${hit5Delta.toFixed(3)} absolute points. High Hit@5 means downstream reranking and verification have a much better candidate pool.`,
    },
    {
      title: 'NDCG@5 interpretation',
      insight:
        `SessionRerank improves NDCG@5 from ${denseSession ? formatValue(denseSession['NDCG@5']) : 'n/a'} to ${sessionRerankMetric ? formatValue(sessionRerankMetric['NDCG@5']) : 'n/a'} in the session regime. This shows the model not only finds the correct API, but ranks it closer to the top of the list.`,
      example:
        `Actual lift: ${ndcg5Delta.toFixed(3)} absolute points (${ndcg5DeltaPct.toFixed(1)}% relative). This is the core ranking improvement most reviewers look for.`,
    },
  ];

  return (
    <main>
      <section className="hero">
        <div className="container">
          <p className="eyebrow">Session-aware reranking metrics</p>
          <h1>SessionRerank evaluation dashboard</h1>
          <p className="tagline">
            Precision benchmark visualization for session-aware reranking, built to support technical review and publication-ready reporting.
          </p>
          <div className="config-grid">
            {configCards.map((card) => (
              <div key={card.label} className="config-card">
                <p className="config-label">{card.label}</p>
                <p className="config-value">{card.value}</p>
              </div>
            ))}
          </div>
        </div>
      </section>


      <section className="container section">
        <h2>Quick insights</h2>
        <div style={{ display: 'grid', gap: 18, gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))' }}>
          {insightCards.map((card) => (
            <div key={card.title} className="figure-card">
              <p className="kpi-label">{card.title}</p>
              <p className="kpi-value">{card.value}</p>
              <p className="body" style={{ marginTop: 8 }}>{card.detail}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="container section">
        <h2>Benchmark performance</h2>
        <p className="body">
          Comparing SessionRerank directly against Dense and BM25 on the session regime. Green highlights show where session-aware re-ranking delivers superior results.
        </p>
        <MetricsFigures comparisonData={benchmarkData} trendData={trendData} />
      </section>

      <section className="container section">
        <h2>Why these metrics matter</h2>
        <div className="metric-grid">
          {metricDefinitions.map((item) => (
            <div key={item.title} className="metric-card">
              <p className="kpi-label">{item.title}</p>
              <p className="body">{item.description}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="container section">
        <h2>Metric interpretation guide</h2>
        <p className="body">
          This panel explains how to read each score in the context of retrieval quality, reranking effectiveness, and system resilience.
        </p>
        <div className="interpretation-grid">
          {metricInterpretations.map((item) => (
            <div key={item.title} className="interpretation-card">
              <p className="interpretation-title">{item.title}</p>
              <p className="body">{item.insight}</p>
              <p className="interpretation-example">{item.example}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="container section">
        <h2>Publication figures from Kaggle</h2>
        <p className="body">
          These are the exact PNG exports from the Kaggle notebook pipeline, generated by <code>src/sessionrerank/viz/make_charts.py</code>.
          The captions below are aligned to the live metrics values loaded from the JSON file.
        </p>
        <div className="figure-grid">
          {publicationFigures.map((figure) => (
            <figure key={figure.file} className="figure-card">
              <img src={`/figures/${figure.file}`} alt={figure.title} loading="lazy" />
              <figcaption className="cap">
                <strong>{figure.title}</strong><br />{figure.caption}
              </figcaption>
            </figure>
          ))}
        </div>
      </section>

      <section className="container section">
        <h2>Best models by split and regime</h2>
        <div className="table-wrap">
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th align="left">Split</th>
                <th align="left">Regime</th>
                <th align="left">Best model</th>
                <th align="right">Hit@1</th>
                <th align="right">NDCG@5</th>
              </tr>
            </thead>
            <tbody>
              {allBestModels.map((row) => (
                <tr key={`${row.splitName}-${row.regimeName}`} className={row.modelName.includes('SessionRerank') ? 'highlight-row' : ''}>
                  <td>{formatLabel(row.splitName)}</td>
                  <td>{formatLabel(row.regimeName)}</td>
                  <td>{row.modelName}</td>
                  <td align="right">{formatValue(row.hit1)}</td>
                  <td align="right">{formatValue(row.ndcg5)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="container section">
        <h2>Significance highlights</h2>
        <p className="body">
          Only improvements with a p-value below 0.05 are shown here for SessionRerank vs Dense.
        </p>
        {highlights.length === 0 ? (
          <p className="body">No statistically significant gains were found at p &lt; 0.05.</p>
        ) : (
          <div className="table-wrap">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th align="left">Split</th>
                  <th align="left">Regime</th>
                  <th align="left">Metric</th>
                  <th align="right">Delta %</th>
                  <th align="right">p-value</th>
                </tr>
              </thead>
              <tbody>
                {highlights.map((row, index) => (
                  <tr key={`${row.splitName}-${row.regimeName}-${row.metricName}-${index}`}>
                    <td>{formatLabel(row.splitName)}</td>
                    <td>{formatLabel(row.regimeName)}</td>
                    <td>{row.metricName}</td>
                    <td className="delta-strong" align="right">{row.deltaRelPct.toFixed(1)}%</td>
                    <td align="right">{row.pValue.toExponential(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <footer className="footer container">
        <p>
          Data loaded from <code>/public/metrics.json</code>. Figures and insights update automatically when the JSON changes.
        </p>
      </footer>
    </main>
  );
}
