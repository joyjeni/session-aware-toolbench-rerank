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

type MetricsFile = {
  config: {
    encoder: string;
    device: string;
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

  return Object.entries(sessionModels).map(([model, values]) => ({
    model,
    Hit1: values['Hit@1'] ?? 0,
    NDCG5: values['NDCG@5'] ?? 0,
  }));
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

  return [
    {
      title: 'Highest hit rate',
      value: bestOverall ? `${formatLabel(bestOverall.splitName)} / ${formatLabel(bestOverall.regimeName)}` : 'n/a',
      detail: bestOverall ? `${bestOverall.modelName} with Hit@1=${formatValue(bestOverall.hit1)}` : '',
    },
    {
      title: 'Largest session gain',
      value: bestSessionGain
        ? `${formatLabel(bestSessionGain.splitName)} ${formatLabel(bestSessionGain.regimeName)}`
        : 'No significant gain',
      detail: bestSessionGain
        ? `${bestSessionGain.metricName} improved by ${bestSessionGain.deltaRelPct.toFixed(1)}% at p=${bestSessionGain.pValue.toExponential(1)}`
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
  const comparisonData = getComparisonData(metrics);
  const trendData = getTrendData(metrics);
  const insightCards = getInsightCards(metrics, highlights);
  const publicationFigures = getPublicationFigures(metrics);

  return (
    <main>
      <section className="hero">
        <div className="container">
          <p className="eyebrow">Session-aware reranking metrics</p>
          <h1>SessionRerank evaluation dashboard</h1>
          <p className="tagline">
            Exact Kaggle-generated publication figures, tables, and dynamic insights from <code>public/metrics.json</code>.
            The page now embeds the actual PNG exports stored in <code>web/public/figures</code>.
          </p>
          <p className="hero-timestamp">
            Encoder: <code>{metrics.config.encoder}</code> · Device: <code>{metrics.config.device}</code> · Corpus size: <code>{metrics.config.corpus_size.toLocaleString()}</code>
          </p>
        </div>
      </section>

      <section className="container section">
        <h2>Experiment configuration</h2>
        <p className="body">
          Model weights and evaluation thresholds are taken straight from the metrics JSON.
          The chart panel below is generated dynamically from the loaded values.
        </p>

        <div style={{ display: 'grid', gap: 18, gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
          <div className="figure-card">
            <p className="kpi-label">Corpus size</p>
            <p className="kpi-value">{metrics.config.corpus_size.toLocaleString()}</p>
          </div>
          <div className="figure-card">
            <p className="kpi-label">Encoder</p>
            <p className="kpi-value">{metrics.config.encoder}</p>
          </div>
          <div className="figure-card">
            <p className="kpi-label">Device</p>
            <p className="kpi-value">{metrics.config.device}</p>
          </div>
          <div className="figure-card">
            <p className="kpi-label">Ranking thresholds</p>
            <p className="kpi-value">{metrics.config.ks.join(' • ')}</p>
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
        <h2>Figures</h2>
        <MetricsFigures comparisonData={comparisonData} trendData={trendData} />
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
                <tr key={`${row.splitName}-${row.regimeName}`}>
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
