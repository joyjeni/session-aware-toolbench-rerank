'use client';

import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

type ChartRow = {
  model: string;
  Hit1: number;
  NDCG5: number;
};

type TrendRow = {
  split: string;
  SessionRerank: number;
  Dense: number;
  BM25: number;
};

type Props = {
  comparisonData: ChartRow[];
  trendData: TrendRow[];
};

export default function MetricsFigures({ comparisonData, trendData }: Props) {
  return (
    <div className="figure-grid">
      <section className="chart-card">
        <h3>Session regime comparison</h3>
        <p className="body">
          Hit@1 and NDCG@5 for each model in the session regime of the first evaluation split.
        </p>
        <div style={{ width: '100%', height: 340 }}>
          <ResponsiveContainer>
            <BarChart data={comparisonData} margin={{ top: 20, right: 20, left: -12, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="model" tick={{ fontSize: 12 }} />
              <YAxis width={50} tickFormatter={(value) => value.toFixed(2)} />
              <Tooltip formatter={(value: number) => value.toFixed(3)} />
              <Legend />
              <Bar dataKey="Hit1" fill="#0f766e" barSize={24}>
                <LabelList dataKey="Hit1" position="top" formatter={(value: number) => value.toFixed(3)} />
              </Bar>
              <Bar dataKey="NDCG5" fill="#2563eb" barSize={24}>
                <LabelList dataKey="NDCG5" position="top" formatter={(value: number) => value.toFixed(3)} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="chart-card">
        <h3>Session rerank trend</h3>
        <p className="body">
          NDCG@5 over the selected splits for SessionRerank, Dense, and BM25 in the session regime.
        </p>
        <div style={{ width: '100%', height: 340 }}>
          <ResponsiveContainer>
            <LineChart data={trendData} margin={{ top: 20, right: 20, left: -12, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="split" tick={{ fontSize: 12 }} />
              <YAxis width={50} tickFormatter={(value) => value.toFixed(2)} />
              <Tooltip formatter={(value: number) => value.toFixed(3)} />
              <Legend />
              <Line type="monotone" dataKey="SessionRerank" stroke="#0f766e" strokeWidth={3} dot={{ r: 4 }}>
                <LabelList dataKey="SessionRerank" position="top" formatter={(value: number) => value.toFixed(3)} />
              </Line>
              <Line type="monotone" dataKey="Dense" stroke="#2563eb" strokeWidth={3} dot={{ r: 4 }}>
                <LabelList dataKey="Dense" position="top" formatter={(value: number) => value.toFixed(3)} />
              </Line>
              <Line type="monotone" dataKey="BM25" stroke="#d97706" strokeWidth={3} dot={{ r: 4 }}>
                <LabelList dataKey="BM25" position="top" formatter={(value: number) => value.toFixed(3)} />
              </Line>
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>
    </div>
  );
}
