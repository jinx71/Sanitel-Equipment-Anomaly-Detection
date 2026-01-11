import { useMemo } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { AnalysisResult } from '../types';

const BINS = 22;

interface Bin {
  bin: number;
  label: string;
  normal: number;
  anomaly: number;
}

export default function ScoreDistribution({ result }: { result: AnalysisResult }) {
  const threshold = result.ensemble.threshold;

  const data = useMemo<Bin[]>(() => {
    const bins: Bin[] = Array.from({ length: BINS }, (_, i) => ({
      bin: i / BINS,
      label: (i / BINS).toFixed(2),
      normal: 0,
      anomaly: 0,
    }));
    result.ensemble.scores.forEach((s, i) => {
      const idx = Math.min(BINS - 1, Math.floor(s * BINS));
      if (result.series[i].is_true_anomaly) bins[idx].anomaly += 1;
      else bins[idx].normal += 1;
    });
    return bins;
  }, [result]);

  return (
    <section className="panel flex flex-col p-4">
      <div className="mb-3">
        <div className="eyebrow">Score separation</div>
        <h2 className="text-[15px] font-semibold text-text-hi">
          Ensemble anomaly score by true label
        </h2>
      </div>

      <div className="h-56 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 8, right: 8, left: -16, bottom: 4 }}>
            <CartesianGrid stroke="#1A2536" vertical={false} />
            <XAxis
              dataKey="label"
              tick={{ fill: '#5E708A', fontSize: 10, fontFamily: 'IBM Plex Mono' }}
              interval={4}
              tickLine={false}
              axisLine={{ stroke: '#22304A' }}
            />
            <YAxis
              tick={{ fill: '#5E708A', fontSize: 10, fontFamily: 'IBM Plex Mono' }}
              tickLine={false}
              axisLine={false}
              allowDecimals={false}
            />
            <Tooltip
              cursor={{ fill: 'rgba(56,189,248,0.06)' }}
              contentStyle={{
                background: '#16212F',
                border: '1px solid #22304A',
                borderRadius: 8,
                fontSize: 12,
              }}
              labelStyle={{ color: '#9FB0C5', fontFamily: 'IBM Plex Mono' }}
              labelFormatter={(v) => `score ≈ ${v}`}
            />
            <ReferenceLine
              x={data.reduce((best, b) =>
                Math.abs(b.bin - threshold) < Math.abs(best.bin - threshold) ? b : best
              ).label}
              stroke="#F5A524"
              strokeDasharray="4 3"
              label={{
                value: 'threshold',
                fill: '#F5A524',
                fontSize: 10,
                position: 'top',
              }}
            />
            <Bar dataKey="normal" stackId="s" fill="#2DD4BF" fillOpacity={0.7} name="In-control" />
            <Bar dataKey="anomaly" stackId="s" fill="#F43F5E" name="Anomaly" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-2 flex items-center gap-4 text-[11px] text-text-mid">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm bg-normal opacity-70" /> In-control
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm bg-alarm" /> True anomaly
        </span>
        <span className="ml-auto text-text-lo">
          Readings right of the threshold are flagged
        </span>
      </div>
    </section>
  );
}
