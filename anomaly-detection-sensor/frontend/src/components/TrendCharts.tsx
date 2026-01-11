import { useMemo } from 'react';
import {
  ComposedChart,
  Line,
  ReferenceArea,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { AnalysisResult, FeatureSpec } from '../types';

function SensorChart({ result, spec }: { result: AnalysisResult; spec: FeatureSpec }) {
  const data = useMemo(() => {
    return result.series.map((row) => {
      const value = row.sensors[spec.key];
      const flagged = row.ensemble_pred === 1;
      return {
        index: row.index,
        value,
        anomTrue: flagged && row.is_true_anomaly ? value : null,
        anomFalse: flagged && !row.is_true_anomaly ? value : null,
      };
    });
  }, [result, spec.key]);

  return (
    <div className="rounded-md border border-line-soft bg-panel-2/40 p-2">
      <div className="mb-1 flex items-baseline justify-between px-1">
        <span className="text-[12px] font-medium text-text-mid">{spec.label}</span>
        <span className="font-mono text-[10px] text-text-lo">
          {spec.low}–{spec.high} {spec.unit}
        </span>
      </div>
      <div className="h-28 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 4, right: 6, left: -22, bottom: -6 }}>
            {/* in-spec operating band */}
            <ReferenceArea
              y1={spec.low}
              y2={spec.high}
              fill="#2DD4BF"
              fillOpacity={0.06}
              stroke="#2DD4BF"
              strokeOpacity={0.18}
              strokeDasharray="3 3"
            />
            <XAxis dataKey="index" hide />
            <YAxis
              tick={{ fill: '#5E708A', fontSize: 9, fontFamily: 'IBM Plex Mono' }}
              tickLine={false}
              axisLine={false}
              width={42}
              domain={['auto', 'auto']}
            />
            <Tooltip
              contentStyle={{
                background: '#16212F',
                border: '1px solid #22304A',
                borderRadius: 8,
                fontSize: 11,
              }}
              labelStyle={{ color: '#9FB0C5', fontFamily: 'IBM Plex Mono' }}
              labelFormatter={(v) => `reading #${v}`}
              formatter={(value: number | string) => [value, spec.label]}
            />
            <Line
              type="monotone"
              dataKey="value"
              stroke="#38BDF8"
              strokeWidth={1.2}
              strokeOpacity={0.85}
              dot={false}
              isAnimationActive={false}
            />
            <Scatter dataKey="anomFalse" fill="#F5A524" />
            <Scatter dataKey="anomTrue" fill="#F43F5E" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default function TrendCharts({ result }: { result: AnalysisResult }) {
  return (
    <section className="panel p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="eyebrow">Sensor telemetry</div>
          <h2 className="text-[15px] font-semibold text-text-hi">
            Per-channel trends vs the in-spec band
          </h2>
        </div>
        <div className="flex items-center gap-3 text-[11px] text-text-mid">
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5 rounded-full bg-alarm" /> Caught anomaly
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5 rounded-full bg-warn" /> False flag
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-2 lg:grid-cols-2">
        {result.dataset.feature_names.map((spec) => (
          <SensorChart key={spec.key} result={result} spec={spec} />
        ))}
      </div>

      <p className="mt-3 text-[11px] leading-relaxed text-text-lo">
        Some flagged readings sit inside every single-sensor band — those are the
        multivariate correlation breaks a univariate alarm on any one channel would miss.
      </p>
    </section>
  );
}
