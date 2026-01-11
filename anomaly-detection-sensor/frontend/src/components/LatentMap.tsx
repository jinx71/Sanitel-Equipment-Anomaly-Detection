import { useMemo, useState } from 'react';
import type { AnalysisResult, ProjectionPoint } from '../types';
import {
  NOISE_CLUSTER,
  NORMAL_CLUSTER,
  anomalyLabel,
  clusterColor,
  convexHull,
  fmt,
} from '../lib/format';

const W = 1000;
const H = 620;
const PAD = 36;

export default function LatentMap({ result }: { result: AnalysisResult }) {
  const { points, method } = result.projection;
  const [hover, setHover] = useState<number | null>(null);

  const { project, hullPath, normal, anomalies } = useMemo(() => {
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const spanX = maxX - minX || 1;
    const spanY = maxY - minY || 1;

    const project = (p: ProjectionPoint) => ({
      px: PAD + ((p.x - minX) / spanX) * (W - 2 * PAD),
      py: PAD + (1 - (p.y - minY) / spanY) * (H - 2 * PAD),
    });

    const normal = points.filter((p) => p.cluster === NORMAL_CLUSTER);
    const anomalies = points.filter((p) => p.cluster !== NORMAL_CLUSTER);

    const hull = convexHull(normal);
    const hullPath =
      hull.length >= 3
        ? hull
            .map((p, i) => {
              const { px, py } = project(p);
              return `${i === 0 ? 'M' : 'L'}${px.toFixed(1)},${py.toFixed(1)}`;
            })
            .join(' ') + ' Z'
        : '';

    return { project, hullPath, normal, anomalies };
  }, [points]);

  const byIndex = useMemo(() => {
    const map = new Map<number, ProjectionPoint>();
    points.forEach((p) => map.set(p.index, p));
    return map;
  }, [points]);

  const hovered = hover !== null ? byIndex.get(hover) : undefined;
  const hoveredRow = hovered ? result.series[hovered.index] : undefined;

  const clusterLegend = result.clusters.map((c) => ({
    cluster: c.cluster,
    label: anomalyLabel(c.dominant_type),
    size: c.size,
  }));

  return (
    <section className="panel p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="eyebrow">Latent operating space</div>
          <h2 className="text-[15px] font-semibold text-text-hi">
            Validated envelope & detected anomalies
          </h2>
        </div>
        <span className="font-mono text-[11px] uppercase tracking-wider text-text-lo">
          {method} projection · {result.dataset.n_features}D → 2D
        </span>
      </div>

      <div className="relative">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="h-auto w-full"
          role="img"
          aria-label="Two-dimensional projection of sensor readings with detected anomalies"
        >
          <defs>
            <linearGradient id="envelope" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#2DD4BF" stopOpacity="0.12" />
              <stop offset="100%" stopColor="#2DD4BF" stopOpacity="0.04" />
            </linearGradient>
          </defs>

          {/* validated operating envelope */}
          {hullPath && (
            <path
              d={hullPath}
              fill="url(#envelope)"
              stroke="#2DD4BF"
              strokeOpacity="0.35"
              strokeWidth="1.5"
              strokeDasharray="4 4"
            />
          )}

          {/* in-control readings */}
          {normal.map((p) => {
            const { px, py } = project(p);
            return (
              <circle
                key={p.index}
                cx={px}
                cy={py}
                r={hover === p.index ? 4.5 : 2.6}
                fill="#2DD4BF"
                fillOpacity={hover === null || hover === p.index ? 0.55 : 0.25}
                onMouseEnter={() => setHover(p.index)}
                onMouseLeave={() => setHover(null)}
              />
            );
          })}

          {/* detected anomalies (on top) */}
          {anomalies.map((p) => {
            const { px, py } = project(p);
            const color = clusterColor(p.cluster);
            const active = hover === p.index;
            return (
              <g
                key={p.index}
                onMouseEnter={() => setHover(p.index)}
                onMouseLeave={() => setHover(null)}
                className="cursor-pointer"
              >
                {active && (
                  <circle cx={px} cy={py} r={6} fill={color} className="animate-pulse-ring" />
                )}
                <circle
                  cx={px}
                  cy={py}
                  r={active ? 7 : 5}
                  fill={color}
                  fillOpacity={0.92}
                  stroke={p.is_true_anomaly ? '#0A0F1A' : color}
                  strokeWidth={p.is_true_anomaly ? 1.5 : 0}
                />
              </g>
            );
          })}
        </svg>

        {hovered && hoveredRow && (
          <div className="pointer-events-none absolute left-3 top-3 w-56 rounded-md border border-line bg-panel-2/95 p-3 text-[12px] shadow-panel backdrop-blur">
            <div className="mb-1 flex items-center justify-between">
              <span className="font-mono text-text-hi">reading #{hovered.index}</span>
              <span
                className="rounded px-1.5 py-0.5 font-mono text-[10px] uppercase"
                style={{ color: clusterColor(hovered.cluster), background: '#0A0F1A' }}
              >
                {hovered.cluster === NORMAL_CLUSTER
                  ? 'in-control'
                  : hovered.cluster === NOISE_CLUSTER
                    ? 'isolated'
                    : `cluster ${hovered.cluster}`}
              </span>
            </div>
            <div className="flex justify-between text-text-mid">
              <span>Ensemble score</span>
              <span className="font-mono text-text-hi tnum">
                {fmt(hoveredRow.ensemble_score, 3)}
              </span>
            </div>
            <div className="flex justify-between text-text-mid">
              <span>Detector votes</span>
              <span className="font-mono text-text-hi tnum">
                {hoveredRow.models_agree}/{result.summary.n_models}
              </span>
            </div>
            <div className="mt-1 flex justify-between text-text-mid">
              <span>True label</span>
              <span style={{ color: hoveredRow.is_true_anomaly ? '#F43F5E' : '#2DD4BF' }}>
                {anomalyLabel(hoveredRow.anomaly_type)}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* legend */}
      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[11px] text-text-mid">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-normal opacity-60" />
          In-control envelope
        </span>
        {clusterLegend.map((c) => (
          <span key={c.cluster} className="flex items-center gap-1.5">
            <span
              className="inline-block h-2.5 w-2.5 rounded-full"
              style={{ background: clusterColor(c.cluster) }}
            />
            {c.label} ({c.size})
          </span>
        ))}
        <span className="ml-auto text-text-lo">Outlined = true anomaly</span>
      </div>
    </section>
  );
}
