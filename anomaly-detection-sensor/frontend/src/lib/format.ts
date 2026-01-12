import type { AnomalyType, ProjectionPoint } from '../types';

// Distinct, on-palette colors for anomaly clusters in the latent map.
const CLUSTER_COLORS = [
  '#F43F5E', // rose
  '#F5A524', // amber
  '#A78BFA', // violet
  '#38BDF8', // sky
  '#FB7185', // pink
  '#FBBF24', // gold
  '#22D3EE', // cyan
  '#F472B6', // magenta
];

export const NORMAL_CLUSTER = -1;
export const NOISE_CLUSTER = -2;

export function clusterColor(cluster: number): string {
  if (cluster === NORMAL_CLUSTER) return '#2DD4BF'; // in-control teal
  if (cluster === NOISE_CLUSTER) return '#7C8AA3'; // isolated anomaly, grey
  return CLUSTER_COLORS[cluster % CLUSTER_COLORS.length];
}

const ANOMALY_LABELS: Record<string, string> = {
  spike: 'Spike',
  drift: 'Drift',
  correlation_break: 'Correlation break',
  excursion: 'Sustained excursion',
};

export function anomalyLabel(type: AnomalyType): string {
  if (!type) return 'Normal';
  return ANOMALY_LABELS[type] ?? type;
}

export function fmt(value: number, decimals = 2): string {
  if (!Number.isFinite(value)) return '—';
  return value.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

export function pct(value: number | null): string {
  if (value === null || value === undefined) return '—';
  return `${(value * 100).toFixed(1)}%`;
}

export function shortTime(iso: string): string {
  // Synthetic timestamps are 1-minute spaced; show HH:MM for trend axes.
  return iso.slice(11, 16);
}

// Andrew's monotone-chain convex hull — used to draw the "validated operating
// envelope" around in-control points in the latent map.
export function convexHull(points: ProjectionPoint[]): ProjectionPoint[] {
  if (points.length < 3) return points;
  const pts = [...points].sort((a, b) => (a.x === b.x ? a.y - b.y : a.x - b.x));
  const cross = (o: ProjectionPoint, a: ProjectionPoint, b: ProjectionPoint) =>
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

  const lower: ProjectionPoint[] = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0)
      lower.pop();
    lower.push(p);
  }
  const upper: ProjectionPoint[] = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0)
      upper.pop();
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}
