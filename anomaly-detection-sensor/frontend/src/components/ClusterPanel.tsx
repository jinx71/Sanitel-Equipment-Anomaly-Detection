import type { AnalysisResult } from '../types';
import { anomalyLabel, clusterColor, fmt } from '../lib/format';

export default function ClusterPanel({ result }: { result: AnalysisResult }) {
  const clusters = result.clusters;

  return (
    <section className="panel flex flex-col p-4">
      <div className="mb-3">
        <div className="eyebrow">Anomaly signatures</div>
        <h2 className="text-[15px] font-semibold text-text-hi">
          DBSCAN clusters of flagged readings
        </h2>
      </div>

      {clusters.length === 0 ? (
        <p className="text-[13px] text-text-lo">
          No dense anomaly clusters — flagged readings are isolated outliers.
        </p>
      ) : (
        <div className="flex flex-col gap-2.5">
          {clusters.map((c) => {
            const color = clusterColor(c.cluster);
            const maxDev = Math.max(...c.top_features.map((f) => Math.abs(f.deviation ?? 0)), 1);
            return (
              <div
                key={c.cluster}
                className="rounded-md border border-line-soft bg-panel-2/50 p-3"
                style={{ borderLeft: `3px solid ${color}` }}
              >
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-[13px] font-semibold text-text-hi">
                    Cluster {c.cluster}
                    <span className="ml-2 font-normal text-text-lo">
                      {c.size} reading{c.size === 1 ? '' : 's'}
                    </span>
                  </span>
                  <span
                    className="rounded px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider"
                    style={{ color, background: '#0A0F1A' }}
                  >
                    {anomalyLabel(c.dominant_type)}
                  </span>
                </div>

                <div className="flex flex-col gap-1.5">
                  {c.top_features.map((f) => {
                    const dev = f.deviation ?? 0;
                    return (
                      <div key={f.key} className="flex items-center gap-2 text-[12px]">
                        <span className="w-36 shrink-0 truncate text-text-mid">{f.label}</span>
                        <div className="relative h-3 flex-1">
                          <div className="absolute left-1/2 top-0 h-full w-px bg-line" />
                          <div
                            className="absolute top-0.5 h-2 rounded-sm"
                            style={{
                              background: color,
                              opacity: 0.7,
                              width: `${(Math.abs(dev) / maxDev) * 50}%`,
                              left: dev >= 0 ? '50%' : undefined,
                              right: dev < 0 ? '50%' : undefined,
                            }}
                          />
                        </div>
                        <span className="w-14 shrink-0 text-right font-mono text-text-hi tnum">
                          {dev >= 0 ? '+' : ''}
                          {fmt(dev, 1)}σ
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      )}

      <p className="mt-3 text-[11px] leading-relaxed text-text-lo">
        Deviations are z-scores from the dataset mean. Each signature points an engineer
        toward a likely failure mode before they open the raw trace.
      </p>
    </section>
  );
}
