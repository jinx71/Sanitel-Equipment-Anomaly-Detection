import type { AnalysisResult } from '../types';
import { anomalyLabel, fmt } from '../lib/format';

export default function FlaggedTable({ result }: { result: AnalysisResult }) {
  const rows = result.flagged;

  return (
    <section className="panel flex flex-col p-4">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <div className="eyebrow">Flagged readings</div>
          <h2 className="text-[15px] font-semibold text-text-hi">
            Ranked by ensemble score
          </h2>
        </div>
        <span className="font-mono text-[11px] text-text-lo">{rows.length} total</span>
      </div>

      <div className="-mx-1 max-h-[420px] overflow-auto">
        <table className="w-full border-collapse text-[12.5px]">
          <thead className="sticky top-0 bg-panel">
            <tr className="text-left text-text-lo">
              <th className="px-2 pb-2 font-medium">#</th>
              <th className="px-2 pb-2 font-medium">Score</th>
              <th className="px-2 pb-2 font-medium">Votes</th>
              <th className="px-2 pb-2 font-medium">Signature</th>
              <th className="hidden px-2 pb-2 font-medium md:table-cell">Top deviations</th>
              <th className="px-2 pb-2 text-right font-medium">Truth</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.index} className="border-t border-line-soft align-top">
                <td className="px-2 py-2 font-mono text-text-mid tnum">{r.index}</td>
                <td className="px-2 py-2">
                  <div className="flex items-center gap-1.5">
                    <div className="h-1.5 w-10 overflow-hidden rounded-full bg-line">
                      <div
                        className="h-full rounded-full bg-accent"
                        style={{ width: `${r.ensemble_score * 100}%` }}
                      />
                    </div>
                    <span className="font-mono text-text-hi tnum">
                      {fmt(r.ensemble_score, 2)}
                    </span>
                  </div>
                </td>
                <td className="px-2 py-2 font-mono text-text-mid tnum">
                  {r.models_agree}/{r.n_models}
                </td>
                <td className="px-2 py-2 text-text-mid">{anomalyLabel(r.anomaly_type)}</td>
                <td className="hidden px-2 py-2 md:table-cell">
                  <div className="flex flex-wrap gap-1">
                    {r.deviations.map((d) => (
                      <span
                        key={d.key}
                        className="rounded border border-line-soft bg-panel-2 px-1.5 py-0.5 font-mono text-[10.5px] text-text-mid"
                      >
                        {d.label.split(' ')[0]} {(d.z ?? 0) >= 0 ? '+' : ''}
                        {fmt(d.z ?? 0, 1)}
                      </span>
                    ))}
                  </div>
                </td>
                <td className="px-2 py-2 text-right">
                  <span
                    className="inline-block rounded px-1.5 py-0.5 font-mono text-[10px] uppercase"
                    style={{
                      color: r.true_anomaly ? '#F43F5E' : '#F5A524',
                      background: '#0A0F1A',
                    }}
                  >
                    {r.true_anomaly ? 'true' : 'false'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
