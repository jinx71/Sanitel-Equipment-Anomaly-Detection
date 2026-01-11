import type { AnalysisResult } from '../types';
import { pct } from '../lib/format';

export default function ModelComparison({ result }: { result: AnalysisResult }) {
  const rows = result.models;
  const bestPr = Math.max(...rows.map((m) => m.metrics.pr_auc ?? 0));

  return (
    <section className="panel flex flex-col p-4">
      <div className="mb-3">
        <div className="eyebrow">Detector comparison</div>
        <h2 className="text-[15px] font-semibold text-text-hi">
          Per-model quality vs the ensemble
        </h2>
      </div>

      <div className="-mx-1 overflow-x-auto">
        <table className="w-full border-collapse text-[13px]">
          <thead>
            <tr className="text-left text-text-lo">
              <th className="px-1 pb-2 font-medium">Detector</th>
              <th className="px-1 pb-2 font-medium">PR-AUC</th>
              <th className="hidden px-1 pb-2 text-right font-medium sm:table-cell">ROC</th>
              <th className="px-1 pb-2 text-right font-medium">F1</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((m) => {
              const pr = m.metrics.pr_auc ?? 0;
              const isBest = pr === bestPr;
              return (
                <tr key={m.name} className="border-t border-line-soft">
                  <td className="px-1 py-2">
                    <div className="flex items-center gap-2">
                      <span className="text-text-hi">{m.label}</span>
                      {isBest && (
                        <span className="rounded bg-normal/15 px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider text-normal">
                          best
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="px-1 py-2">
                    <div className="flex items-center gap-2">
                      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-line">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${pr * 100}%`,
                            background: isBest ? '#2DD4BF' : '#38BDF8',
                          }}
                        />
                      </div>
                      <span className="font-mono text-text-mid tnum">{pct(pr)}</span>
                    </div>
                  </td>
                  <td className="hidden px-1 py-2 text-right font-mono text-text-mid tnum sm:table-cell">
                    {pct(m.metrics.roc_auc)}
                  </td>
                  <td className="px-1 py-2 text-right font-mono text-text-mid tnum">
                    {pct(m.metrics.f1)}
                  </td>
                </tr>
              );
            })}

            {/* ensemble row */}
            <tr className="border-t-2 border-line">
              <td className="px-1 py-2 font-semibold text-accent">Ensemble</td>
              <td className="px-1 py-2">
                <div className="flex items-center gap-2">
                  <div className="h-1.5 w-16 overflow-hidden rounded-full bg-line">
                    <div
                      className="h-full rounded-full bg-accent"
                      style={{ width: `${(result.ensemble.metrics.pr_auc ?? 0) * 100}%` }}
                    />
                  </div>
                  <span className="font-mono text-text-hi tnum">
                    {pct(result.ensemble.metrics.pr_auc)}
                  </span>
                </div>
              </td>
              <td className="hidden px-1 py-2 text-right font-mono text-text-hi tnum sm:table-cell">
                {pct(result.ensemble.metrics.roc_auc)}
              </td>
              <td className="px-1 py-2 text-right font-mono text-text-hi tnum">
                {pct(result.ensemble.metrics.f1)}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <p className="mt-3 text-[11px] leading-relaxed text-text-lo">
        Metrics are computed against held-out ground truth for reporting only — no
        detector sees labels during training.
      </p>
    </section>
  );
}
