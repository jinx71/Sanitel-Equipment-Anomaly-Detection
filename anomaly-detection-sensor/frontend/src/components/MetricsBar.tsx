import type { AnalysisResult } from '../types';
import { pct } from '../lib/format';

function Stat({
  label,
  value,
  hint,
  tone = 'default',
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: 'default' | 'normal' | 'alarm';
}) {
  const valueColor =
    tone === 'normal' ? 'text-normal' : tone === 'alarm' ? 'text-alarm' : 'text-text-hi';
  return (
    <div className="flex flex-col gap-0.5 px-4 py-3">
      <span className="eyebrow">{label}</span>
      <span className={`font-mono text-[22px] font-medium tnum ${valueColor}`}>{value}</span>
      {hint && <span className="text-[11px] text-text-lo">{hint}</span>}
    </div>
  );
}

export default function MetricsBar({ result }: { result: AnalysisResult }) {
  const m = result.ensemble.metrics;
  const d = result.dataset;
  return (
    <div className="panel grid grid-cols-2 divide-x divide-y divide-line-soft sm:grid-cols-3 lg:grid-cols-6 lg:divide-y-0">
      <Stat
        label="PR-AUC"
        value={pct(m.pr_auc)}
        hint="ensemble, threshold-free"
        tone="normal"
      />
      <Stat label="ROC-AUC" value={pct(m.roc_auc)} hint="ensemble ranking" />
      <Stat label="Precision" value={pct(m.precision)} hint={`${m.tp} true / ${m.fp} false`} />
      <Stat label="Recall" value={pct(m.recall)} hint={`${m.fn} missed`} />
      <Stat
        label="Flagged"
        value={String(result.summary.n_flagged)}
        hint={`of ${d.n_samples} readings`}
        tone="alarm"
      />
      <Stat
        label="Ground truth"
        value={String(d.n_true_anomalies)}
        hint={`injected · seed ${d.seed}`}
      />
    </div>
  );
}
