import type { AnalyzeParams, EquipmentProfile, ModelMeta } from '../types';

interface Props {
  equipment: EquipmentProfile[];
  models: ModelMeta[];
  params: AnalyzeParams;
  onChange: (next: Partial<AnalyzeParams>) => void;
  onRun: () => void;
  loading: boolean;
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  display,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (v: number) => void;
}) {
  return (
    <label className="block">
      <div className="mb-1.5 flex items-baseline justify-between">
        <span className="text-[13px] text-text-mid">{label}</span>
        <span className="font-mono text-[13px] text-text-hi tnum">{display}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-line accent-accent"
      />
    </label>
  );
}

export default function ControlPanel({
  equipment,
  models,
  params,
  onChange,
  onRun,
  loading,
}: Props) {
  const selected = new Set(params.models ?? models.map((m) => m.name));

  const toggleModel = (name: string) => {
    const next = new Set(selected);
    if (next.has(name)) {
      if (next.size > 1) next.delete(name); // keep at least one detector
    } else {
      next.add(name);
    }
    onChange({ models: models.filter((m) => next.has(m.name)).map((m) => m.name) });
  };

  return (
    <aside className="panel flex h-full flex-col gap-5 p-4">
      <div>
        <div className="eyebrow mb-2">Equipment</div>
        <select
          value={params.equipment}
          onChange={(e) => onChange({ equipment: e.target.value })}
          className="w-full rounded-md border border-line bg-panel-2 px-3 py-2 text-sm text-text-hi focus-visible:ring-2"
        >
          {equipment.map((p) => (
            <option key={p.key} value={p.key}>
              {p.label}
            </option>
          ))}
        </select>
        <p className="mt-2 text-[12px] leading-relaxed text-text-lo">
          {equipment.find((p) => p.key === params.equipment)?.description}
        </p>
      </div>

      <div>
        <div className="eyebrow mb-2">Detectors</div>
        <div className="flex flex-col gap-1.5">
          {models.map((m) => {
            const on = selected.has(m.name);
            return (
              <button
                key={m.name}
                onClick={() => toggleModel(m.name)}
                aria-pressed={on}
                title={m.description}
                className={`flex items-center justify-between rounded-md border px-3 py-2 text-left text-[13px] transition-colors ${
                  on
                    ? 'border-accent/50 bg-accent/10 text-text-hi'
                    : 'border-line bg-panel-2 text-text-lo hover:border-line hover:text-text-mid'
                }`}
              >
                <span className="font-medium">{m.label}</span>
                <span className="ml-2 font-mono text-[10px] uppercase tracking-wider text-text-lo">
                  {m.family.split(' ')[0]}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="flex flex-col gap-4">
        <div className="eyebrow">Dataset & budget</div>
        <Slider
          label="Readings"
          value={params.n_samples}
          min={200}
          max={1500}
          step={50}
          display={String(params.n_samples)}
          onChange={(v) => onChange({ n_samples: v })}
        />
        <Slider
          label="Injected anomaly rate"
          value={params.anomaly_rate}
          min={0.01}
          max={0.2}
          step={0.01}
          display={`${(params.anomaly_rate * 100).toFixed(0)}%`}
          onChange={(v) => onChange({ anomaly_rate: Number(v.toFixed(2)) })}
        />
        <Slider
          label="Detector budget (contamination)"
          value={params.contamination}
          min={0.01}
          max={0.2}
          step={0.01}
          display={`${(params.contamination * 100).toFixed(0)}%`}
          onChange={(v) => onChange({ contamination: Number(v.toFixed(2)) })}
        />
      </div>

      <div>
        <div className="eyebrow mb-2">Latent projection</div>
        <div className="flex rounded-md border border-line bg-panel-2 p-0.5">
          {(['pca', 'umap'] as const).map((method) => (
            <button
              key={method}
              onClick={() => onChange({ projection_method: method })}
              aria-pressed={params.projection_method === method}
              className={`flex-1 rounded px-3 py-1.5 text-[12px] font-medium uppercase tracking-wider transition-colors ${
                params.projection_method === method
                  ? 'bg-accent/15 text-accent'
                  : 'text-text-lo hover:text-text-mid'
              }`}
            >
              {method}
            </button>
          ))}
        </div>
      </div>

      <button
        onClick={onRun}
        disabled={loading}
        className="mt-auto rounded-md bg-accent px-4 py-2.5 text-sm font-semibold text-ink transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {loading ? 'Analysing…' : 'Run detection'}
      </button>
    </aside>
  );
}
