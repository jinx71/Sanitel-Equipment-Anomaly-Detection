import { useCallback, useEffect, useRef, useState } from 'react';
import { analyze, getEquipment, getModels } from './api/client';
import type { AnalysisResult, AnalyzeParams, EquipmentProfile, ModelMeta } from './types';
import ControlPanel from './components/ControlPanel';
import MetricsBar from './components/MetricsBar';
import LatentMap from './components/LatentMap';
import ModelComparison from './components/ModelComparison';
import ScoreDistribution from './components/ScoreDistribution';
import TrendCharts from './components/TrendCharts';
import ClusterPanel from './components/ClusterPanel';
import FlaggedTable from './components/FlaggedTable';

const DEFAULT_PARAMS: AnalyzeParams = {
  equipment: 'wfi_loop',
  n_samples: 600,
  anomaly_rate: 0.06,
  contamination: 0.06,
  seed: 7,
  models: null,
  projection_method: 'pca',
};

export default function App() {
  const [equipment, setEquipment] = useState<EquipmentProfile[]>([]);
  const [models, setModels] = useState<ModelMeta[]>([]);
  const [params, setParams] = useState<AnalyzeParams>(DEFAULT_PARAMS);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const didAutoRun = useRef(false);

  const run = useCallback(async (p: AnalyzeParams) => {
    setLoading(true);
    setError(null);
    try {
      const res = await analyze(p);
      setResult(res);
    } catch (e) {
      setError(
        e instanceof Error
          ? `Analysis failed: ${e.message}. Is the API running on :8000?`
          : 'Analysis failed.'
      );
    } finally {
      setLoading(false);
    }
  }, []);

  // Load metadata, then auto-run a first analysis so the console is never empty.
  useEffect(() => {
    (async () => {
      try {
        const [eq, md] = await Promise.all([getEquipment(), getModels()]);
        setEquipment(eq);
        setModels(md);
        if (!didAutoRun.current) {
          didAutoRun.current = true;
          await run(DEFAULT_PARAMS);
        }
      } catch {
        setError('Could not reach the API. Start the backend on :8000 and reload.');
      }
    })();
  }, [run]);

  const update = (next: Partial<AnalyzeParams>) => setParams((prev) => ({ ...prev, ...next }));

  const reshuffle = () => {
    const seed = Math.floor(Math.random() * 100000);
    const next = { ...params, seed };
    setParams(next);
    run(next);
  };

  return (
    <div className="min-h-screen">
      <header className="border-b border-line bg-panel/60 backdrop-blur">
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="relative flex h-9 w-9 items-center justify-center rounded-md border border-line bg-panel-2">
              <span className="absolute h-2 w-2 rounded-full bg-normal animate-pulse-ring" />
              <span className="h-2 w-2 rounded-full bg-normal" />
            </div>
            <div>
              <h1 className="font-mono text-[15px] font-semibold tracking-wide text-text-hi">
                SENTINEL
              </h1>
              <p className="text-[11px] text-text-lo">
                Unsupervised anomaly detection · equipment telemetry
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {result && (
              <span className="hidden font-mono text-[11px] text-text-mid sm:inline">
                {result.dataset.label} · {result.summary.n_models} detectors ·{' '}
                {(result.summary.contamination * 100).toFixed(0)}% budget
              </span>
            )}
            <button
              onClick={reshuffle}
              disabled={loading}
              className="rounded-md border border-line bg-panel-2 px-3 py-1.5 text-[12px] text-text-mid transition-colors hover:text-text-hi disabled:opacity-50"
            >
              ↻ New sample
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1400px] px-4 py-5 sm:px-6">
        {error && (
          <div className="mb-4 rounded-md border border-alarm/40 bg-alarm/10 px-4 py-3 text-[13px] text-alarm">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[300px_1fr]">
          <div className="lg:sticky lg:top-5 lg:h-[calc(100vh-2.5rem)]">
            <ControlPanel
              equipment={equipment}
              models={models}
              params={params}
              onChange={update}
              onRun={() => run(params)}
              loading={loading}
            />
          </div>

          <div className="min-w-0">
            {!result && loading && (
              <div className="flex h-96 items-center justify-center panel">
                <div className="flex flex-col items-center gap-3 text-text-lo">
                  <div className="h-8 w-8 animate-spin rounded-full border-2 border-line border-t-accent" />
                  <span className="font-mono text-[12px]">Running detectors…</span>
                </div>
              </div>
            )}

            {result && (
              <div className="flex animate-fade-up flex-col gap-4">
                <MetricsBar result={result} />
                <LatentMap result={result} />
                <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
                  <ModelComparison result={result} />
                  <ScoreDistribution result={result} />
                </div>
                <TrendCharts result={result} />
                <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
                  <ClusterPanel result={result} />
                  <FlaggedTable result={result} />
                </div>
              </div>
            )}
          </div>
        </div>

        <footer className="mt-8 border-t border-line-soft pt-4 text-center text-[11px] text-text-lo">
          Sentinel · synthetic GMP equipment telemetry · labels used for evaluation only,
          never for training
        </footer>
      </main>
    </div>
  );
}
