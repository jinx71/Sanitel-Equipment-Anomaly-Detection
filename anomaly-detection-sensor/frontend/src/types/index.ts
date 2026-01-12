// Shapes mirror the FastAPI pipeline payload (backend/app/pipeline.py).

export interface ApiEnvelope<T> {
  success: boolean;
  data: T;
  message: string;
}

export interface SensorMeta {
  key: string;
  label: string;
  unit: string;
}

export interface EquipmentProfile {
  key: string;
  label: string;
  description: string;
  n_features: number;
  sensors: SensorMeta[];
}

export interface ModelMeta {
  name: string;
  label: string;
  family: string;
  description: string;
}

export interface FeatureSpec {
  key: string;
  label: string;
  unit: string;
  low: number;
  high: number;
  decimals: number;
}

export interface Metrics {
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number | null;
  pr_auc: number | null;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
}

export interface ModelResult {
  name: string;
  label: string;
  description: string;
  threshold: number;
  metrics: Metrics;
  scores: number[];
  predictions: number[];
}

export interface EnsembleResult {
  label: string;
  model_names: string[];
  threshold: number;
  metrics: Metrics;
  scores: number[];
  predictions: number[];
}

export type AnomalyType = 'spike' | 'drift' | 'correlation_break' | 'excursion' | null;

export interface SeriesRow {
  index: number;
  timestamp: string;
  sensors: Record<string, number>;
  is_true_anomaly: boolean;
  anomaly_type: AnomalyType;
  ensemble_score: number;
  ensemble_pred: number;
  models_agree: number;
  cluster: number;
}

export interface ProjectionPoint {
  index: number;
  x: number;
  y: number;
  is_anomaly: number;
  is_true_anomaly: boolean;
  cluster: number;
}

export interface FeatureDeviation {
  key: string;
  label: string;
  z?: number;
  deviation?: number;
  direction: 'high' | 'low';
}

export interface ClusterSummary {
  cluster: number;
  size: number;
  dominant_type: AnomalyType;
  top_features: FeatureDeviation[];
}

export interface FlaggedRow {
  index: number;
  timestamp: string;
  ensemble_score: number;
  models_agree: number;
  n_models: number;
  anomaly_type: AnomalyType;
  true_anomaly: boolean;
  cluster: number;
  values: Record<string, number>;
  deviations: FeatureDeviation[];
}

export interface DatasetMeta {
  equipment: string;
  label: string;
  description: string;
  n_samples: number;
  n_features: number;
  feature_names: FeatureSpec[];
  anomaly_rate: number;
  n_true_anomalies: number;
  seed: number;
  sample_interval_s: number;
}

export interface AnalysisResult {
  dataset: DatasetMeta;
  series: SeriesRow[];
  models: ModelResult[];
  ensemble: EnsembleResult;
  projection: { method: string; points: ProjectionPoint[] };
  clusters: ClusterSummary[];
  flagged: FlaggedRow[];
  summary: { n_flagged: number; n_models: number; contamination: number };
}

export interface AnalyzeParams {
  equipment: string;
  n_samples: number;
  anomaly_rate: number;
  contamination: number;
  seed: number | null;
  models: string[] | null;
  projection_method: 'pca' | 'umap';
}
