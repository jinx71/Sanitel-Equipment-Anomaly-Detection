import axios from 'axios';
import type {
  AnalysisResult,
  AnalyzeParams,
  ApiEnvelope,
  EquipmentProfile,
  ModelMeta,
} from '../types';

// In production set VITE_API_URL to the deployed API origin (no trailing
// slash). In dev it is empty and Vite proxies /api to the backend.
const baseURL = import.meta.env.VITE_API_URL ?? '';

const http = axios.create({ baseURL, timeout: 60_000 });

export async function getEquipment(): Promise<EquipmentProfile[]> {
  const { data } = await http.get<ApiEnvelope<EquipmentProfile[]>>('/api/equipment');
  return data.data;
}

export async function getModels(): Promise<ModelMeta[]> {
  const { data } = await http.get<ApiEnvelope<ModelMeta[]>>('/api/models');
  return data.data;
}

export async function analyze(params: AnalyzeParams): Promise<AnalysisResult> {
  const { data } = await http.post<ApiEnvelope<AnalysisResult>>('/api/analyze', params);
  return data.data;
}
