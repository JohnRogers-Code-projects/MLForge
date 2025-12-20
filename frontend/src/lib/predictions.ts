/**
 * Prediction API service functions.
 */

import { api } from "./api";
import type { Prediction, PredictionListResponse } from "@/types/api";

export interface CreatePredictionRequest {
  input_data: Record<string, unknown>;
  request_id?: string;
  skip_cache?: boolean;
}

export type ListPredictionsParams = Record<string, string | number | boolean | undefined>;

/**
 * Run a prediction on a model.
 */
export async function createPrediction(
  modelId: string,
  data: CreatePredictionRequest
): Promise<Prediction> {
  return api.post<Prediction>(`/models/${modelId}/predict`, data);
}

/**
 * List predictions for a model with optional pagination.
 */
export async function listPredictions(
  modelId: string,
  params?: ListPredictionsParams
): Promise<PredictionListResponse> {
  return api.get<PredictionListResponse>(`/models/${modelId}/predictions`, params);
}
