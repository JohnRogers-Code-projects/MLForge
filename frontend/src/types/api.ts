/**
 * API response types for ModelForge.
 *
 * These types mirror the backend Pydantic schemas.
 */

// Model types
export type ModelStatus = "pending" | "uploaded" | "validating" | "ready" | "error" | "archived";

export interface Model {
  id: string;
  name: string;
  version: string;
  description: string | null;
  status: ModelStatus;
  file_path: string | null;
  file_size_bytes: number | null;
  file_hash: string | null;
  input_schema: Record<string, unknown>[] | null;
  output_schema: Record<string, unknown>[] | null;
  model_metadata: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
}

export interface ModelListResponse {
  items: Model[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface ModelUploadResponse {
  id: string;
  file_path: string;
  file_size_bytes: number;
  file_hash: string;
  status: ModelStatus;
  message: string;
}

export interface TensorSchema {
  name: string;
  dtype: string;
  shape: (number | null)[];
}

export interface ModelValidateResponse {
  id: string;
  valid: boolean;
  status: ModelStatus;
  input_schema: TensorSchema[] | null;
  output_schema: TensorSchema[] | null;
  model_metadata: Record<string, unknown> | null;
  error_message: string | null;
  message: string;
}

// Job types
export type JobStatus = "pending" | "queued" | "running" | "completed" | "failed" | "cancelled";
export type JobPriority = "low" | "normal" | "high";

export interface Job {
  id: string;
  model_id: string;
  status: JobStatus;
  priority: JobPriority;
  input_data: Record<string, unknown>;
  output_data: Record<string, unknown> | null;
  celery_task_id: string | null;
  worker_id: string | null;
  error_message: string | null;
  error_traceback: string | null;
  inference_time_ms: number | null;
  queue_time_ms: number | null;
  retries: number;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface JobListResponse {
  items: Job[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface JobResultResponse {
  job_id: string;
  status: JobStatus;
  result: Record<string, unknown> | null;
  error_message: string | null;
  error_traceback: string | null;
  inference_time_ms: number | null;
  completed_at: string | null;
}

// Prediction types
export interface Prediction {
  id: string;
  model_id: string;
  input_data: Record<string, unknown>;
  output_data: Record<string, unknown> | null;
  inference_time_ms: number | null;
  cached: boolean;
  created_at: string;
}

export interface PredictionListResponse {
  items: Prediction[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

// Health check types
export interface HealthResponse {
  status: string;
  version: string;
  environment: string;
  timestamp: string;
  database: string;
  redis: string;
  celery: string;
}

// Cache metrics types
export interface CacheMetrics {
  hits: number;
  misses: number;
  hit_rate: number;
}
