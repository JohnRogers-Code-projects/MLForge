/**
 * Model API service functions.
 */

import { api } from "./api";
import { config } from "./config";
import type { Model, ModelListResponse, ModelUploadResponse, ModelValidateResponse } from "@/types/api";

export interface CreateModelRequest {
  name: string;
  version: string;
  description?: string;
}

export interface UpdateModelRequest {
  name?: string;
  version?: string;
  description?: string;
}

export type ListModelsParams = Record<string, string | number | boolean | undefined>;

/**
 * List models with optional pagination and filters.
 */
export async function listModels(params?: ListModelsParams): Promise<ModelListResponse> {
  return api.get<ModelListResponse>("/models", params);
}

/**
 * Get a single model by ID.
 */
export async function getModel(id: string): Promise<Model> {
  return api.get<Model>(`/models/${id}`);
}

/**
 * Create a new model.
 */
export async function createModel(data: CreateModelRequest): Promise<Model> {
  return api.post<Model>("/models", data);
}

/**
 * Update an existing model.
 */
export async function updateModel(id: string, data: UpdateModelRequest): Promise<Model> {
  return api.patch<Model>(`/models/${id}`, data);
}

/**
 * Delete a model.
 */
export async function deleteModel(id: string): Promise<void> {
  return api.delete<void>(`/models/${id}`);
}

/**
 * Upload an ONNX file to a model.
 */
export async function uploadModelFile(id: string, file: File): Promise<ModelUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(
    `${config.api.baseUrl}${config.api.prefix}/models/${id}/upload`,
    {
      method: "POST",
      body: formData,
    }
  );

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Upload failed: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Validate an uploaded model.
 */
export async function validateModel(id: string): Promise<ModelValidateResponse> {
  return api.post<ModelValidateResponse>(`/models/${id}/validate`);
}

/**
 * Archive a model (set status to archived).
 */
export async function archiveModel(id: string): Promise<Model> {
  return api.patch<Model>(`/models/${id}`, { status: "archived" });
}
