/**
 * Job API service functions.
 */

import { api } from "./api";
import type { Job, JobListResponse, JobResultResponse, JobStatus } from "@/types/api";

export interface CreateJobRequest {
  model_id: string;
  input_data: Record<string, unknown>;
  priority?: "low" | "normal" | "high";
}

export interface ListJobsParams {
  page?: number;
  page_size?: number;
  status?: JobStatus;
}

/**
 * Create a new async inference job.
 */
export async function createJob(data: CreateJobRequest): Promise<Job> {
  return api.post<Job>("/jobs", data);
}

/**
 * List jobs with optional pagination and status filter.
 */
export async function listJobs(params?: ListJobsParams): Promise<JobListResponse> {
  const queryParams: Record<string, string | number | undefined> = {
    page: params?.page,
    page_size: params?.page_size,
    status: params?.status,
  };
  return api.get<JobListResponse>("/jobs", queryParams);
}

/**
 * Get a specific job by ID.
 */
export async function getJob(id: string): Promise<Job> {
  return api.get<Job>(`/jobs/${id}`);
}

/**
 * Get job result with optional wait.
 * Returns JobResultResponse if completed, or a processing status if still running.
 */
export async function getJobResult(
  id: string,
  wait: number = 0
): Promise<JobResultResponse | { job_id: string; status: string; message: string }> {
  const response = await fetch(
    `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/v1/jobs/${id}/result?wait=${wait}`,
    { method: "GET" }
  );

  const data = await response.json();

  if (response.status === 202) {
    // Still processing
    return data as { job_id: string; status: string; message: string };
  }

  if (!response.ok) {
    throw new Error(data.detail || `Failed to get job result: ${response.statusText}`);
  }

  return data as JobResultResponse;
}

/**
 * Cancel a pending, queued, or running job.
 */
export async function cancelJob(id: string): Promise<Job> {
  return api.post<Job>(`/jobs/${id}/cancel`);
}

/**
 * Delete a completed, failed, or cancelled job.
 */
export async function deleteJob(id: string): Promise<void> {
  return api.delete<void>(`/jobs/${id}`);
}
