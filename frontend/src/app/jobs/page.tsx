"use client";

/**
 * Job queue dashboard page - lists all jobs with filtering and actions.
 */

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { Header } from "@/components/Header";
import { JobStatusBadge } from "@/components/JobStatusBadge";
import { listJobs, cancelJob, deleteJob } from "@/lib/jobs";
import { formatDateTime, formatDuration } from "@/lib/utils";
import type { Job, JobListResponse, JobStatus } from "@/types/api";

const PAGE_SIZE = 20;
const PREVIEW_ID_LENGTH = 8;
const POLL_INTERVAL_MS = 5000;

const JOB_STATUSES: (JobStatus | "all")[] = [
  "all",
  "pending",
  "queued",
  "running",
  "completed",
  "failed",
  "cancelled",
];

export default function JobsPage() {
  const [data, setData] = useState<JobListResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [statusFilter, setStatusFilter] = useState<JobStatus | "all">("all");
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [selectedJob, setSelectedJob] = useState<Job | null>(null);

  const fetchJobs = useCallback(async () => {
    try {
      setError(null);
      const params: { page: number; page_size: number; status?: JobStatus } = {
        page,
        page_size: PAGE_SIZE,
      };
      if (statusFilter !== "all") {
        params.status = statusFilter;
      }
      const result = await listJobs(params);
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load jobs");
    } finally {
      setLoading(false);
    }
  }, [page, statusFilter]);

  useEffect(() => {
    fetchJobs();
  }, [fetchJobs]);

  // Polling for real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      fetchJobs();
    }, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [fetchJobs]);

  const handleCancel = async (job: Job) => {
    if (!confirm(`Cancel job ${job.id.slice(0, PREVIEW_ID_LENGTH)}...?`)) return;
    try {
      setActionLoading(job.id);
      await cancelJob(job.id);
      await fetchJobs();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to cancel job");
    } finally {
      setActionLoading(null);
    }
  };

  const handleDelete = async (job: Job) => {
    if (!confirm(`Delete job ${job.id.slice(0, PREVIEW_ID_LENGTH)}...? This cannot be undone.`)) return;
    try {
      setActionLoading(job.id);
      await deleteJob(job.id);
      if (selectedJob?.id === job.id) {
        setSelectedJob(null);
      }
      await fetchJobs();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete job");
    } finally {
      setActionLoading(null);
    }
  };

  const canCancel = (status: JobStatus) =>
    status === "pending" || status === "queued" || status === "running";

  const canDelete = (status: JobStatus) =>
    status === "completed" || status === "failed" || status === "cancelled";

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Header />

      <main className="container mx-auto px-4 py-8">
        <div className="mb-6 flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              Job Queue
            </h1>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              Monitor and manage async inference jobs. Auto-refreshes every 5 seconds.
            </p>
          </div>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
            <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
          </div>
        )}

        {/* Status Filter */}
        <div className="mb-6 flex gap-2 flex-wrap">
          {JOB_STATUSES.map((status) => (
            <button
              key={status}
              onClick={() => {
                setStatusFilter(status);
                setPage(1);
              }}
              className={`px-3 py-1.5 text-sm rounded-md transition-colors ${
                statusFilter === status
                  ? "bg-gray-900 text-white dark:bg-gray-100 dark:text-gray-900"
                  : "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700"
              }`}
            >
              {status === "all" ? "All" : status}
            </button>
          ))}
        </div>

        {loading ? (
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-8">
            <div className="flex justify-center" role="status" aria-label="Loading jobs">
              <div className="animate-pulse text-gray-500 dark:text-gray-400">
                Loading jobs...
              </div>
            </div>
          </div>
        ) : data && data.items.length > 0 ? (
          <>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
              <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                <thead className="bg-gray-50 dark:bg-gray-900">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Job ID
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Model
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Priority
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Created
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Duration
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                  {data.items.map((job) => (
                    <tr
                      key={job.id}
                      className="hover:bg-gray-50 dark:hover:bg-gray-700"
                    >
                      <td className="px-6 py-4 whitespace-nowrap">
                        <button
                          onClick={() => setSelectedJob(job)}
                          className="text-sm font-mono text-blue-600 dark:text-blue-400 hover:underline"
                        >
                          {job.id.slice(0, PREVIEW_ID_LENGTH)}...
                        </button>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <Link
                          href={`/models/${job.model_id}`}
                          className="text-sm text-blue-600 dark:text-blue-400 hover:underline font-mono"
                        >
                          {job.model_id.slice(0, PREVIEW_ID_LENGTH)}...
                        </Link>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <JobStatusBadge status={job.status} />
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`text-sm ${
                          job.priority === "high"
                            ? "text-red-600 dark:text-red-400 font-medium"
                            : job.priority === "low"
                            ? "text-gray-400 dark:text-gray-500"
                            : "text-gray-600 dark:text-gray-400"
                        }`}>
                          {job.priority}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                        {formatDateTime(job.created_at)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                        {job.inference_time_ms
                          ? formatDuration(job.inference_time_ms)
                          : job.status === "running"
                          ? "Running..."
                          : "-"}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                        <div className="flex justify-end gap-2">
                          {canCancel(job.status) && (
                            <button
                              onClick={() => handleCancel(job)}
                              disabled={actionLoading === job.id}
                              className="text-yellow-600 dark:text-yellow-400 hover:text-yellow-800 dark:hover:text-yellow-300 disabled:opacity-50"
                            >
                              {actionLoading === job.id ? "..." : "Cancel"}
                            </button>
                          )}
                          {canDelete(job.status) && (
                            <button
                              onClick={() => handleDelete(job)}
                              disabled={actionLoading === job.id}
                              className="text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-300 disabled:opacity-50"
                            >
                              {actionLoading === job.id ? "..." : "Delete"}
                            </button>
                          )}
                          <button
                            onClick={() => setSelectedJob(job)}
                            className="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300"
                          >
                            Details
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            {data.total_pages > 1 && (
              <div className="mt-6 flex justify-between items-center">
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Showing {(page - 1) * PAGE_SIZE + 1} to{" "}
                  {Math.min(page * PAGE_SIZE, data.total)} of {data.total} jobs
                </p>
                <div className="flex gap-2">
                  <button
                    onClick={() => setPage((p) => Math.max(1, p - 1))}
                    disabled={page === 1}
                    className="px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Previous
                  </button>
                  <span className="px-3 py-1.5 text-sm text-gray-600 dark:text-gray-400">
                    Page {page} of {data.total_pages}
                  </span>
                  <button
                    onClick={() => setPage((p) => Math.min(data.total_pages, p + 1))}
                    disabled={page === data.total_pages}
                    className="px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Next
                  </button>
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-8">
            <div className="text-center">
              <p className="text-gray-500 dark:text-gray-400">
                {statusFilter === "all"
                  ? "No jobs found. Create a job by running an async prediction."
                  : `No ${statusFilter} jobs found.`}
              </p>
            </div>
          </div>
        )}

        {/* Job Detail Modal */}
        {selectedJob && (
          <div
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
            onClick={() => setSelectedJob(null)}
            role="dialog"
            aria-modal="true"
            aria-labelledby="job-modal-title"
            tabIndex={-1}
            onKeyDown={(e) => {
              if (e.key === "Escape") {
                setSelectedJob(null);
              }
            }}
          >
            <div
              className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
                <h2 id="job-modal-title" className="text-lg font-bold text-gray-900 dark:text-white">
                  Job Details
                </h2>
                <button
                  onClick={() => setSelectedJob(null)}
                  className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                  aria-label="Close"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              <div className="p-6 overflow-y-auto max-h-[calc(90vh-8rem)]">
                {/* Status and Actions */}
                <div className="flex justify-between items-center mb-6">
                  <JobStatusBadge status={selectedJob.status} size="md" />
                  <div className="flex gap-2">
                    {canCancel(selectedJob.status) && (
                      <button
                        onClick={() => handleCancel(selectedJob)}
                        disabled={actionLoading === selectedJob.id}
                        className="px-3 py-1.5 text-sm bg-yellow-100 text-yellow-800 rounded-md hover:bg-yellow-200 disabled:opacity-50"
                      >
                        {actionLoading === selectedJob.id ? "Cancelling..." : "Cancel Job"}
                      </button>
                    )}
                    {canDelete(selectedJob.status) && (
                      <button
                        onClick={() => handleDelete(selectedJob)}
                        disabled={actionLoading === selectedJob.id}
                        className="px-3 py-1.5 text-sm bg-red-100 text-red-800 rounded-md hover:bg-red-200 disabled:opacity-50"
                      >
                        {actionLoading === selectedJob.id ? "Deleting..." : "Delete Job"}
                      </button>
                    )}
                  </div>
                </div>

                {/* Metadata Grid */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  <div>
                    <span className="text-sm text-gray-500 dark:text-gray-400">Job ID</span>
                    <p className="font-mono text-sm text-gray-900 dark:text-white break-all">
                      {selectedJob.id}
                    </p>
                  </div>
                  <div>
                    <span className="text-sm text-gray-500 dark:text-gray-400">Model ID</span>
                    <p className="font-mono text-sm text-gray-900 dark:text-white break-all">
                      <Link
                        href={`/models/${selectedJob.model_id}`}
                        className="text-blue-600 dark:text-blue-400 hover:underline"
                      >
                        {selectedJob.model_id}
                      </Link>
                    </p>
                  </div>
                  <div>
                    <span className="text-sm text-gray-500 dark:text-gray-400">Priority</span>
                    <p className="text-sm text-gray-900 dark:text-white capitalize">
                      {selectedJob.priority}
                    </p>
                  </div>
                  <div>
                    <span className="text-sm text-gray-500 dark:text-gray-400">Retries</span>
                    <p className="text-sm text-gray-900 dark:text-white">
                      {selectedJob.retries}
                    </p>
                  </div>
                </div>

                {/* Timing */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  <div>
                    <span className="text-sm text-gray-500 dark:text-gray-400">Created</span>
                    <p className="text-sm text-gray-900 dark:text-white">
                      {formatDateTime(selectedJob.created_at)}
                    </p>
                  </div>
                  <div>
                    <span className="text-sm text-gray-500 dark:text-gray-400">Started</span>
                    <p className="text-sm text-gray-900 dark:text-white">
                      {selectedJob.started_at ? formatDateTime(selectedJob.started_at) : "-"}
                    </p>
                  </div>
                  <div>
                    <span className="text-sm text-gray-500 dark:text-gray-400">Completed</span>
                    <p className="text-sm text-gray-900 dark:text-white">
                      {selectedJob.completed_at ? formatDateTime(selectedJob.completed_at) : "-"}
                    </p>
                  </div>
                  <div>
                    <span className="text-sm text-gray-500 dark:text-gray-400">Queue Time</span>
                    <p className="text-sm text-gray-900 dark:text-white">
                      {formatDuration(selectedJob.queue_time_ms)}
                    </p>
                  </div>
                </div>

                {/* Worker Info */}
                {(selectedJob.celery_task_id || selectedJob.worker_id) && (
                  <div className="grid grid-cols-2 gap-4 mb-6">
                    <div>
                      <span className="text-sm text-gray-500 dark:text-gray-400">Celery Task ID</span>
                      <p className="font-mono text-sm text-gray-900 dark:text-white break-all">
                        {selectedJob.celery_task_id || "-"}
                      </p>
                    </div>
                    <div>
                      <span className="text-sm text-gray-500 dark:text-gray-400">Worker ID</span>
                      <p className="font-mono text-sm text-gray-900 dark:text-white break-all">
                        {selectedJob.worker_id || "-"}
                      </p>
                    </div>
                  </div>
                )}

                {/* Input Data */}
                <div className="mb-6">
                  <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Input Data
                  </h3>
                  <pre className="bg-gray-50 dark:bg-gray-900 rounded-md p-4 overflow-x-auto text-sm text-gray-800 dark:text-gray-200 font-mono">
                    {JSON.stringify(selectedJob.input_data, null, 2)}
                  </pre>
                </div>

                {/* Output Data */}
                {selectedJob.output_data && (
                  <div className="mb-6">
                    <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Output Data
                    </h3>
                    <pre className="bg-gray-50 dark:bg-gray-900 rounded-md p-4 overflow-x-auto text-sm text-gray-800 dark:text-gray-200 font-mono">
                      {JSON.stringify(selectedJob.output_data, null, 2)}
                    </pre>
                  </div>
                )}

                {/* Error Info */}
                {selectedJob.error_message && (
                  <div className="mb-6">
                    <h3 className="text-sm font-medium text-red-700 dark:text-red-300 mb-2">
                      Error Message
                    </h3>
                    <div className="bg-red-50 dark:bg-red-900/20 rounded-md p-4 border border-red-200 dark:border-red-800">
                      <p className="text-sm text-red-800 dark:text-red-200">
                        {selectedJob.error_message}
                      </p>
                    </div>
                  </div>
                )}

                {/* Error Traceback */}
                {selectedJob.error_traceback && (
                  <div>
                    <h3 className="text-sm font-medium text-red-700 dark:text-red-300 mb-2">
                      Error Traceback
                    </h3>
                    <pre className="bg-red-50 dark:bg-red-900/20 rounded-md p-4 overflow-x-auto text-sm text-red-800 dark:text-red-200 font-mono border border-red-200 dark:border-red-800">
                      {selectedJob.error_traceback}
                    </pre>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
