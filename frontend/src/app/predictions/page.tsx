"use client";

/**
 * Predictions overview page - lists models with links to their prediction history.
 */

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { Header } from "@/components/Header";
import { StatusBadge } from "@/components/StatusBadge";
import { listModels } from "@/lib/models";
import { formatDate } from "@/lib/utils";
import type { ModelListResponse } from "@/types/api";

const PAGE_SIZE = 10;

export default function PredictionsPage() {
  const [data, setData] = useState<ModelListResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);

  const fetchModels = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      // Only show ready models since those are the ones that can have predictions
      const result = await listModels({ page, page_size: PAGE_SIZE });
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load models");
    } finally {
      setLoading(false);
    }
  }, [page]);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Header />

      <main className="container mx-auto px-4 py-8">
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Predictions
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Select a model to view its prediction history or run new predictions.
          </p>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
            <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
          </div>
        )}

        {loading ? (
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-8">
            <div className="flex justify-center">
              <div className="animate-pulse text-gray-500 dark:text-gray-400">
                Loading models...
              </div>
            </div>
          </div>
        ) : data && data.items.length > 0 ? (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {data.items.map((model) => (
                <div
                  key={model.id}
                  className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6"
                >
                  <div className="flex justify-between items-start mb-3">
                    <div>
                      <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                        {model.name}
                      </h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        v{model.version}
                      </p>
                    </div>
                    <StatusBadge status={model.status} />
                  </div>

                  {model.description && (
                    <p className="text-sm text-gray-600 dark:text-gray-300 mb-4 line-clamp-2">
                      {model.description}
                    </p>
                  )}

                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-4">
                    Created {formatDate(model.created_at)}
                  </p>

                  <div className="flex gap-2">
                    {model.status === "ready" && (
                      <Link
                        href={`/models/${model.id}/predict`}
                        className="flex-1 px-3 py-2 bg-green-600 text-white text-sm text-center rounded-md hover:bg-green-700 transition-colors"
                      >
                        Run Prediction
                      </Link>
                    )}
                    <Link
                      href={`/models/${model.id}/predictions`}
                      className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 text-sm text-center rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                    >
                      View History
                    </Link>
                  </div>
                </div>
              ))}
            </div>

            {/* Pagination */}
            {data.total_pages > 1 && (
              <div className="mt-6 flex justify-between items-center">
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Showing {(page - 1) * PAGE_SIZE + 1} to{" "}
                  {Math.min(page * PAGE_SIZE, data.total)} of {data.total} models
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
              <p className="text-gray-500 dark:text-gray-400 mb-4">
                No models found. Upload a model first to run predictions.
              </p>
              <Link
                href="/models/new"
                className="inline-block px-4 py-2 bg-gray-900 dark:bg-gray-700 text-white text-sm rounded-md hover:bg-gray-800 dark:hover:bg-gray-600 transition-colors"
              >
                Upload Model
              </Link>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
