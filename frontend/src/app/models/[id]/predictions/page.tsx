"use client";

/**
 * Prediction history page for a specific model.
 */

import { useEffect, useState, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { Header } from "@/components/Header";
import { getModel } from "@/lib/models";
import { listPredictions } from "@/lib/predictions";
import { formatDateTime } from "@/lib/utils";
import type { Model, Prediction, PredictionListResponse } from "@/types/api";

const PAGE_SIZE = 20;
const PREVIEW_ID_LENGTH = 8;

export default function PredictionsHistoryPage() {
  const params = useParams();
  const id = params.id as string;

  const [model, setModel] = useState<Model | null>(null);
  const [data, setData] = useState<PredictionListResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [selectedPrediction, setSelectedPrediction] = useState<Prediction | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const [modelData, predictionsData] = await Promise.all([
        getModel(id),
        listPredictions(id, { page, page_size: PAGE_SIZE }),
      ]);
      setModel(modelData);
      setData(predictionsData);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load data");
    } finally {
      setLoading(false);
    }
  }, [id, page]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const exportToCsv = () => {
    if (!data || data.items.length === 0) return;

    const headers = [
      "id",
      "created_at",
      "inference_time_ms",
      "cached",
      "input_data",
      "output_data",
    ];

    // Escape CSV cell: handle quotes, newlines, and potential CSV injection
    const escapeCsvCell = (cell: string) => {
      let value = cell.replace(/"/g, '""').replace(/\r\n|\n|\r/g, "\\n");
      // Prevent CSV injection by neutralizing values that look like formulas
      if (/^[=+\-@]/.test(value)) {
        value = "'" + value;
      }
      return `"${value}"`;
    };

    const rows = data.items.map((p) => [
      p.id,
      p.created_at,
      p.inference_time_ms?.toString() ?? "",
      p.cached.toString(),
      JSON.stringify(p.input_data),
      JSON.stringify(p.output_data),
    ]);

    const csvContent = [
      headers.join(","),
      ...rows.map((row) => row.map((cell) => escapeCsvCell(cell)).join(",")),
    ].join("\n");

    // Sanitize filename: remove/replace filesystem-unsafe characters
    const sanitizeFilename = (name: string) =>
      name.replace(/[/\\:*?"<>|]/g, "-").replace(/\s+/g, "_");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    try {
      link.href = url;
      const safeName = sanitizeFilename(model?.name ?? id);
      link.download = `predictions-${safeName}-${new Date().toISOString().split("T")[0]}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } finally {
      URL.revokeObjectURL(url);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Header />

      <main className="container mx-auto px-4 py-8">
        <div className="mb-6 flex justify-between items-center">
          <Link
            href={`/models/${id}`}
            className="text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
          >
            &larr; Back to Model
          </Link>
          {model && (
            <Link
              href={`/models/${id}/predict`}
              className="px-4 py-2 bg-green-600 text-white text-sm rounded-md hover:bg-green-700 transition-colors"
            >
              New Prediction
            </Link>
          )}
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
            <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
          </div>
        )}

        <div className="flex justify-between items-center mb-6">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              Prediction History
            </h1>
            {model && (
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                {model.name} v{model.version}
              </p>
            )}
          </div>
          {data && data.items.length > 0 && (
            <button
              onClick={exportToCsv}
              className="px-4 py-2 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 text-sm rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
              title={`Exports current page only (up to ${PAGE_SIZE} predictions)`}
            >
              Export Page to CSV
            </button>
          )}
        </div>

        {loading ? (
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-8">
            <div className="flex justify-center">
              <div className="animate-pulse text-gray-500 dark:text-gray-400">
                Loading predictions...
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
                      ID
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Created
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Inference Time
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Cached
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                  {data.items.map((prediction) => (
                    <tr
                      key={prediction.id}
                      className="hover:bg-gray-50 dark:hover:bg-gray-700"
                    >
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className="text-sm font-mono text-gray-900 dark:text-white">
                          {prediction.id.slice(0, PREVIEW_ID_LENGTH)}...
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                        {formatDateTime(prediction.created_at)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                        {prediction.inference_time_ms?.toFixed(2) ?? "-"} ms
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        {prediction.cached ? (
                          <span className="px-2 py-1 text-xs font-medium rounded-full bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300">
                            Yes
                          </span>
                        ) : (
                          <span className="px-2 py-1 text-xs font-medium rounded-full bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300">
                            No
                          </span>
                        )}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                        <button
                          onClick={() => setSelectedPrediction(prediction)}
                          className="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300"
                        >
                          View Details
                        </button>
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
                  {Math.min(page * PAGE_SIZE, data.total)} of {data.total} predictions
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
                No predictions found for this model.
              </p>
              <Link
                href={`/models/${id}/predict`}
                className="inline-block px-4 py-2 bg-green-600 text-white text-sm rounded-md hover:bg-green-700 transition-colors"
              >
                Run First Prediction
              </Link>
            </div>
          </div>
        )}

        {/* Prediction Detail Modal */}
        {selectedPrediction && (
          <div
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
            onClick={() => setSelectedPrediction(null)}
            role="dialog"
            aria-modal="true"
            aria-labelledby="modal-title"
            tabIndex={-1}
            onKeyDown={(e) => {
              if (e.key === "Escape") {
                setSelectedPrediction(null);
              }
            }}
          >
            <div
              className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
                <h2 id="modal-title" className="text-lg font-bold text-gray-900 dark:text-white">
                  Prediction Details
                </h2>
                <button
                  onClick={() => setSelectedPrediction(null)}
                  className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                  aria-label="Close"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              <div className="p-6 overflow-y-auto max-h-[calc(90vh-8rem)]">
                {/* Metadata */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  <div>
                    <span className="text-sm text-gray-500 dark:text-gray-400">ID</span>
                    <p className="font-mono text-sm text-gray-900 dark:text-white break-all">
                      {selectedPrediction.id}
                    </p>
                  </div>
                  <div>
                    <span className="text-sm text-gray-500 dark:text-gray-400">Created</span>
                    <p className="text-sm text-gray-900 dark:text-white">
                      {formatDateTime(selectedPrediction.created_at)}
                    </p>
                  </div>
                  <div>
                    <span className="text-sm text-gray-500 dark:text-gray-400">Inference Time</span>
                    <p className="text-sm text-gray-900 dark:text-white">
                      {selectedPrediction.inference_time_ms?.toFixed(2) ?? "-"} ms
                    </p>
                  </div>
                  <div>
                    <span className="text-sm text-gray-500 dark:text-gray-400">Cached</span>
                    <p className="text-sm text-gray-900 dark:text-white">
                      {selectedPrediction.cached ? "Yes" : "No"}
                    </p>
                  </div>
                </div>

                {/* Input Data */}
                <div className="mb-6">
                  <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Input Data
                  </h3>
                  <pre className="bg-gray-50 dark:bg-gray-900 rounded-md p-4 overflow-x-auto text-sm text-gray-800 dark:text-gray-200 font-mono">
                    {JSON.stringify(selectedPrediction.input_data, null, 2)}
                  </pre>
                </div>

                {/* Output Data */}
                <div>
                  <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Output Data
                  </h3>
                  <pre className="bg-gray-50 dark:bg-gray-900 rounded-md p-4 overflow-x-auto text-sm text-gray-800 dark:text-gray-200 font-mono">
                    {JSON.stringify(selectedPrediction.output_data, null, 2)}
                  </pre>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
