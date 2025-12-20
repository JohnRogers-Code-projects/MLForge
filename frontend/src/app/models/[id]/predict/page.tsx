"use client";

/**
 * Prediction form page for running inference on a model.
 */

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { Header } from "@/components/Header";
import { StatusBadge } from "@/components/StatusBadge";
import { getModel } from "@/lib/models";
import { createPrediction } from "@/lib/predictions";
import { formatDateTime } from "@/lib/utils";
import type { Model, Prediction } from "@/types/api";

/**
 * Create a sample array based on tensor shape for input pre-population.
 */
function createSampleArray(shape: (number | null)[], dtype: string): unknown {
  if (shape.length === 0) {
    return dtype.includes("int") ? 0 : 0.0;
  }

  const dim = shape[0] ?? 1; // Use 1 for dynamic dimensions
  const restShape = shape.slice(1);

  if (restShape.length === 0) {
    // Base case: create array of values
    return Array(dim).fill(dtype.includes("int") ? 0 : 0.0);
  }

  // Recursive case: create array of arrays
  return Array(dim)
    .fill(null)
    .map(() => createSampleArray(restShape, dtype));
}

export default function PredictPage() {
  const params = useParams();
  const id = params.id as string;

  const [model, setModel] = useState<Model | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Form state
  const [inputJson, setInputJson] = useState("{\n  \n}");
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [skipCache, setSkipCache] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  // Result state
  const [result, setResult] = useState<Prediction | null>(null);

  useEffect(() => {
    async function fetchModel() {
      try {
        setLoading(true);
        setError(null);
        const data = await getModel(id);
        setModel(data);

        // Pre-populate input JSON based on model's input schema if available
        if (data.input_schema && data.input_schema.length > 0) {
          const sampleInput: Record<string, unknown> = {};
          for (const schema of data.input_schema) {
            const tensorSchema = schema as { name: string; dtype: string; shape: (number | null)[] };
            // Create a sample array based on shape
            sampleInput[tensorSchema.name] = createSampleArray(tensorSchema.shape, tensorSchema.dtype);
          }
          setInputJson(JSON.stringify(sampleInput, null, 2));
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load model");
      } finally {
        setLoading(false);
      }
    }

    fetchModel();
  }, [id]);

  const validateJson = (value: string): boolean => {
    try {
      JSON.parse(value);
      setJsonError(null);
      return true;
    } catch (e) {
      setJsonError(e instanceof Error ? e.message : "Invalid JSON");
      return false;
    }
  };

  const handleInputChange = (value: string) => {
    setInputJson(value);
    if (value.trim()) {
      validateJson(value);
    } else {
      setJsonError(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validateJson(inputJson)) {
      return;
    }

    try {
      setSubmitting(true);
      setError(null);
      setResult(null);

      const inputData = JSON.parse(inputJson);
      const prediction = await createPrediction(id, {
        input_data: inputData,
        skip_cache: skipCache,
      });

      setResult(prediction);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setSubmitting(false);
    }
  };

  const formatJson = () => {
    try {
      const parsed = JSON.parse(inputJson);
      setInputJson(JSON.stringify(parsed, null, 2));
      setJsonError(null);
    } catch {
      // Already has error displayed
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Header />

      <main className="container mx-auto px-4 py-8">
        <div className="mb-6">
          <Link
            href={`/models/${id}`}
            className="text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
          >
            &larr; Back to Model
          </Link>
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
                Loading model...
              </div>
            </div>
          </div>
        ) : model ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Input Form */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
                <div className="flex justify-between items-center">
                  <div>
                    <h1 className="text-xl font-bold text-gray-900 dark:text-white">
                      Run Prediction
                    </h1>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                      {model.name} v{model.version}
                    </p>
                  </div>
                  <StatusBadge status={model.status} />
                </div>
              </div>

              {model.status !== "ready" ? (
                <div className="p-6">
                  <p className="text-gray-500 dark:text-gray-400">
                    Model must be in &quot;ready&quot; status to run predictions.
                    Current status: <StatusBadge status={model.status} />
                  </p>
                </div>
              ) : (
                <form onSubmit={handleSubmit} className="p-6">
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between items-center mb-2">
                        <label
                          htmlFor="input-json"
                          className="block text-sm font-medium text-gray-700 dark:text-gray-300"
                        >
                          Input Data (JSON)
                        </label>
                        <button
                          type="button"
                          onClick={formatJson}
                          className="text-xs text-blue-600 dark:text-blue-400 hover:underline"
                        >
                          Format JSON
                        </button>
                      </div>
                      <textarea
                        id="input-json"
                        value={inputJson}
                        onChange={(e) => handleInputChange(e.target.value)}
                        rows={12}
                        className={`w-full px-3 py-2 font-mono text-sm border rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent ${
                          jsonError
                            ? "border-red-500 dark:border-red-500"
                            : "border-gray-300 dark:border-gray-600"
                        }`}
                        placeholder='{"input": [[1.0, 2.0, 3.0]]}'
                      />
                      {jsonError && (
                        <p className="mt-1 text-sm text-red-600 dark:text-red-400">
                          {jsonError}
                        </p>
                      )}
                    </div>

                    <div className="flex items-center">
                      <input
                        type="checkbox"
                        id="skip-cache"
                        checked={skipCache}
                        onChange={(e) => setSkipCache(e.target.checked)}
                        className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                      />
                      <label
                        htmlFor="skip-cache"
                        className="ml-2 block text-sm text-gray-700 dark:text-gray-300"
                      >
                        Skip cache (force fresh inference)
                      </label>
                    </div>

                    {/* Input Schema Reference */}
                    {model.input_schema && model.input_schema.length > 0 && (
                      <div className="bg-gray-50 dark:bg-gray-900 rounded-md p-4">
                        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          Expected Input Schema
                        </h3>
                        <div className="space-y-1">
                          {model.input_schema.map((schema, idx) => {
                            const s = schema as { name: string; dtype: string; shape: (number | null)[] };
                            return (
                              <div key={idx} className="text-xs font-mono text-gray-600 dark:text-gray-400">
                                <span className="text-blue-600 dark:text-blue-400">{s.name}</span>
                                : {s.dtype} [{s.shape.map(d => d ?? "?").join(", ")}]
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="mt-6">
                    <button
                      type="submit"
                      disabled={submitting || !!jsonError}
                      className="w-full px-4 py-2 bg-green-600 text-white text-sm font-medium rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                      {submitting ? "Running Prediction..." : "Run Prediction"}
                    </button>
                  </div>
                </form>
              )}
            </div>

            {/* Result Panel */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
                <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                  Result
                </h2>
              </div>

              <div className="p-6">
                {submitting ? (
                  <div className="flex justify-center py-12">
                    <div className="animate-spin rounded-full h-8 w-8 border-2 border-gray-200 border-t-green-600" />
                  </div>
                ) : result ? (
                  <div className="space-y-4">
                    {/* Result Metadata */}
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-gray-500 dark:text-gray-400">Prediction ID</span>
                        <p className="font-mono text-gray-900 dark:text-white truncate">
                          {result.id}
                        </p>
                      </div>
                      <div>
                        <span className="text-gray-500 dark:text-gray-400">Inference Time</span>
                        <p className="text-gray-900 dark:text-white">
                          {result.inference_time_ms?.toFixed(2) ?? "-"} ms
                        </p>
                      </div>
                      <div>
                        <span className="text-gray-500 dark:text-gray-400">Cached</span>
                        <p className="text-gray-900 dark:text-white">
                          {result.cached ? (
                            <span className="text-green-600 dark:text-green-400">Yes</span>
                          ) : (
                            <span className="text-gray-600 dark:text-gray-400">No</span>
                          )}
                        </p>
                      </div>
                      <div>
                        <span className="text-gray-500 dark:text-gray-400">Created</span>
                        <p className="text-gray-900 dark:text-white">
                          {formatDateTime(result.created_at)}
                        </p>
                      </div>
                    </div>

                    {/* Output Data */}
                    <div>
                      <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Output Data
                      </h3>
                      <pre className="bg-gray-50 dark:bg-gray-900 rounded-md p-4 overflow-x-auto text-sm text-gray-800 dark:text-gray-200 font-mono">
                        {JSON.stringify(result.output_data, null, 2)}
                      </pre>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12 text-gray-500 dark:text-gray-400">
                    <p>Run a prediction to see results here.</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : null}

        {/* Link to Prediction History */}
        {model && (
          <div className="mt-6 text-center">
            <Link
              href={`/models/${id}/predictions`}
              className="text-sm text-blue-600 dark:text-blue-400 hover:underline"
            >
              View Prediction History for this Model &rarr;
            </Link>
          </div>
        )}
      </main>
    </div>
  );
}
