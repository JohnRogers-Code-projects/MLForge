"use client";

/**
 * Model detail page with metadata display.
 */

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { Header } from "@/components/Header";
import { getModel, deleteModel, validateModel, archiveModel } from "@/lib/models";
import type { Model } from "@/types/api";

function StatusBadge({ status }: { status: Model["status"] }) {
  const colors: Record<Model["status"], string> = {
    pending: "bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300",
    uploaded: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300",
    validating: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300",
    ready: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300",
    error: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300",
    archived: "bg-gray-100 text-gray-500 dark:bg-gray-700 dark:text-gray-400",
  };

  return (
    <span className={`px-2.5 py-1 text-sm font-medium rounded-full ${colors[status]}`}>
      {status}
    </span>
  );
}

function formatBytes(bytes: number | null): string {
  if (bytes === null) return "-";
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

function formatDateTime(dateString: string): string {
  return new Date(dateString).toLocaleString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function MetadataSection({ title, data }: { title: string; data: unknown }) {
  if (!data) return null;

  return (
    <div className="mt-6">
      <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">
        {title}
      </h3>
      <pre className="bg-gray-50 dark:bg-gray-900 rounded-md p-4 overflow-x-auto text-sm text-gray-800 dark:text-gray-200">
        {JSON.stringify(data, null, 2)}
      </pre>
    </div>
  );
}

export default function ModelDetailPage() {
  const params = useParams();
  const router = useRouter();
  const id = params.id as string;

  const [model, setModel] = useState<Model | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  useEffect(() => {
    async function fetchModel() {
      try {
        setLoading(true);
        setError(null);
        const result = await getModel(id);
        setModel(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load model");
      } finally {
        setLoading(false);
      }
    }

    fetchModel();
  }, [id]);

  const handleValidate = async () => {
    if (!model) return;
    try {
      setActionLoading("validate");
      const updated = await validateModel(model.id);
      setModel(updated);
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to validate model");
    } finally {
      setActionLoading(null);
    }
  };

  const handleArchive = async () => {
    if (!model) return;
    if (!confirm(`Are you sure you want to archive "${model.name}"?`)) return;

    try {
      setActionLoading("archive");
      const updated = await archiveModel(model.id);
      setModel(updated);
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to archive model");
    } finally {
      setActionLoading(null);
    }
  };

  const handleDelete = async () => {
    if (!model) return;
    if (!confirm(`Are you sure you want to delete "${model.name}"? This action cannot be undone.`)) return;

    try {
      setActionLoading("delete");
      await deleteModel(model.id);
      router.push("/models");
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to delete model");
      setActionLoading(null);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Header />

      <main className="container mx-auto px-4 py-8">
        <div className="mb-6">
          <Link
            href="/models"
            className="text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
          >
            &larr; Back to Models
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
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
            {/* Header */}
            <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
              <div className="flex justify-between items-start">
                <div>
                  <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                    {model.name}
                  </h1>
                  <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                    Version {model.version}
                  </p>
                </div>
                <StatusBadge status={model.status} />
              </div>
              {model.description && (
                <p className="mt-2 text-gray-600 dark:text-gray-300">
                  {model.description}
                </p>
              )}
            </div>

            {/* Details */}
            <div className="px-6 py-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">
                    Model ID
                  </h3>
                  <p className="mt-1 text-sm text-gray-900 dark:text-white font-mono">
                    {model.id}
                  </p>
                </div>

                <div>
                  <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">
                    File Size
                  </h3>
                  <p className="mt-1 text-sm text-gray-900 dark:text-white">
                    {formatBytes(model.file_size_bytes)}
                  </p>
                </div>

                <div>
                  <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">
                    Created
                  </h3>
                  <p className="mt-1 text-sm text-gray-900 dark:text-white">
                    {formatDateTime(model.created_at)}
                  </p>
                </div>

                <div>
                  <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">
                    Updated
                  </h3>
                  <p className="mt-1 text-sm text-gray-900 dark:text-white">
                    {formatDateTime(model.updated_at)}
                  </p>
                </div>

                {model.file_hash && (
                  <div className="md:col-span-2">
                    <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">
                      File Hash (SHA-256)
                    </h3>
                    <p className="mt-1 text-sm text-gray-900 dark:text-white font-mono break-all">
                      {model.file_hash}
                    </p>
                  </div>
                )}
              </div>

              <MetadataSection title="Input Schema" data={model.input_schema} />
              <MetadataSection title="Output Schema" data={model.output_schema} />
              <MetadataSection title="Model Metadata" data={model.model_metadata} />
            </div>

            {/* Actions */}
            <div className="px-6 py-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 rounded-b-lg">
              <div className="flex flex-wrap gap-3">
                {model.status === "uploaded" && (
                  <button
                    onClick={handleValidate}
                    disabled={actionLoading !== null}
                    className="px-4 py-2 bg-blue-600 text-white text-sm rounded-md hover:bg-blue-700 disabled:opacity-50 transition-colors"
                  >
                    {actionLoading === "validate" ? "Validating..." : "Validate Model"}
                  </button>
                )}

                {model.status === "ready" && (
                  <Link
                    href={`/models/${model.id}/predict`}
                    className="px-4 py-2 bg-green-600 text-white text-sm rounded-md hover:bg-green-700 transition-colors"
                  >
                    Run Prediction
                  </Link>
                )}

                {model.status !== "archived" && (
                  <button
                    onClick={handleArchive}
                    disabled={actionLoading !== null}
                    className="px-4 py-2 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 text-sm rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 transition-colors"
                  >
                    {actionLoading === "archive" ? "Archiving..." : "Archive"}
                  </button>
                )}

                <button
                  onClick={handleDelete}
                  disabled={actionLoading !== null}
                  className="px-4 py-2 border border-red-300 dark:border-red-700 text-red-600 dark:text-red-400 text-sm rounded-md hover:bg-red-50 dark:hover:bg-red-900/20 disabled:opacity-50 transition-colors"
                >
                  {actionLoading === "delete" ? "Deleting..." : "Delete"}
                </button>
              </div>
            </div>
          </div>
        ) : null}
      </main>
    </div>
  );
}
