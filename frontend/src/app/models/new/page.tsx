"use client";

/**
 * New model upload page with drag-and-drop.
 */

import { useState, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Header } from "@/components/Header";
import { createModel, uploadModelFile, validateModel } from "@/lib/models";

type UploadStep = "form" | "uploading" | "validating" | "done";

export default function NewModelPage() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [name, setName] = useState("");
  const [version, setVersion] = useState("1.0.0");
  const [description, setDescription] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [step, setStep] = useState<UploadStep>("form");
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState("");

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.name.endsWith(".onnx")) {
        setFile(droppedFile);
        setError(null);
        // Auto-fill name from filename if empty
        if (!name) {
          setName(droppedFile.name.replace(".onnx", ""));
        }
      } else {
        setError("Please upload an .onnx file");
      }
    }
  }, [name]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      if (selectedFile.name.endsWith(".onnx")) {
        setFile(selectedFile);
        setError(null);
        if (!name) {
          setName(selectedFile.name.replace(".onnx", ""));
        }
      } else {
        setError("Please upload an .onnx file");
      }
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!file) {
      setError("Please select an ONNX file to upload");
      return;
    }

    if (!name.trim()) {
      setError("Please enter a model name");
      return;
    }

    try {
      setError(null);
      setStep("uploading");
      setProgress("Creating model record...");

      // Step 1: Create model
      const model = await createModel({
        name: name.trim(),
        version: version.trim(),
        description: description.trim() || undefined,
      });

      setProgress("Uploading file...");

      // Step 2: Upload file
      await uploadModelFile(model.id, file);

      setStep("validating");
      setProgress("Validating model...");

      // Step 3: Validate
      await validateModel(model.id);

      setStep("done");
      setProgress("Model uploaded and validated successfully!");

      // Redirect after short delay
      setTimeout(() => {
        router.push(`/models/${model.id}`);
      }, 1500);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
      setStep("form");
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
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

        <div className="max-w-2xl mx-auto">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            Upload New Model
          </h1>

          {error && (
            <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
              <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
            </div>
          )}

          {step !== "form" ? (
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-8">
              <div className="text-center">
                {step === "done" ? (
                  <div className="text-green-500 mb-4">
                    <svg
                      className="w-16 h-16 mx-auto"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M5 13l4 4L19 7"
                      />
                    </svg>
                  </div>
                ) : (
                  <div className="animate-spin rounded-full h-16 w-16 border-4 border-gray-200 border-t-blue-600 mx-auto mb-4" />
                )}
                <p className="text-gray-600 dark:text-gray-300">{progress}</p>
              </div>
            </div>
          ) : (
            <form onSubmit={handleSubmit}>
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                {/* Drag and Drop Area */}
                <div
                  className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                    dragActive
                      ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                      : file
                      ? "border-green-500 bg-green-50 dark:bg-green-900/20"
                      : "border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500"
                  }`}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".onnx"
                    onChange={handleFileSelect}
                    className="hidden"
                  />

                  {file ? (
                    <div>
                      <div className="text-green-500 mb-2">
                        <svg
                          className="w-12 h-12 mx-auto"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                          />
                        </svg>
                      </div>
                      <p className="text-sm font-medium text-gray-900 dark:text-white">
                        {file.name}
                      </p>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        {formatFileSize(file.size)}
                      </p>
                      <button
                        type="button"
                        onClick={() => setFile(null)}
                        className="mt-2 text-sm text-red-600 dark:text-red-400 hover:underline"
                      >
                        Remove
                      </button>
                    </div>
                  ) : (
                    <div>
                      <div className="text-gray-400 mb-2">
                        <svg
                          className="w-12 h-12 mx-auto"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                          />
                        </svg>
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-300 mb-1">
                        Drag and drop your ONNX model here, or{" "}
                        <button
                          type="button"
                          onClick={() => fileInputRef.current?.click()}
                          className="text-blue-600 dark:text-blue-400 hover:underline"
                        >
                          browse
                        </button>
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        Only .onnx files are supported
                      </p>
                    </div>
                  )}
                </div>

                {/* Form Fields */}
                <div className="mt-6 space-y-4">
                  <div>
                    <label
                      htmlFor="name"
                      className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
                    >
                      Model Name *
                    </label>
                    <input
                      type="text"
                      id="name"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      placeholder="my-model"
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      required
                    />
                  </div>

                  <div>
                    <label
                      htmlFor="version"
                      className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
                    >
                      Version *
                    </label>
                    <input
                      type="text"
                      id="version"
                      value={version}
                      onChange={(e) => setVersion(e.target.value)}
                      placeholder="1.0.0"
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      required
                    />
                  </div>

                  <div>
                    <label
                      htmlFor="description"
                      className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
                    >
                      Description
                    </label>
                    <textarea
                      id="description"
                      value={description}
                      onChange={(e) => setDescription(e.target.value)}
                      placeholder="Optional description of your model..."
                      rows={3}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                </div>
              </div>

              <div className="mt-6 flex justify-end gap-3">
                <Link
                  href="/models"
                  className="px-4 py-2 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 text-sm rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                >
                  Cancel
                </Link>
                <button
                  type="submit"
                  disabled={!file}
                  className="px-4 py-2 bg-gray-900 dark:bg-gray-700 text-white text-sm rounded-md hover:bg-gray-800 dark:hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  Upload Model
                </button>
              </div>
            </form>
          )}
        </div>
      </main>
    </div>
  );
}
