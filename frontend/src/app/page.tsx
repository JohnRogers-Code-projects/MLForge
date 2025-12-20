import { config } from "@/lib/config";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <main className="container mx-auto px-4 py-16">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-5xl font-bold text-gray-900 dark:text-white mb-6">
            ModelForge
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 mb-8">
            ML Model Serving Platform
          </p>
          <p className="text-gray-500 dark:text-gray-400 mb-12">
            Deploy, manage, and serve machine learning models with ease.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12">
            <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                Models
              </h3>
              <p className="text-gray-500 dark:text-gray-400 text-sm">
                Upload and manage ONNX models with versioning support.
              </p>
            </div>
            <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                Predictions
              </h3>
              <p className="text-gray-500 dark:text-gray-400 text-sm">
                Run inference with caching and batch processing.
              </p>
            </div>
            <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                Jobs
              </h3>
              <p className="text-gray-500 dark:text-gray-400 text-sm">
                Async job processing with real-time status updates.
              </p>
            </div>
          </div>

          <div className="mt-12">
            <p className="text-sm text-gray-400 dark:text-gray-500">
              API: <code className="bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded text-xs">
                {config.api.baseUrl}
              </code>
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
