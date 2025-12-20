"use client";

import { useSession, signIn, signOut } from "next-auth/react";
import Link from "next/link";
import Image from "next/image";
import { config } from "@/lib/config";

function AuthButton() {
  const { data: session, status } = useSession();

  if (status === "loading") {
    return (
      <div className="h-10 w-24 bg-gray-200 dark:bg-gray-700 rounded-md animate-pulse" />
    );
  }

  if (session) {
    return (
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          {session.user.image && (
            <Image
              src={session.user.image}
              alt=""
              width={32}
              height={32}
              className="rounded-full"
            />
          )}
          <span className="text-sm text-gray-700 dark:text-gray-300">
            {session.user.name}
          </span>
        </div>
        <button
          onClick={() => signOut()}
          className="px-4 py-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
        >
          Sign out
        </button>
      </div>
    );
  }

  return (
    <button
      onClick={() => signIn()}
      className="px-4 py-2 bg-gray-900 dark:bg-gray-700 text-white text-sm rounded-md hover:bg-gray-800 dark:hover:bg-gray-600 transition-colors"
    >
      Sign in
    </button>
  );
}

export default function Home() {
  const { data: session } = useSession();

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <header className="border-b border-gray-200 dark:border-gray-700 bg-white/50 dark:bg-gray-800/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <Link href="/" className="text-xl font-bold text-gray-900 dark:text-white">
            ModelForge
          </Link>
          <AuthButton />
        </div>
      </header>

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

          {session && (
            <div className="mb-12">
              <p className="text-gray-600 dark:text-gray-300 mb-4">
                Welcome back, {session.user.name}!
              </p>
              <div className="flex justify-center gap-4">
                <Link
                  href="/models"
                  className="px-6 py-3 bg-gray-900 dark:bg-gray-700 text-white rounded-md hover:bg-gray-800 dark:hover:bg-gray-600 transition-colors"
                >
                  View Models
                </Link>
                <Link
                  href="/jobs"
                  className="px-6 py-3 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                >
                  View Jobs
                </Link>
              </div>
            </div>
          )}

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
              API:{" "}
              <code className="bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded text-xs">
                {config.api.baseUrl}
              </code>
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
