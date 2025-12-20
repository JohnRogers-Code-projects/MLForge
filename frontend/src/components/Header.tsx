"use client";

/**
 * Shared header component for authenticated pages.
 */

import { useSession, signOut } from "next-auth/react";
import Link from "next/link";
import Image from "next/image";

export function Header() {
  const { data: session } = useSession();

  return (
    <header className="border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
      <div className="container mx-auto px-4 py-4 flex justify-between items-center">
        <div className="flex items-center gap-8">
          <Link
            href="/"
            className="text-xl font-bold text-gray-900 dark:text-white"
          >
            ModelForge
          </Link>
          <nav className="hidden md:flex items-center gap-6">
            <Link
              href="/models"
              className="text-sm text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors"
            >
              Models
            </Link>
            <Link
              href="/jobs"
              className="text-sm text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors"
            >
              Jobs
            </Link>
            <Link
              href="/predictions"
              className="text-sm text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors"
            >
              Predictions
            </Link>
          </nav>
        </div>

        {session && (
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              {session.user.image && (
                <Image
                  src={session.user.image}
                  alt={session.user.name ?? "User avatar"}
                  width={32}
                  height={32}
                  className="rounded-full"
                />
              )}
              <span className="text-sm text-gray-700 dark:text-gray-300 hidden sm:inline">
                {session.user.name || session.user.email || "User"}
              </span>
            </div>
            <button
              onClick={() => signOut()}
              className="px-3 py-1.5 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
            >
              Sign out
            </button>
          </div>
        )}
      </div>
    </header>
  );
}
