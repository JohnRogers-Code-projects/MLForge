/**
 * Status badge component for displaying model status.
 */

import type { ModelStatus } from "@/types/api";

interface StatusBadgeProps {
  status: ModelStatus;
  size?: "sm" | "md";
}

const colors: Record<ModelStatus, string> = {
  pending: "bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300",
  uploaded: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300",
  validating: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300",
  ready: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300",
  error: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300",
  archived: "bg-gray-100 text-gray-500 dark:bg-gray-700 dark:text-gray-400",
};

export function StatusBadge({ status, size = "sm" }: StatusBadgeProps) {
  const sizeClasses = size === "sm" ? "px-2 py-1 text-xs" : "px-2.5 py-1 text-sm";

  return (
    <span className={`${sizeClasses} font-medium rounded-full ${colors[status]}`}>
      {status}
    </span>
  );
}
