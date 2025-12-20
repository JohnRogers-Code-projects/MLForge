/**
 * Status badge component for displaying job status.
 */

import type { JobStatus } from "@/types/api";

interface JobStatusBadgeProps {
  status: JobStatus;
  size?: "sm" | "md";
}

const colors: Record<JobStatus, string> = {
  pending: "bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300",
  queued: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300",
  running: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300",
  completed: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300",
  failed: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300",
  cancelled: "bg-gray-100 text-gray-500 dark:bg-gray-700 dark:text-gray-400",
};

export function JobStatusBadge({ status, size = "sm" }: JobStatusBadgeProps) {
  const sizeClasses = size === "sm" ? "px-2 py-1 text-xs" : "px-2.5 py-1 text-sm";

  return (
    <span className={`${sizeClasses} font-medium rounded-full ${colors[status]}`}>
      {status}
    </span>
  );
}
