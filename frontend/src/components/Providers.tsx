"use client";

/**
 * Client-side providers wrapper.
 *
 * Wraps children with SessionProvider for NextAuth.js context.
 */

import { SessionProvider } from "next-auth/react";

interface ProvidersProps {
  children: React.ReactNode;
}

export function Providers({ children }: ProvidersProps) {
  return <SessionProvider>{children}</SessionProvider>;
}
