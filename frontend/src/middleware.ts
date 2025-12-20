/**
 * NextAuth.js middleware for route protection.
 *
 * Protects all routes except:
 * - / (home page - public landing)
 * - /login (auth page)
 * - /api/auth/* (NextAuth.js routes)
 * - /_next/* (Next.js internals)
 * - Static files
 */

import { withAuth } from "next-auth/middleware";

export default withAuth({
  pages: {
    signIn: "/login",
  },
});

// Protect these routes - require authentication
export const config = {
  matcher: [
    // Protected routes - add patterns for routes that require auth
    "/models/:path*",
    "/predictions/:path*",
    "/jobs/:path*",
    "/dashboard/:path*",
    "/settings/:path*",
  ],
};
