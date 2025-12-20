/**
 * NextAuth.js API route handler.
 *
 * Handles all /api/auth/* requests for authentication.
 */

import NextAuth from "next-auth";
import { authOptions } from "@/lib/auth";

const handler = NextAuth(authOptions);

export { handler as GET, handler as POST };
