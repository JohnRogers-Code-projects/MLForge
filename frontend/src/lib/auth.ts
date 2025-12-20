/**
 * NextAuth.js configuration for ModelForge.
 *
 * Uses GitHub OAuth for authentication.
 * Add GITHUB_ID and GITHUB_SECRET to your environment.
 */

import type { NextAuthOptions } from "next-auth";
import GithubProvider from "next-auth/providers/github";

function getGithubCredentials() {
  const clientId = process.env.GITHUB_ID;
  const clientSecret = process.env.GITHUB_SECRET;

  if (!clientId || !clientSecret) {
    // During build, return placeholder values
    // NextAuth will fail at runtime if these are not set
    if (process.env.NODE_ENV === "production" && !process.env.NEXTAUTH_SECRET) {
      console.warn(
        "Warning: GitHub OAuth credentials not configured. " +
          "Set GITHUB_ID and GITHUB_SECRET in your environment."
      );
    }
    return {
      clientId: clientId || "placeholder",
      clientSecret: clientSecret || "placeholder",
    };
  }

  return { clientId, clientSecret };
}

const { clientId, clientSecret } = getGithubCredentials();

export const authOptions: NextAuthOptions = {
  providers: [
    GithubProvider({
      clientId,
      clientSecret,
    }),
  ],
  pages: {
    signIn: "/login",
    error: "/login",
  },
  callbacks: {
    async session({ session, token }) {
      // Add user ID to session for API calls
      if (session.user && token.sub) {
        session.user.id = token.sub;
      }
      return session;
    },
    async jwt({ token, account }) {
      // Persist OAuth access token if needed
      if (account?.access_token) {
        token.accessToken = account.access_token;
      }
      return token;
    },
  },
  session: {
    strategy: "jwt",
  },
  secret: process.env.NEXTAUTH_SECRET,
};
