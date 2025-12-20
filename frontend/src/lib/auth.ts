/**
 * NextAuth.js configuration for ModelForge.
 *
 * Uses GitHub OAuth for authentication.
 * Add GITHUB_ID and GITHUB_SECRET to your environment.
 */

import type { NextAuthOptions } from "next-auth";
import GithubProvider from "next-auth/providers/github";

const githubClientId = process.env.GITHUB_ID;
const githubClientSecret = process.env.GITHUB_SECRET;

if (!githubClientId) {
  throw new Error(
    "Missing environment variable GITHUB_ID for GitHub OAuth. " +
      "Set GITHUB_ID in your environment (e.g. .env.local) to configure authentication."
  );
}

if (!githubClientSecret) {
  throw new Error(
    "Missing environment variable GITHUB_SECRET for GitHub OAuth. " +
      "Set GITHUB_SECRET in your environment (e.g. .env.local) to configure authentication."
  );
}

export const authOptions: NextAuthOptions = {
  providers: [
    GithubProvider({
      clientId: githubClientId,
      clientSecret: githubClientSecret,
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
