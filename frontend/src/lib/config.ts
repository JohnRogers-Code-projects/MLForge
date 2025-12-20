/**
 * Application configuration.
 *
 * All configuration values are loaded from environment variables
 * with sensible defaults for development.
 */

export const config = {
  // API configuration
  api: {
    baseUrl: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
    prefix: "/api/v1",
  },

  // App metadata
  app: {
    name: "ModelForge",
    description: "ML Model Serving Platform",
    version: process.env.NEXT_PUBLIC_APP_VERSION || "0.1.0",
  },

  // Feature flags
  features: {
    // Enable/disable features as needed
  },
} as const;

export default config;
