# Frontend Dockerfile for Next.js
# Multi-stage build for optimized production image
# NOTE: Build context is project root for Railway deployment

# Stage 1: Dependencies
FROM node:20-alpine AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci

# Stage 2: Build
FROM node:20-alpine AS builder
WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY frontend/. .

# Set build-time environment variables
# NOTE: NEXT_PUBLIC_API_URL must be set as a build-time variable in Railway
# service settings, not just as a runtime variable
ARG NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}

# Ensure public directory exists (Next.js doesn't require it)
RUN mkdir -p public

# Build the Next.js application
RUN npm run build

# Stage 3: Production
FROM node:20-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production

# Create non-root user for security
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Copy built assets
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

# Set correct ownership
RUN chown -R nextjs:nodejs /app

USER nextjs

EXPOSE 3000

# Railway sets PORT dynamically
ENV HOSTNAME="0.0.0.0"

# Use shell form to expand $PORT at runtime
CMD node server.js
