# ModelForge Frontend

Next.js 14 dashboard for the ModelForge ML Model Serving Platform.

## Getting Started

### Prerequisites

- Node.js 18+
- npm, yarn, pnpm, or bun

### Installation

```bash
npm install
```

### Environment Variables

Copy `.env.example` to `.env.local` and configure:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the dashboard.

### Build

```bash
npm run build
npm start
```

## Project Structure

```
src/
├── app/                 # Next.js App Router pages
│   ├── layout.tsx       # Root layout with Inter font
│   ├── page.tsx         # Home page
│   └── globals.css      # Global styles with Tailwind
├── lib/
│   ├── api.ts           # Typed API client
│   └── config.ts        # Centralized configuration
└── types/
    └── api.ts           # TypeScript types for API responses
```

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript 5
- **Styling**: Tailwind CSS
- **Font**: Inter (via next/font)

## API Integration

The frontend connects to the FastAPI backend at the URL specified in `NEXT_PUBLIC_API_URL`. All API calls go through the typed client in `src/lib/api.ts`.
