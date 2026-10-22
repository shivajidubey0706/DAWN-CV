import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The dev server proxies /api to the FastAPI backend, so the browser never needs CORS
// or any secret. Override the target with SGERUP_API_TARGET.
const target = process.env.SGERUP_API_TARGET ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { '/api': { target, changeOrigin: true } },
  },
  preview: {
    port: 4173,
    proxy: { '/api': { target, changeOrigin: true } },
  },
  build: {
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        manualChunks: {
          charts: ['recharts'],
          markdown: ['react-markdown', 'remark-gfm', 'katex'],
          motion: ['framer-motion'],
        },
      },
    },
  },
})
