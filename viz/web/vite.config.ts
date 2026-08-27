import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API is a separate process holding the GPU; in dev the Vite server proxies to it so the
// browser sees one origin, and in production FastAPI serves this build directly from viz/web/dist.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8130" } },
  build: { outDir: "dist", emptyOutDir: true },
});
