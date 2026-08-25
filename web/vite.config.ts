import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /api to the Python backend so the frontend is served
// from one origin in development and in production alike. That avoids CORS
// entirely -- and CORS on a tool that can start and stop processes is a
// permission surface worth simply not having.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 7316,
    strictPort: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:7315",
        changeOrigin: false,
        // Server-Sent Events must not be buffered by the proxy or telemetry
        // arrives in bursts instead of once a second.
        configure: (proxy) => {
          proxy.on("proxyRes", (proxyRes) => {
            if (proxyRes.headers["content-type"]?.includes("event-stream")) {
              proxyRes.headers["cache-control"] = "no-cache";
            }
          });
        },
      },
    },
  },
  build: {
    // Built assets are served by the Python app, so they land where it looks.
    outDir: "dist",
    emptyOutDir: true,
  },
});
