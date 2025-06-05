// client/vite.config.js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // All backend endpoints now live under /api
      "/api/auth": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/api/users": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/api/datasets": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/api/models": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/api/upload-csv": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/api/clean": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/api/correlation": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      // other /api/... routes
    },
  },
});
