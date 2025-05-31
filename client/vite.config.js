import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // authentication & user
      '/auth': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/users': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },

      // datasets list & detail (includes /datasets and /datasets/:id)
      '/datasets': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },

      // model runner API
      '/models': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },

      // CSV upload
      '/upload-csv': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },

      // cleaning
      '/clean': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },

      // correlation heatmap
      '/correlation': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})