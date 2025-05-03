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

      // CSV upload endpoint
      '/upload-csv': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },

      // if DataCleaning.jsx calls something like `/clean`:
      '/clean': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },

      // if CorrelationHeatmap.jsx calls `/correlation`:
      '/correlation': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },

      // add any other API prefixes your new components use…
    },
  },
})
