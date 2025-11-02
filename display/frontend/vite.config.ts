import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: '/ausdevs_conversations/',
  server: {
    proxy: {
      '/ausdevs_conversations/api': {
        target: 'http://localhost:9000',
        changeOrigin: true,
      }
    }
  }
})
