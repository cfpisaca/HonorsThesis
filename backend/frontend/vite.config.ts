import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/video_feed': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
        secure: false
      },
      '/recognized_text': {  
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
        secure: false
      },
      '/clear_text': {  
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
        secure: false
      },
      '/set_model_type': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
        secure: false
      }
    }
  }
})
