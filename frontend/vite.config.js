import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const apiTarget = process.env.VITE_API_PROXY_TARGET || 'http://127.0.0.1:8000';

const proxyConfig = {
  target: apiTarget,
  changeOrigin: true,
  secure: false,
};

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/auth': proxyConfig,
      '/train': proxyConfig,
      '/jobs': proxyConfig,
      '/models': proxyConfig,
      '/recommend': proxyConfig,
      '/similar': proxyConfig,
      '/explain': proxyConfig,
      '/health': proxyConfig,
      '/session': proxyConfig,
      '/algorithms': proxyConfig,
      '/smart-db-csv': proxyConfig,
    },
  },
  build: {
    outDir: 'dist',
  },
});