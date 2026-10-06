import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  server: { host: true, port: 5173, strictPort: true },
  preview: { host: true, port: 4173, strictPort: true },
  build: { target: 'esnext', chunkSizeWarningLimit: 4000 },
  assetsInclude: ['**/*.hdr', '**/*.ktx2', '**/*.png', '**/*.jpg', '**/*.webp'],
});
