import { defineConfig, loadEnv } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const apiTarget = env.VITE_API_BASE_URL || 'https://cv-parsing-api-iftlfnhyta-ew.a.run.app'

  return {
    plugins: [svelte()],
    server: {
      proxy: {
        // In dev, browser calls /api/* (same origin = no CORS). We forward to deployed Cloud Run.
        '/api': {
          target: apiTarget,
          changeOrigin: true,
          secure: true,
          rewrite: (path) => path.replace(/^\/api/, ''),
          configure: (proxy) => {
            proxy.on('proxyReq', (proxyReq, req) => {
              const auth = req.headers.authorization
              if (auth) proxyReq.setHeader('Authorization', auth)
            })
          },
        },
      },
    },
  }
})
