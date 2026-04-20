const http = require('http');
const fs = require('fs');
const path = require('path');

// Load .env manually (no extra packages needed)
try {
  const env = fs.readFileSync(path.join(__dirname, '.env'), 'utf8');
  env.split('\n').forEach(line => {
    const [key, ...val] = line.split('=');
    if (key && val.length) process.env[key.trim()] = val.join('=').trim();
  });
} catch (_) {
  // .env not found — rely on system environment variables (e.g. on Render/Railway)
}

const PORT = process.env.PORT || 3000;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

if (!OPENAI_API_KEY) {
  console.error('❌  OPENAI_API_KEY is not set. Add it to your .env file or environment variables.');
  process.exit(1);
}

const server = http.createServer(async (req, res) => {

  // ── Proxy: POST /api/ai-report → OpenAI ──
  if (req.method === 'POST' && req.url === '/api/ai-report') {
    let body = '';
    req.on('data', chunk => (body += chunk));
    req.on('end', async () => {
      try {
        // Node 18+ has built-in fetch; for older Node use this polyfill-free approach
        const https = require('https');
        const payload = Buffer.from(body);

        const options = {
          hostname: 'api.openai.com',
          path: '/v1/chat/completions',
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${OPENAI_API_KEY}`,
            'Content-Length': payload.length
          }
        };

        const proxyReq = https.request(options, (proxyRes) => {
          let data = '';
          proxyRes.on('data', chunk => (data += chunk));
          proxyRes.on('end', () => {
            res.writeHead(proxyRes.statusCode, { 'Content-Type': 'application/json' });
            res.end(data);
          });
        });

        proxyReq.on('error', (err) => {
          console.error('[Proxy Error]', err.message);
          res.writeHead(502, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: { message: 'Failed to reach OpenAI: ' + err.message } }));
        });

        proxyReq.write(payload);
        proxyReq.end();

      } catch (err) {
        console.error('[Server Error]', err.message);
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: { message: err.message } }));
      }
    });
    return;
  }

  // ── Serve index.html for all other GET requests ──
  if (req.method === 'GET') {
    const filePath = path.join(__dirname, 'index.html');
    fs.readFile(filePath, (err, data) => {
      if (err) {
        res.writeHead(404);
        res.end('index.html not found');
        return;
      }
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end(data);
    });
    return;
  }

  res.writeHead(405);
  res.end('Method Not Allowed');
});

server.listen(PORT, () => {
  console.log(`✅  NuMind MAPS running at http://localhost:${PORT}`);
});
