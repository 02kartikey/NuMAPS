const http    = require('http');
const https   = require('https');
const fs      = require('fs');
const path    = require('path');
const cluster = require('cluster');
const os      = require('os');

// ── Load .env (no extra packages needed) ──────────────
try {
  fs.readFileSync(path.join(__dirname, '.env'), 'utf8')
    .split('\n').forEach(line => {
      const [key, ...val] = line.split('=');
      if (key && val.length) process.env[key.trim()] = val.join('=').trim();
    });
} catch (_) {
  // .env not found — rely on system env vars (Render / Railway / etc.)
}

const PORT           = process.env.PORT           || 3000;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
// Max simultaneous OpenAI proxy requests *per worker*.
// With 4 cores → up to 4 × 12 = 48 in-flight streams.
// Keep below your OpenAI RPM tier limit.
const MAX_CONCURRENT = parseInt(process.env.MAX_CONCURRENT || '12',    10);
// Hard timeout for the full round-trip (ms). GPT-4o streams can run ~60 s.
const REQ_TIMEOUT_MS = parseInt(process.env.REQ_TIMEOUT_MS || '90000', 10);

if (!OPENAI_API_KEY) {
  console.error('❌  OPENAI_API_KEY is not set. Add it to your .env file or environment variables.');
  process.exit(1);
}

// ── Cluster: one worker per CPU core ──────────────────
// Primary process only forks — all real work happens in workers.
if (cluster.isPrimary) {
  const numCPUs = os.cpus().length;
  console.log(`✅  Primary ${process.pid} — spawning ${numCPUs} worker(s) on :${PORT}`);
  for (let i = 0; i < numCPUs; i++) cluster.fork();
  cluster.on('exit', (worker, code) => {
    console.warn(`⚠️  Worker ${worker.process.pid} exited (code ${code}) — restarting`);
    cluster.fork(); // auto-restart so one crash doesn't kill the app
  });
  return; // primary does nothing else
}

// ════════════════════════════════════════════════════
//  Session-primary, IP-fallback sliding-window rate limiter
//
//  KEY STRATEGY
//  ┌──────────────────────────────────────────────────┐
//  │ 1. X-Session-ID header present (sent by app.js) │
//  │    → rate-limit by session ID                   │
//  │    One student = one session = one limit bucket │
//  │                                                  │
//  │ 2. Header absent (direct API calls, curl, etc.) │
//  │    → fall back to IP address                    │
//  │    Preserves abuse protection for callers that  │
//  │    don't send the header                        │
//  └──────────────────────────────────────────────────┘
//
//  WHY SESSION > IP
//  A school's entire student body shares one NAT IP.
//  With IP-only limiting, the 6th student to click
//  "Generate Report" in a 60-second window would be
//  hard-blocked even though each student is distinct.
//  Session IDs (generated per registration in app.js,
//  format: NMSUITE-<timestamp>-<random>) are globally
//  unique so each student gets their own quota.
//
//  LIMITS (tunable via env vars)
//    RATE_WINDOW_MS  — rolling window length        (default: 60 000 ms)
//    RATE_MAX_REQS   — max requests per key/window  (default: 5)
//    SESSION_ID_MAX_LEN — max header length accepted (default: 64 chars)
//      Capped to prevent oversized keys being used as
//      a memory-exhaustion vector.
// ════════════════════════════════════════════════════
const RATE_WINDOW_MS    = parseInt(process.env.RATE_WINDOW_MS    || '60000', 10);
const RATE_MAX_REQS     = parseInt(process.env.RATE_MAX_REQS     || '5',     10);
const SESSION_ID_MAX_LEN = parseInt(process.env.SESSION_ID_MAX_LEN || '64',   10);

// Map<key: string, timestamps: number[]>
// Keys are either session IDs or IP addresses.
const _rateLimitMap = new Map();

// Prune stale entries every 5 minutes so the Map never grows unbounded.
setInterval(() => {
  const now = Date.now();
  for (const [key, ts] of _rateLimitMap) {
    const fresh = ts.filter(t => now - t < RATE_WINDOW_MS);
    if (fresh.length === 0) _rateLimitMap.delete(key);
    else                    _rateLimitMap.set(key, fresh);
  }
}, 5 * 60 * 1000);

/**
 * Derive the rate-limit key for an incoming request.
 * Prefers X-Session-ID; falls back to the client IP.
 * Returns { key, type } where type is 'session' | 'ip'.
 */
function _rateLimitKey(req) {
  const raw = (req.headers['x-session-id'] || '').trim();
  // Accept the header only if it looks like a plausible session ID:
  // non-empty, within the length cap, and containing only safe chars
  // (alphanumerics + hyphens). Rejects injected or malformed values.
  if (raw && raw.length <= SESSION_ID_MAX_LEN && /^[A-Za-z0-9\-]+$/.test(raw)) {
    return { key: 'sid:' + raw, type: 'session' };
  }
  const ip = (req.headers['x-forwarded-for'] || '').split(',')[0].trim()
           || req.socket.remoteAddress
           || 'unknown';
  return { key: 'ip:' + ip, type: 'ip' };
}

/**
 * Check + record a request for the derived key.
 * Returns { allowed: true } or { allowed: false, retryAfter: <seconds> }.
 */
function checkRateLimit(req) {
  const { key, type } = _rateLimitKey(req);
  const now  = Date.now();
  const ts   = (_rateLimitMap.get(key) || []).filter(t => now - t < RATE_WINDOW_MS);
  if (ts.length >= RATE_MAX_REQS) {
    const retryAfter = Math.ceil((ts[0] + RATE_WINDOW_MS - now) / 1000);
    return { allowed: false, retryAfter: Math.max(retryAfter, 1), key, type };
  }
  ts.push(now);
  _rateLimitMap.set(key, ts);
  return { allowed: true, key, type };
}

// ── Worker: reusable HTTPS agent (keep-alive to OpenAI) ──
const openaiAgent = new https.Agent({
  keepAlive:      true,
  maxSockets:     MAX_CONCURRENT, // don't open more sockets than we need
  maxFreeSockets: 4,
  timeout:        REQ_TIMEOUT_MS,
});

// ════════════════════════════════════════════════════
//  AI Report Response Cache
//  Key   = SHA-256 of the score payload (student name
//          and personal fields are stripped before hashing
//          so identical score profiles share one cache entry).
//  Value = { body: Buffer, ts: number }
//  TTL   = CACHE_TTL_MS (default: 24 hours).
//          Reports are deterministic enough at temperature 0.65
//          that repeating within a session adds no value, and
//          caching across students with identical profiles saves
//          quota without meaningfully reducing quality.
//  Size  = capped at CACHE_MAX_ENTRIES LRU-style (oldest evicted).
// ════════════════════════════════════════════════════
const crypto = require('crypto');
const CACHE_TTL_MS     = parseInt(process.env.CACHE_TTL_MS     || String(24 * 60 * 60 * 1000), 10);
const CACHE_MAX_ENTRIES = parseInt(process.env.CACHE_MAX_ENTRIES || '500', 10);

// Map<cacheKey: string, { body: Buffer, ts: number }>
const _reportCache = new Map();

/** Build a stable cache key from a report request payload.
 *  Strips student identity fields so two students with identical
 *  scores share one cached response.
 */
function _cacheKey(rawPayload) {
  try {
    const parsed = JSON.parse(rawPayload.toString());
    const msgs   = parsed.messages || [];
    // The user message contains all the score data.
    // Redact the name/school/gender/age line so identity doesn't
    // affect cache matching — only scores matter.
    const scored = msgs.map(m => {
      if (m.role !== 'user' || typeof m.content !== 'string') return m;
      // Replace "STUDENT: <name>, Class X, <gender>, Age Y, <school>"
      // with a normalised placeholder so only the score lines remain.
      const normalised = m.content.replace(/STUDENT:.*?\n/, 'STUDENT: [REDACTED]\n');
      return { role: m.role, content: normalised };
    });
    return crypto
      .createHash('sha256')
      .update(JSON.stringify({ model: parsed.model, msgs: scored }))
      .digest('hex');
  } catch (_) {
    return null; // unparseable payload → skip cache
  }
}

/**
 * Extract the student's first name and full name from the request payload
 * so the server can anonymise cached AI text before storing.
 *
 * The prompt line is: "STUDENT: <fullName>, Class X, <gender>, Age Y, <school>"
 * and the writing rules line is: "- Use <firstName>'s name naturally throughout."
 * Both are parsed here so the anonymiser can replace every occurrence.
 *
 * Returns { firstName, fullName } or null if unparseable.
 */
function _extractNamesFromPayload(rawPayload) {
  try {
    const parsed = JSON.parse(rawPayload.toString());
    const msgs   = parsed.messages || [];
    for (const m of msgs) {
      if (m.role !== 'user' || typeof m.content !== 'string') continue;

      // Match: "STUDENT: Kartikey Sharma, Class 11 A, Male, Age 17, DPS"
      const studentMatch = m.content.match(/^STUDENT:\s*(.+?),\s*Class\s/m);
      if (!studentMatch) continue;
      const fullName  = studentMatch[1].trim();

      // Match: "- Use Kartikey's name naturally throughout."
      const firstMatch = m.content.match(/Use\s+(\S+?)'s name naturally throughout/);
      const firstName  = firstMatch ? firstMatch[1].trim() : fullName.split(' ')[0];

      return { firstName, fullName };
    }
    return null;
  } catch (_) {
    return null;
  }
}

/**
 * Anonymise all text fields in a parsed AI report JSON body by replacing
 * every occurrence of the student's real name with neutral placeholders.
 *
 * Placeholders used:
 *   __FIRST_NAME__  →  replaced with firstName at render time on the client
 *   __FULL_NAME__   →  replaced with fullName  at render time on the client
 *
 * Replacement is case-sensitive and whole-word-aware (uses \b boundaries)
 * so "Kartikey" in "Kartikey's" is replaced correctly.
 * fullName is replaced before firstName to avoid partial replacement
 * (e.g. "Kartikey Sharma" → "__FULL_NAME__" rather than "__FIRST_NAME__ Sharma").
 *
 * Text fields anonymised: all string values in the top-level report object,
 * and the "rationale" field inside each career_table entry.
 * Non-text fields (numbers, arrays) are left untouched.
 */
function _anonymiseBody(jsonText, firstName, fullName) {
  if (!firstName && !fullName) return jsonText;
  // Escape special regex characters in names (e.g. hyphens in compound names).
  function escRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }
  // \b word boundaries prevent partial-word replacement:
  // "Amit" must not corrupt "Amitabh", "Jay" must not corrupt "Jayant", etc.
  // fullName replaced first (longer string) so "Kartikey Sharma" becomes
  // __FULL_NAME__ rather than "__FIRST_NAME__ Sharma".
  let out = jsonText;
  if (fullName)  out = out.replace(new RegExp('\\b' + escRe(fullName)  + '\\b', 'g'), '__FULL_NAME__');
  if (firstName) out = out.replace(new RegExp('\\b' + escRe(firstName) + '\\b', 'g'), '__FIRST_NAME__');
  return out;
}

/** Retrieve a cached report body, or null if missing/expired. */
function _cacheGet(key) {
  if (!key) return null;
  const entry = _reportCache.get(key);
  if (!entry) return null;
  if (Date.now() - entry.ts > CACHE_TTL_MS) { _reportCache.delete(key); return null; }
  return entry.body;
}

/** Store a completed (non-streaming) report body in the cache. */
function _cacheSet(key, body) {
  if (!key) return;
  // LRU eviction: delete oldest entry when at capacity.
  if (_reportCache.size >= CACHE_MAX_ENTRIES) {
    _reportCache.delete(_reportCache.keys().next().value);
  }
  _reportCache.set(key, { body, ts: Date.now() });
}

// Prune expired entries every hour so the Map doesn't bloat.
setInterval(() => {
  const now = Date.now();
  for (const [k, v] of _reportCache) {
    if (now - v.ts > CACHE_TTL_MS) _reportCache.delete(k);
  }
}, 60 * 60 * 1000);

// Per-worker concurrency counter + overflow queue
let activeRequests = 0;

// Maximum number of requests to hold in queue before hard-rejecting.
// Tune via MAX_QUEUE_SIZE env var (default: 50 per worker).
const MAX_QUEUE_SIZE = parseInt(process.env.MAX_QUEUE_SIZE || '50', 10);

// Each entry: { run: Function, cancelled: Boolean }
// cancelled=true means the client disconnected while waiting — drainQueue skips it
// rather than firing a wasted OpenAI API slot.
const requestQueue = [];

// Try to drain the next queued request if a slot is free.
// Skips entries that were cancelled due to client disconnect.
function drainQueue() {
  while (requestQueue.length > 0 && activeRequests < MAX_CONCURRENT) {
    const next = requestQueue.shift();
    if (!next.cancelled) {
      next.run();
      return; // one slot consumed — stop here
    }
    // Entry was cancelled (client disconnected) — skip silently and loop
  }
}

// ── Static-file cache (in-memory, workers hold small files) ──
const _fileCache = new Map();
function serveStatic(filePath, res) {
  if (_fileCache.has(filePath)) {
    const { data, ct, etag } = _fileCache.get(filePath);
    res.writeHead(200, { 'Content-Type': ct, 'ETag': etag, 'Cache-Control': 'public, max-age=3600' });
    res.end(data);
    return;
  }
  fs.readFile(filePath, (err, data) => {
    if (err) { res.writeHead(404); res.end(path.basename(filePath) + ' not found'); return; }
    const MIME = { '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8', '.js': 'text/javascript; charset=utf-8' };
    const ct   = MIME[path.extname(filePath)] || 'text/plain';
    const etag = '"' + data.length + '-' + Date.now() + '"';
    // Cache JS + CSS; never cache HTML so deploys take effect immediately
    if (path.extname(filePath) !== '.html') _fileCache.set(filePath, { data, ct, etag });
    res.writeHead(200, {
      'Content-Type':  ct,
      'Cache-Control': path.extname(filePath) === '.html' ? 'no-cache' : 'public, max-age=3600',
    });
    res.end(data);
  });
}

// ════════════════════════════════════════════════════
//  Proxy runner — called directly or from queue drain
//  FIX 1: Defined at module scope, not inside the HTTP
//  handler — so it is not re-created on every request.
// ════════════════════════════════════════════════════
function runProxyRequest(payload, req, res) {
  // Guard: client may have closed while waiting in queue.
  if (res.writableEnded || !req.socket?.readable) { drainQueue(); return; }

  // ── Cache check ──────────────────────────────────────────────────────
  // Non-streaming requests whose score payload matches a cached entry are
  // served instantly without touching OpenAI at all. Streaming requests
  // are not served from cache (the client expects SSE), but the completed
  // body is stored so a later non-streaming request for the same profile
  // benefits. Cache hit/miss is logged so you can tune TTL and size.
  const cacheKey = _cacheKey(payload);
  let   isStream = false;
  try { isStream = !!JSON.parse(payload.toString()).stream; } catch (_) {}

  // Cache check fires for ALL requests (streaming or not).
  // A cache hit is a complete JSON body — no need to stream it.
  // The client detects X-Cache: HIT and reads it as plain JSON,
  // so streaming callers also benefit from cached entries.
  if (cacheKey) {
    const cached = _cacheGet(cacheKey);
    if (cached) {
      console.log(`[Cache] HIT  key=${cacheKey.slice(0, 12)}... size=${_reportCache.size} stream=${isStream}`);
      res.writeHead(200, {
        'Content-Type':  'application/json',
        'Cache-Control': 'no-cache',
        'X-Cache':       'HIT',
      });
      res.end(cached);
      // No activeRequests slot consumed — drainQueue immediately.
      drainQueue();
      return;
    }
    console.log(`[Cache] MISS key=${cacheKey.slice(0, 12)}... size=${_reportCache.size}`);
  }
  // ────────────────────────────────────────────────────────────────────

  activeRequests++;
  const releaseSlot = (() => {
    let released = false;
    return () => { if (!released) { released = true; activeRequests--; drainQueue(); } };
  })();
  res.on('finish', releaseSlot);
  res.on('close',  releaseSlot);

  const options = {
    hostname: 'api.openai.com',
    path:     '/v1/chat/completions',
    method:   'POST',
    agent:    openaiAgent,
    timeout:  REQ_TIMEOUT_MS,
    headers: {
      'Content-Type':   'application/json',
      'Authorization':  `Bearer ${OPENAI_API_KEY}`,
      'Content-Length': payload.length,
      'Connection':     'keep-alive',
    },
  };

  const proxyReq = https.request(options, (proxyRes) => {
    const status = proxyRes.statusCode;

    const forwardHeaders = {
      'Content-Type':      isStream ? 'text/event-stream' : 'application/json',
      'Cache-Control':     'no-cache',
      'X-Accel-Buffering': 'no',
      'Connection':        'keep-alive',
      'X-Cache':           'MISS',
    };
    [
      'retry-after',
      'x-ratelimit-limit-requests',  'x-ratelimit-remaining-requests',
      'x-ratelimit-limit-tokens',    'x-ratelimit-remaining-tokens',
      'x-ratelimit-reset-requests',  'x-ratelimit-reset-tokens',
    ].forEach(k => { if (proxyRes.headers[k]) forwardHeaders[k] = proxyRes.headers[k]; });

    res.writeHead(status, forwardHeaders);

    if (isStream) {
      // ── Streaming: pipe directly; also accumulate for cache ──
      // We buffer the full SSE body in memory so we can cache the
      // JSON once streaming completes. This costs ~6–8 KB of RAM
      // per in-flight stream — negligible vs. the API cost saved.
      const sseChunks = [];
      proxyRes.on('data', chunk => {
        if (!res.writableEnded) res.write(chunk);
        sseChunks.push(chunk);
      });
      proxyRes.on('end', () => {
        if (!res.writableEnded) res.end();
        // Only cache 200 responses; errors are not worth caching.
        if (status === 200 && cacheKey) {
          // Extract the concatenated JSON content from the SSE stream
          // so future non-streaming calls for the same profile hit cache.
          try {
            const raw = Buffer.concat(sseChunks).toString('utf8');
            let accumulated = '';
            for (const line of raw.split('\n')) {
              if (!line.startsWith('data: ')) continue;
              const sseData = line.slice(6).trim();
              if (sseData === '[DONE]') break;
              try {
                const parsed = JSON.parse(sseData);
                accumulated += parsed?.choices?.[0]?.delta?.content || '';
              } catch (_) {}
            }
            if (accumulated) {
              // Store as a minimal non-streaming response so the cache
              // can be replayed for non-streaming callers.
              // Anonymise the name before storing — placeholder will be
              // re-personalised on the client for every student served.
              const names       = _extractNamesFromPayload(payload);
              const anonText    = _anonymiseBody(accumulated, names && names.firstName, names && names.fullName);
              const syntheticBody = JSON.stringify({
                choices: [{ message: { content: anonText } }],
                _cached: true,
              });
              _cacheSet(cacheKey, Buffer.from(syntheticBody));
              console.log(`[Cache] SET  key=${cacheKey.slice(0, 12)}… (from stream, anonymised: fn=${names && names.firstName})`);
            }
          } catch (_) { /* best-effort — never crash the response */ }
        }
      });
      proxyRes.on('error', err => {
        console.error('[Stream Error]', err.message);
        if (!res.writableEnded) res.end();
      });
      req.on('close', () => { if (!proxyRes.destroyed) proxyRes.destroy(); });
    } else {
      // ── Non-streaming: buffer, cache on success, then flush ──
      const parts = [];
      proxyRes.on('data', c => parts.push(c));
      proxyRes.on('end', () => {
        const body = Buffer.concat(parts);
        if (status === 200 && cacheKey) {
          // Anonymise the AI text before storing — replace the student's real
          // name with __FIRST_NAME__ / __FULL_NAME__ placeholders so cached
          // entries are safe to serve to any student with the same score profile.
          const names       = _extractNamesFromPayload(payload);
          const anonText    = _anonymiseBody(body.toString('utf8'), names && names.firstName, names && names.fullName);
          _cacheSet(cacheKey, Buffer.from(anonText, 'utf8'));
          console.log(`[Cache] SET  key=${cacheKey.slice(0, 12)}… (anonymised: fn=${names && names.firstName})`);
        }
        if (!res.writableEnded) res.end(body);
      });
    }
  });

  proxyReq.on('timeout', () => {
    proxyReq.destroy();
    if (!res.headersSent) res.writeHead(504, { 'Content-Type': 'application/json' });
    if (!res.writableEnded) res.end(JSON.stringify({ error: { message: 'OpenAI did not respond in time.' } }));
  });

  proxyReq.on('error', err => {
    console.error('[Proxy Error]', err.message);
    if (!res.headersSent) res.writeHead(502, { 'Content-Type': 'application/json' });
    if (!res.writableEnded) res.end(JSON.stringify({ error: { message: 'Failed to reach OpenAI: ' + err.message } }));
  });

  proxyReq.write(payload);
  proxyReq.end();
}

// ── HTTP server ────────────────────────────────────────
const server = http.createServer((req, res) => {

  // Hard per-request timeout
  req.setTimeout(REQ_TIMEOUT_MS, () => {
    if (!res.headersSent) res.writeHead(504, { 'Content-Type': 'application/json' });
    if (!res.writableEnded) res.end(JSON.stringify({ error: { message: 'Request timed out.' } }));
  });

  // ════════════════════════════════════════════════════
  //  Proxy: POST /api/ai-report  →  OpenAI
  //  • Concurrency-capped per worker
  //  • Upstream timeout
  //  • SSE streaming piped directly — zero buffering
  //  • Client disconnect aborts upstream (saves quota)
  // ════════════════════════════════════════════════════
  if (req.method === 'POST' && req.url === '/api/ai-report') {

    // ── Session-primary, IP-fallback rate limit ────────────────
    // Keyed by X-Session-ID if present, otherwise by client IP.
    // See the rate-limiter block above for full rationale.
    const rl = checkRateLimit(req);
    if (!rl.allowed) {
      res.writeHead(429, {
        'Content-Type': 'application/json',
        'Retry-After':  String(rl.retryAfter),
      });
      res.end(JSON.stringify({
        error: {
          message: `Too many report requests. Please wait ${rl.retryAfter} second(s) and try again.`,
          retryAfter: rl.retryAfter,
        },
      }));
      console.warn(`[RateLimit] ${rl.type}=${rl.key} blocked — ${RATE_MAX_REQS} req/${RATE_WINDOW_MS}ms window. Retry in ${rl.retryAfter}s`);
      return;
    }
    // ───────────────────────────────────────────────────────────

    // Collect the request body first (before we know if we'll queue or run immediately)
    const chunks = [];
    req.on('data', chunk => chunks.push(chunk));
    req.on('end', () => {
      const payload = Buffer.concat(chunks);

      // Guard: if the client disconnected while we were reading the body, bail.
      if (res.writableEnded) return;

      // Slot available → run immediately; otherwise queue or hard-reject.
      if (activeRequests < MAX_CONCURRENT) {
        runProxyRequest(payload, req, res);
      } else if (requestQueue.length < MAX_QUEUE_SIZE) {
        // FIX 2: Wire up client-disconnect to mark the entry cancelled so
        // drainQueue skips it and frees the slot for another request.
        // FIX 3: X-Queue-Position header removed — it can't be kept accurate
        // as the queue drains and a stale position is misleading to clients.
        const entry = { cancelled: false, run: null };
        entry.run = () => runProxyRequest(payload, req, res);

        req.on('close', () => {
          if (entry.cancelled) return;
          entry.cancelled = true;
          console.log(`[Queue] Client disconnected while queued — slot freed (worker ${process.pid})`);
        });

        requestQueue.push(entry);
        console.log(`[Queue] ${requestQueue.length} request(s) waiting (worker ${process.pid})`);
      } else {
        // Queue full → only now do we hard-reject.
        res.writeHead(429, { 'Content-Type': 'application/json', 'Retry-After': '30' });
        res.end(JSON.stringify({ error: { message: 'Server overloaded — please try again in a moment.' } }));
      }
    });
    return;
  }

  // ════════════════════════════════════════════════════
  //  Static files: index.html · app.js · styles.css
  // ════════════════════════════════════════════════════
  if (req.method === 'GET') {
    const urlPath  = req.url.split('?')[0];
    const ext      = path.extname(urlPath);
    const allowed  = ['.html', '.css', '.js'];
    const filePath = allowed.includes(ext)
      ? path.join(__dirname, urlPath)
      : path.join(__dirname, 'index.html');
    serveStatic(filePath, res);
    return;
  }

  res.writeHead(405);
  res.end('Method Not Allowed');
});

server.listen(PORT, () => {
  console.log(`✅  Worker ${process.pid} listening on :${PORT}`);
});
