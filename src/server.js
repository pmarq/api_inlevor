import dotenv from "dotenv";
import Fastify from "fastify";
import multipart from "@fastify/multipart";
import { QdrantClient } from "@qdrant/js-client-rest";
import { createRequire } from "node:module";

if (process.env.ENV_FILE) {
  const result = dotenv.config({ path: process.env.ENV_FILE });
  if (result.error) {
    throw new Error(
      `Failed to load ENV_FILE ${process.env.ENV_FILE}: ${result.error.message}`,
    );
  }
  console.log(`Loaded env from ${process.env.ENV_FILE}`);
} else {
  dotenv.config();
  console.log("Loaded env from default .env");
}

const fastify = Fastify({ logger: true });

const PORT = Number(process.env.PORT || 3000);
const QDRANT_URL = process.env.QDRANT_URL;
const QDRANT_API_KEY = process.env.QDRANT_API_KEY;
const COLLECTION_NAME = process.env.QDRANT_COLLECTION || "knowledge_chunks";
const VECTOR_SIZE = Number(process.env.VECTOR_SIZE || 1536);
const EMBEDDINGS_ENDPOINT =
  process.env.EMBEDDINGS_ENDPOINT || "https://api.openai.com/v1/embeddings";
const EMBEDDINGS_MODEL =
  process.env.EMBEDDINGS_MODEL || "text-embedding-3-small";
const EMBEDDINGS_API_KEY =
  process.env.EMBEDDINGS_API_KEY || process.env.OPENAI_API_KEY;
const PDF_EXTRACT_API_KEY = process.env.PDF_EXTRACT_API_KEY;

if (!QDRANT_URL) {
  throw new Error("Missing required env var: QDRANT_URL");
}

const qdrant = new QdrantClient({
  url: QDRANT_URL,
  apiKey: QDRANT_API_KEY,
});

const require = createRequire(import.meta.url);

const chunkArray = (items, size) => {
  const chunks = [];
  for (let i = 0; i < items.length; i += size) {
    chunks.push(items.slice(i, i + size));
  }
  return chunks;
};

const toNumber = (value, fallback) => {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
};

const normalizeString = (value) =>
  typeof value === "string" ? value.trim() : "";

const ensureEmbeddingsApiKey = () => {
  if (!EMBEDDINGS_API_KEY) {
    throw new Error("Missing EMBEDDINGS_API_KEY (or OPENAI_API_KEY)");
  }
};

const ensureCanvasPolyfill = async () => {
  if (globalThis.DOMMatrix && globalThis.ImageData && globalThis.Path2D) {
    return;
  }
  try {
    const canvas = await import("@napi-rs/canvas");
    if (!globalThis.DOMMatrix) globalThis.DOMMatrix = canvas.DOMMatrix;
    if (!globalThis.ImageData) globalThis.ImageData = canvas.ImageData;
    if (!globalThis.Path2D) globalThis.Path2D = canvas.Path2D;
  } catch (err) {
    fastify.log.warn(
      { err },
      "Canvas polyfill not available; PDF extraction may fail",
    );
  }
};

const loadPdfParse = async () => {
  const mod = require("pdf-parse");
  return mod?.default ?? mod;
};

const isPageTextPoor = (text) => {
  const t = (text || "").trim();
  const charCount = t.length;
  const words = t.split(/\s+/).filter(Boolean);
  const wordCount = words.length;

  const alphaCount = (t.match(/[A-Za-z\u00C0-\u00FF]/g) || []).length;
  const alphaRatio = charCount ? alphaCount / charCount : 0;

  const uniq = new Set(words.map((w) => w.toLowerCase()));
  const uniqueWordRatio = wordCount ? uniq.size / wordCount : 0;

  if (charCount < 200) return true;
  if (wordCount < 40) return true;
  if (alphaRatio < 0.22) return true;
  if (uniqueWordRatio < 0.25 && wordCount > 80) return true;

  return false;
};

const extractTextByPage = async (data) => {
  await ensureCanvasPolyfill();
  const pdfjs = await import("pdfjs-dist/legacy/build/pdf.mjs");
  const uint8Array = Buffer.isBuffer(data) ? new Uint8Array(data) : data;
  const loadingTask = pdfjs.getDocument({
    data: uint8Array,
    disableWorker: true,
  });
  const pdf = await loadingTask.promise;

  const pages = [];
  for (let p = 1; p <= pdf.numPages; p += 1) {
    const page = await pdf.getPage(p);
    const content = await page.getTextContent();
    const text = content.items
      .map((item) => (typeof item.str === "string" ? item.str : ""))
      .join(" ")
      .replace(/\s+/g, " ")
      .trim();
    const wordCount = text ? text.split(/\s+/).filter(Boolean).length : 0;
    pages.push({
      page: p,
      text,
      charCount: text.length,
      wordCount,
    });
  }

  await pdf.destroy();
  return pages;
};

const buildTextFromPages = (pages) => {
  const poorPages = [];
  const goodPages = [];

  const scored = pages.map((page) => {
    const poor = isPageTextPoor(page.text);
    if (poor) poorPages.push(page.page);
    else goodPages.push(page.page);
    return { ...page, poor };
  });

  const goodText = scored
    .filter((page) => !page.poor)
    .map((page) => page.text)
    .filter(Boolean)
    .join("\n\n");

  const fullText = scored
    .map((page) => page.text)
    .filter(Boolean)
    .join("\n\n");

  const chosenText = goodText.length >= 500 ? goodText : fullText;

  const metrics = {
    totalPages: pages.length,
    poorPages,
    goodPages,
    chosenChars: chosenText.length,
    fullChars: fullText.length,
    goodChars: goodText.length,
  };

  return { chosenText, metrics };
};

async function embedTexts(texts) {
  ensureEmbeddingsApiKey();

  const res = await fetch(EMBEDDINGS_ENDPOINT, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${EMBEDDINGS_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: EMBEDDINGS_MODEL,
      input: texts,
    }),
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Embeddings request failed: ${res.status} ${body}`);
  }

  const payload = await res.json();
  const vectors = payload?.data?.map((item) => item?.embedding) || [];

  if (!vectors.length) {
    throw new Error("Embeddings response is empty");
  }

  return vectors;
}

fastify.register(multipart, {
  limits: {
    fileSize: 25 * 1024 * 1024,
  },
});

fastify.get("/health", async () => {
  return { status: "ok", service: "api_inlevor" };
});

fastify.get("/qdrant/health", async (req, reply) => {
  try {
    const collections = await qdrant.getCollections();
    return {
      ok: true,
      qdrant: "reachable",
      collections: collections?.collections?.map((c) => c.name) || [],
    };
  } catch (err) {
    req.log.error({ err }, "Qdrant health check failed");
    return reply.code(503).send({
      ok: false,
      error: "Qdrant unavailable",
    });
  }
});

async function ensureCollection() {
  const collectionsResponse = await qdrant.getCollections();
  const collections = collectionsResponse?.collections ?? [];
  const exists = collections.some((c) => c.name === COLLECTION_NAME);

  if (!exists) {
    await qdrant.createCollection(COLLECTION_NAME, {
      vectors: {
        size: VECTOR_SIZE,
        distance: "Cosine",
      },
    });
    fastify.log.info(`Collection created: ${COLLECTION_NAME}`);
  } else {
    fastify.log.info(`Collection already exists: ${COLLECTION_NAME}`);
  }
}

fastify.post("/ai/retrieve", async (req, reply) => {
  const body =
    typeof req.body === "object" && req.body !== null ? req.body : {};
  const query = normalizeString(body.query);
  const sourceProject = normalizeString(body.sourceProject);
  const projectId = normalizeString(body.projectId);
  const limit = Math.min(Math.max(toNumber(body.limit, 5), 1), 20);
  const scoreThreshold = Number.isFinite(Number(body.scoreThreshold))
    ? Number(body.scoreThreshold)
    : undefined;

  let vector = Array.isArray(body.vector) ? body.vector : null;

  if (!vector && !query) {
    return reply
      .code(400)
      .send({ error: "query or vector is required" });
  }

  if (!vector) {
    const vectors = await embedTexts([query]);
    vector = vectors[0];
  }

  if (!Array.isArray(vector) || vector.length !== VECTOR_SIZE) {
    return reply.code(400).send({
      error: `vector size mismatch (expected ${VECTOR_SIZE})`,
    });
  }

  const must = [];
  if (sourceProject) {
    must.push({ key: "sourceProject", match: { value: sourceProject } });
  }
  if (projectId) {
    must.push({ key: "projectId", match: { value: projectId } });
  }

  const filter = must.length ? { must } : undefined;

  const results = await qdrant.search(COLLECTION_NAME, {
    vector,
    limit,
    filter,
    score_threshold: scoreThreshold,
    with_payload: true,
    with_vector: false,
  });

  return {
    ok: true,
    count: results.length,
    results,
  };
});

fastify.post("/ai/ingest", async (req, reply) => {
  const body =
    typeof req.body === "object" && req.body !== null ? req.body : {};
  const projectId = normalizeString(body.projectId);
  const sourceProject = normalizeString(body.sourceProject);
  const source = normalizeString(body.source || "raw");
  const rawChunks = Array.isArray(body.chunks) ? body.chunks : [];

  if (!projectId || !sourceProject || rawChunks.length === 0) {
    return reply.code(400).send({
      error: "projectId, sourceProject and chunks are required",
    });
  }

  const chunks = rawChunks
    .map((item, idx) => {
      const text = normalizeString(item?.text);
      const vector = Array.isArray(item?.vector) ? item.vector : null;
      const index = Number.isFinite(Number(item?.index))
        ? Number(item.index)
        : idx;
      const id =
        normalizeString(item?.id) || `${sourceProject}:${projectId}:${index}`;
      return { id, text, vector, index };
    })
    .filter((item) => item.text || item.vector);

  if (!chunks.length) {
    return reply.code(400).send({ error: "chunks are empty" });
  }

  const needsEmbedding = chunks.some((c) => !c.vector);
  let vectors = [];

  if (needsEmbedding) {
    const texts = chunks.map((c) => c.text || "");
    vectors = await embedTexts(texts);
  }

  const points = chunks.map((chunk, idx) => {
    const vector = chunk.vector || vectors[idx];
    if (!Array.isArray(vector) || vector.length !== VECTOR_SIZE) {
      throw new Error(`vector size mismatch for chunk ${chunk.id}`);
    }
    return {
      id: chunk.id,
      vector,
      payload: {
        projectId,
        sourceProject,
        source,
        text: chunk.text,
        index: chunk.index,
      },
    };
  });

  const batches = chunkArray(points, 64);
  for (const batch of batches) {
    await qdrant.upsert(COLLECTION_NAME, { points: batch });
  }

  return {
    ok: true,
    inserted: points.length,
  };
});

fastify.post("/pdf/extract", async (req, reply) => {
  const apiKeyHeader = req.headers["x-api-key"];
  if (PDF_EXTRACT_API_KEY && apiKeyHeader !== PDF_EXTRACT_API_KEY) {
    return reply.code(401).send({ ok: false, error: "Unauthorized" });
  }

  let fileBuffer = null;

  try {
    const parts = req.parts();
    for await (const part of parts) {
      if (part.type === "file" && part.fieldname === "file") {
        fileBuffer = await part.toBuffer();
      }
    }
  } catch (err) {
    req.log.error({ err }, "Failed to read multipart payload");
    return reply.code(400).send({ ok: false, error: "Payload inválido." });
  }

  if (!fileBuffer) {
    return reply.code(400).send({ ok: false, error: "PDF não enviado." });
  }

  try {
    const pages = await extractTextByPage(fileBuffer);
    const { chosenText, metrics } = buildTextFromPages(pages);
    return {
      ok: true,
      rawText: chosenText,
      pages,
      pageMetrics: metrics,
    };
  } catch (error) {
    req.log.warn(
      { error },
      "pdfjs failed, fallback pdf-parse on /pdf/extract",
    );
    const pdfParse = await loadPdfParse();
    const parser = new pdfParse.PDFParse({ data: fileBuffer });
    const parsed = await parser.getText();
    await parser.destroy?.();
    const rawText = parsed.text || "";
    return {
      ok: true,
      rawText,
      pageMetrics: { fallback: "pdf-parse", rawChars: rawText.length },
    };
  }
});

async function start() {
  try {
    await ensureCollection();
    await fastify.listen({ port: PORT, host: "0.0.0.0" });
    fastify.log.info(`API Inlevor listening on port ${PORT}`);
  } catch (err) {
    fastify.log.error({ err }, "Failed to start server");
    process.exit(1);
  }
}

start();
