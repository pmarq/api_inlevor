import "dotenv/config";
import Fastify from "fastify";
import { QdrantClient } from "@qdrant/js-client-rest";

const fastify = Fastify({ logger: true });

const PORT = Number(process.env.PORT || 3000);
const QDRANT_URL = process.env.QDRANT_URL;
const QDRANT_API_KEY = process.env.QDRANT_API_KEY;

if (!QDRANT_URL) {
  throw new Error("Missing required env var: QDRANT_URL");
}

const qdrant = new QdrantClient({
  url: QDRANT_URL,
  apiKey: QDRANT_API_KEY,
});

const COLLECTION_NAME = "knowledge_chunks";
const VECTOR_SIZE = 1536;

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
  const query = body.query;
  const rawLimit = Number(body.limit || 5);
  const limit =
    Number.isFinite(rawLimit) && rawLimit > 0 && rawLimit <= 20 ? rawLimit : 5;

  if (!query || typeof query !== "string" || !query.trim()) {
    return reply.code(400).send({ error: "query is required" });
  }

  return {
    ok: true,
    mode: "placeholder",
    received: {
      query: query.trim(),
      limit,
    },
    note: "Busca vetorial real entra no proximo passo",
  };
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
