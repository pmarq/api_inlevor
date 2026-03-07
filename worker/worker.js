import dotenv from "dotenv";
import admin from "firebase-admin";

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

const REQUIRED_ENV = [
  "FIREBASE_PROJECT_ID",
  "FIREBASE_PRIVATE_KEY_ID",
  "FIREBASE_PRIVATE_KEY",
  "FIREBASE_CLIENT_EMAIL",
  "FIREBASE_CLIENT_ID",
  "QDRANT_INGEST_URL",
  "SOURCE_PROJECT",
];

for (const key of REQUIRED_ENV) {
  if (!process.env[key]) {
    throw new Error(`Missing required env var: ${key}`);
  }
}

const app = admin.initializeApp({
  credential: admin.credential.cert({
    type: "service_account",
    project_id: process.env.FIREBASE_PROJECT_ID,
    private_key_id: process.env.FIREBASE_PRIVATE_KEY_ID,
    private_key: process.env.FIREBASE_PRIVATE_KEY?.replace(/\\n/g, "\n"),
    client_email: process.env.FIREBASE_CLIENT_EMAIL,
    client_id: process.env.FIREBASE_CLIENT_ID,
    auth_uri: "https://accounts.google.com/o/oauth2/auth",
    token_uri: "https://oauth2.googleapis.com/token",
    auth_provider_x509_cert_url: "https://www.googleapis.com/oauth2/v1/certs",
    client_x509_cert_url: process.env.FIREBASE_CLIENT_CERT_URL,
    universe_domain: "googleapis.com",
  }),
});

const db = admin.firestore(app);

const POLL_INTERVAL_MS = Number(process.env.POLL_INTERVAL_MS || 8000);
const CHUNK_SIZE = Number(process.env.CHUNK_SIZE || 1200);
const CHUNK_OVERLAP = Number(process.env.CHUNK_OVERLAP || 200);
const MAX_JOBS_PER_TICK = Number(process.env.MAX_JOBS_PER_TICK || 3);
const JOB_TYPE = process.env.JOB_TYPE || "extract_facts";
const MAX_SOURCE_CHARS = Number(process.env.MAX_SOURCE_CHARS || 500000);
const INGEST_BATCH_SIZE = Number(process.env.INGEST_BATCH_SIZE || 64);

const QDRANT_INGEST_URL = process.env.QDRANT_INGEST_URL;
const SOURCE_PROJECT = process.env.SOURCE_PROJECT;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function validateChunkConfig() {
  if (CHUNK_SIZE <= 0) {
    throw new Error(`Invalid CHUNK_SIZE: ${CHUNK_SIZE}`);
  }

  if (CHUNK_OVERLAP < 0) {
    throw new Error(`Invalid CHUNK_OVERLAP: ${CHUNK_OVERLAP}`);
  }

  if (CHUNK_OVERLAP >= CHUNK_SIZE) {
    throw new Error(
      `Invalid chunk config: CHUNK_OVERLAP (${CHUNK_OVERLAP}) must be smaller than CHUNK_SIZE (${CHUNK_SIZE})`,
    );
  }
}

function* chunkTextIterator(text) {
  const cleaned = (text || "").replace(/\s+/g, " ").trim();
  if (!cleaned) return;

  validateChunkConfig();

  let start = 0;

  while (start < cleaned.length) {
    const end = Math.min(start + CHUNK_SIZE, cleaned.length);
    const chunk = cleaned.slice(start, end).trim();

    if (chunk) {
      yield chunk;
    }

    if (end >= cleaned.length) {
      break;
    }

    const nextStart = end - CHUNK_OVERLAP;

    if (nextStart <= start) {
      throw new Error(
        `Invalid chunk progression: start=${start}, end=${end}, nextStart=${nextStart}`,
      );
    }

    start = nextStart;
  }
}

async function sendBatch(payload) {
  const res = await fetch(QDRANT_INGEST_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Ingest API failed: ${res.status} ${body}`);
  }
}

async function claimJob(docRef) {
  return db.runTransaction(async (tx) => {
    const snap = await tx.get(docRef);
    if (!snap.exists) return null;

    const data = snap.data();
    if (!data || data.status !== "queued") return null;

    tx.update(docRef, {
      status: "processing",
      updatedAt: new Date(),
    });

    return { id: snap.id, ...data };
  });
}

async function processJob(job) {
  const projectId = job?.refs?.projectId || job?.projectId || job?.id;
  if (!projectId) {
    throw new Error("Job missing projectId");
  }

  const assetSnap = await db.collection("kb_assets").doc(projectId).get();
  if (!assetSnap.exists) {
    throw new Error("kb_assets not found for projectId");
  }

  const asset = assetSnap.data();
  const rawText = (asset?.rawTextSanitized || "").trim();
  const residentialText = (asset?.residentialTextSanitized || "").trim();
  const sourceText = residentialText || rawText;

  if (!sourceText) {
    throw new Error("No sanitized text available for embeddings");
  }

  const limitedText =
    sourceText.length > MAX_SOURCE_CHARS
      ? sourceText.slice(0, MAX_SOURCE_CHARS)
      : sourceText;

  const source = residentialText ? "residential" : "raw";
  const batch = [];
  let index = 0;

  for (const chunk of chunkTextIterator(limitedText)) {
    batch.push({
      id: `${SOURCE_PROJECT}:${projectId}:${index}`,
      text: chunk,
      index,
    });
    index += 1;

    if (batch.length >= INGEST_BATCH_SIZE) {
      await sendBatch({
        sourceProject: SOURCE_PROJECT,
        projectId,
        source,
        chunks: batch,
      });
      batch.length = 0;
    }
  }

  if (!index) {
    throw new Error("Chunking produced no text");
  }

  if (batch.length) {
    await sendBatch({
      sourceProject: SOURCE_PROJECT,
      projectId,
      source,
      chunks: batch,
    });
  }
}

async function markJob(docRef, status, error) {
  const update = {
    status,
    updatedAt: new Date(),
  };

  if (error) {
    update.error = String(error).slice(0, 1000);
  }

  await docRef.update(update);
}

async function tick() {
  const snapshot = await db
    .collection("kb_jobs")
    .where("status", "==", "queued")
    .where("jobType", "==", JOB_TYPE)
    .limit(MAX_JOBS_PER_TICK)
    .get();

  if (snapshot.empty) return;

  for (const doc of snapshot.docs) {
    const claimed = await claimJob(doc.ref);
    if (!claimed) continue;

    try {
      await processJob(claimed);
      await markJob(doc.ref, "done");
    } catch (err) {
      console.error("Job processing error", {
        jobId: doc.id,
        error: err instanceof Error ? err.message : String(err),
      });
      await markJob(doc.ref, "error", err);
    }
  }
}

async function main() {
  console.log("KB worker started");
  console.log(
    JSON.stringify({
      SOURCE_PROJECT,
      JOB_TYPE,
      POLL_INTERVAL_MS,
      CHUNK_SIZE,
      CHUNK_OVERLAP,
      MAX_JOBS_PER_TICK,
      MAX_SOURCE_CHARS,
      INGEST_BATCH_SIZE,
    }),
  );

  while (true) {
    try {
      await tick();
    } catch (err) {
      console.error("Tick error", err);
    }

    await sleep(POLL_INTERVAL_MS);
  }
}

main();
