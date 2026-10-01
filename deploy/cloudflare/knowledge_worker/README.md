# OAI-2.0 Cloudflare Knowledge Worker

_Reconciliation baseline: `8520cdb5d8d3e49bc9e12dfd79ccf59e7422e44d` (source state before this documentation commit)._

This directory is a **public-safe deployment scaffold** for the private OAI-2.0
knowledge Worker. It contains no live account IDs, database IDs, namespace IDs,
bucket names, endpoints, or credentials.

## Current source boundary

The entrypoint wires the repository's current async source implementation:

- D1 authoritative knowledge metadata + corpus revision;
- D1 GC deletion-lease authority;
- R2 content-addressed bodies;
- Vectorize binding wrapper;
- best-effort Workers KV query cache;
- authenticated versioned `KnowledgeTransportRequest/Response`;
- async `put/get/retrieve` runtime.

The Worker currently accepts **POST** JSON requests only. A bearer token from
the `KNOWLEDGE_AUTH_TOKEN` Worker secret is authenticated before the Worker
creates the normalized read/write capability context. Any client-supplied
`auth` field is overwritten and is never trusted as authentication evidence.

## Configure privately

1. Copy `wrangler.example.toml` to your private deployment configuration.
2. Replace the resource placeholders outside public source.
3. Set the authentication secret:

   `npx wrangler secret put KNOWLEDGE_AUTH_TOKEN`

4. Keep `EMBEDDING_VERSION` and `EMBEDDING_DIGEST` aligned with the
   embedding/index version actually deployed.
5. Deploy only after private D1/R2/Vectorize/KV bindings exist and backup /
   recovery prerequisites for destructive GC are satisfied.

## Embeddings

The generic Worker entrypoint currently assembles the transport with no concrete
embedding provider. Therefore normal topic retrieval works through authoritative
D1/R2, while semantic Vectorize retrieval requires injecting a verified
`EmbeddingProvider` in the private deployment. The public repository does not
hard-code a provider/model arrangement.

## Evidence boundary

This scaffold is **source implementation**, not proof of a deployed Worker.
Issue #19 remains open until a controlled private deployment demonstrates live
put/get/update/retrieve, Vectorize known-neighbor retrieval, revisioned cache
invalidation, and explicit dependency failures. Issue #233 remains open until
the conservative sweep planner is wired through the D1/R2 delete boundary and
the private concurrency/recovery demonstration passes.
