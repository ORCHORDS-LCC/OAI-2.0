# Public Source References

_Last reviewed: 2026-10-01._

Primary sources used by OAI-2.0 architecture documentation. Re-check them before implementation because APIs evolve.

## Cloudflare Workers / storage

- Workers bindings: https://developers.cloudflare.com/workers/runtime-apis/bindings/
- Python Workers: https://developers.cloudflare.com/workers/languages/python/
- Python Workers FFI/bindings: https://developers.cloudflare.com/workers/languages/python/ffi/
- D1 Workers Binding API: https://developers.cloudflare.com/d1/worker-api/
- D1 prepared statements: https://developers.cloudflare.com/d1/worker-api/prepared-statements/
- R2 Workers API reference: https://developers.cloudflare.com/r2/api/workers/workers-api-reference/
- Vectorize API: https://developers.cloudflare.com/vectorize/reference/client-api/
- Workers KV bindings: https://developers.cloudflare.com/kv/concepts/kv-bindings/
- Workers KV reads/cache TTL: https://developers.cloudflare.com/kv/api/read-key-value-pairs/
- Workers KV writes/expiration: https://developers.cloudflare.com/kv/api/write-key-value-pairs/
- Workers KV consistency: https://developers.cloudflare.com/kv/concepts/how-kv-works/

## q-pipe public source

OAI-2.0's q-pipe import contract is checked against current public q-pipe source, especially:

- https://github.com/ORCHORDS/q-pipe/blob/main/qpipe/cloudflare_learning.py
- https://github.com/ORCHORDS/q-pipe/blob/main/qpipe/memory.py
- https://github.com/ORCHORDS/q-pipe/blob/main/docs/LEARNING.md
- https://github.com/ORCHORDS/q-pipe/blob/main/docs/ANDROID_CURRICULUM.md

Re-check q-pipe before future migrations because its export gate can evolve.

## Android Studio

- Local model support: https://developer.android.com/studio/gemini/use-a-local-model
- Gemini/agent features: https://developer.android.com/studio/gemini/features
- MCP: https://developer.android.com/studio/gemini/add-mcp-server

## Hermes Agent

- MCP: https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp/
- ACP: https://hermes-agent.nousresearch.com/docs/user-guide/features/acp

## OpenCode

- Providers: https://opencode.ai/docs/providers/
- MCP: https://opencode.ai/docs/mcp-servers/

## Source policy

A vendor/repository statement is not an OAI-2.0 support claim. Support requires our own versioned integration/evaluation evidence.
