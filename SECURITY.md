# Security Reporting

_Last reviewed: 2026-10-02._

Do **not** open a public GitHub issue for vulnerabilities, exposed secrets, private-data disclosure, sandbox escapes, privilege escalation, prompt-injection bypasses, or other security-sensitive findings.

Email **crm@orchords.com** with a concise description, affected public resource if known, safe reproduction information, potential impact, and preferred contact details.

## OAI-2.0 public-repository boundary

Never commit or publish:

- credentials, API tokens, or private keys;
- Cloudflare account IDs, real resource IDs, private hostnames, or private Worker endpoints;
- private provider details;
- customer/private repository content;
- sensitive training sources or corpora;
- hidden verifier content;
- private infrastructure topology.

`.env.example` must remain placeholder-only.

## Agent/runtime security

Tool permissions must be enforced independently of model text. Untrusted repository, web, issue, screenshot/OCR, and tool content cannot expand the runtime capability set.

The current tool dispatcher is an **EXPERIMENTAL** policy scaffold. It is not a claim that autonomous production execution is secure.

## Knowledge security

The Cloudflare knowledge layer now includes source-level D1/R2/KV/Vectorize wrappers, knowledge/GC contracts, and deterministic mocks. A complete live Worker deployment is still not verified end-to-end. Resource bindings, account/resource IDs, credentials, and private endpoints must remain outside public source.

q-pipe knowledge imports must pass the strict verified export gate before entering OAI-2.0's shared knowledge layer.

Destructive knowledge lifecycle operations must remain fail-closed: dry-run first, explicit authorization/recovery readiness, authoritative D1 reference/lease checks, and no deletion based on KV/cache state.

## License/security contact

**crm@orchords.com**
