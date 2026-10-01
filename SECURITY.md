# Security Reporting

_Last reviewed: 2026-10-01._

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

The current Cloudflare code is a logical/mock contract only. Live Cloudflare deployment must keep resource bindings and secrets out of public source.

q-pipe knowledge imports must pass the strict verified export gate before entering OAI-2.0's shared knowledge layer.

## License/security contact

**crm@orchords.com**
