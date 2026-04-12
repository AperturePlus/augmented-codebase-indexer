# Security

ACI includes built-in protections that cannot be overridden by user configuration.

## System Directory Protection

Indexing system directories is blocked across all interfaces (CLI, HTTP, MCP):

- Unix: `/etc`, `/var`, `/sys`, `/proc`, `/boot`, `/dev`
- Windows: `C:\Windows`, `C:\System32`

## Sensitive File Denylist

The following files are automatically excluded from indexing regardless of configuration:

| Category | Patterns |
|----------|----------|
| SSH keys & dirs | `.ssh`, `id_rsa`, `id_ed25519`, etc. |
| GPG | `.gnupg` |
| Certificates & private keys | `*.pem`, `*.key`, `*.p12`, `*.pfx`, `*.crt` |
| Environment files | `.env`, `.env.*` |
| Credential files | `.netrc`, `.npmrc`, `.pypirc` |

These rules apply at the indexing layer and are enforced before any user-supplied include/exclude patterns are evaluated.
