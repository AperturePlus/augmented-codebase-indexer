# Security Policy

## Handling Sensitive Information

ACI (Augmented Codebase Indexer) takes security seriously and implements multiple safeguards to protect sensitive information.

### Automatic Protection

The following measures are automatically in place:

#### 1. Sensitive File Denylist

ACI automatically excludes sensitive files from indexing, regardless of user configuration:

- **SSH keys and directories**: `.ssh`, `id_rsa`, `id_ed25519`, etc.
- **GPG directories**: `.gnupg`
- **Certificates and private keys**: `*.pem`, `*.key`, `*.p12`, `*.pfx`, `*.crt`, `*.keystore`
- **Environment files**: `.env`, `.env.*`, `.env.local`, `.env.production`, `.env.development`
- **Credential files**: `.netrc`, `.npmrc`, `.pypirc`

See `src/aci/core/file_scanner.py` for the complete list.

#### 2. Safe Configuration Serialization

When logging or debugging configuration:

```python
from aci.core.config import ACIConfig

config = ACIConfig()

# ❌ UNSAFE - exposes API keys
print(config.to_dict())  # Contains actual API keys

# ✅ SAFE - redacts sensitive information
print(config.to_dict_safe())  # API keys replaced with [REDACTED]
```

The `to_dict_safe()` method automatically redacts:
- `embedding.api_key`
- `search.rerank_api_key`

#### 3. Debug Logging

Debug logging only shows the presence of API keys, not their values:

```python
# Example from src/aci/mcp/handlers.py
_debug(f"API key present: {bool(cfg.embedding.api_key)}")  # ✅ Safe
```

### Best Practices for Developers

#### Environment Variables

**Always** use environment variables for sensitive information:

```bash
# Use .env file (automatically excluded from git)
ACI_EMBEDDING_API_KEY=your_api_key_here
ACI_SEARCH_RERANK_API_KEY=your_rerank_key_here
```

**Never** commit `.env` files. Use `.env.example` instead:

```bash
# .env.example (safe to commit)
ACI_EMBEDDING_API_KEY=your_api_key_here
```

#### Git Configuration

Ensure your `.gitignore` includes:

```gitignore
.env
.env.local
.env.*.local
*.key
*.pem
secrets/
.secrets/
```

#### Code Reviews

When logging or printing configuration:

1. ✅ Use `config.to_dict_safe()` instead of `config.to_dict()`
2. ✅ Log boolean presence: `bool(api_key)` 
3. ❌ Never log API key values or prefixes
4. ❌ Never log sensitive configuration fields directly

#### Testing

Use clearly fake values in tests:

```python
# ✅ Good - obviously not real
api_key="test-key"
api_key="fake-key-for-testing"
api_key="mock"

# ❌ Bad - looks like it could be real
api_key="sk-abc123xyz789"
```

### Reporting Security Issues

If you discover a security vulnerability in ACI, please report it by:

1. **Do NOT** open a public GitHub issue
2. Email the maintainers directly (see project maintainers in README.md)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will acknowledge receipt within 48 hours and provide a detailed response within 5 business days.

### Security Scanning

This project uses:

- **CodeQL**: Automated security scanning for code vulnerabilities
- **Dependency scanning**: Regular checks for vulnerable dependencies
- **Pre-commit hooks**: Prevent accidental commits of sensitive files

### Version Support

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | ✅ Yes             |
| < 0.2   | ❌ No              |

### Changes

For security-related changes, see our [CHANGELOG.md](CHANGELOG.md) (if available) or commit history.

---

**Last Updated**: December 2025
