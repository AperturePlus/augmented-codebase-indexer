# CLI Usage

## Basic Commands

```bash
# Index a codebase
aci index /path/to/codebase

# Search for code
aci search "function that handles authentication"

# Check index status
aci status

# Update index incrementally
aci update

# Reset index (drop collection & metadata)
aci reset

# Start interactive shell mode
aci shell

# Start HTTP server (FastAPI)
aci serve --host 0.0.0.0 --port 8000

# Also available via python -m
uv run python -m aci serve
```

## Search Query Modifiers

Filter results inline without extra flags:

| Modifier | Description | Example |
|----------|-------------|---------|
| `path:<pattern>` | Include only files matching pattern | `path:*.py`, `path:src/**` |
| `file:<pattern>` | Alias for `path:` | `file:handlers.py` |
| `-path:<pattern>` | Exclude files matching pattern | `-path:tests` |
| `exclude:<pattern>` | Alias for `-path:` | `exclude:fixtures` |

```bash
aci search "parse config path:*.py"
aci search "database connection -path:tests -path:fixtures"
```

## Artifact Type Filtering

ACI indexes code at multiple granularity levels. Filter with `--type` / `-t`:

| Type | Description |
|------|-------------|
| `chunk` | Functions, classes, or fixed-size code blocks |
| `function_summary` | Natural language summaries of functions |
| `class_summary` | Natural language summaries of classes |
| `file_summary` | File-level summaries describing overall purpose |

```bash
# Search only code chunks
aci search "authentication" --type chunk

# High-level semantic queries
aci search "what handles user login" --type function_summary --type class_summary

# Mix types
aci search "config parsing" -t chunk -t file_summary
```

By default (no `--type`), all artifact types are searched.

## Interactive Shell

`aci shell` launches a REPL with command history, tab completion, and persistent history across sessions.

```bash
aci shell
```

### Available Shell Commands

| Command | Description |
|---------|-------------|
| `index <path>` | Index a directory |
| `search <query>` | Search the indexed codebase |
| `status` | Show index statistics |
| `update <path>` | Incrementally update the index |
| `list` | List indexed repositories |
| `reset` | Clear the index (requires confirmation) |
| `help` / `?` | Display available commands |
| `exit` / `quit` / `q` | Exit the shell |

### Example Session

```text
$ aci shell

    _    ____ ___   ____  _          _ _
   / \  / ___|_ _| / ___|| |__   ___| | |
  / _ \| |    | |  \___ \| '_ \ / _ \ | |
 / ___ \ |___ | |   ___) | | | |  __/ | |
/_/   \_\____|___| |____/|_| |_|\___|_|_|

Welcome to ACI Interactive Shell
Type 'help' for available commands, 'exit' to quit

aci> index ./src
Indexing ./src...
✓ Indexed 42 files, 156 chunks

aci> search "authentication handler"
Found 3 results:
...

aci> search "config parser path:src/*.py -path:tests"
Found 2 results:
...

aci> exit
Goodbye!
```
