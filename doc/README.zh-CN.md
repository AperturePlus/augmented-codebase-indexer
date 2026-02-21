# Project ACI - 增强代码库索引器

Language: [English](../README.md) | **简体中文**

一个用于语义化代码搜索的 Python 工具，支持精确到行号级别的定位结果。

## 功能特性

- 基于 Embedding 的语义代码搜索（兼容 OpenAI API）
- 精确到行号级别的定位结果
- 支持 Python、JavaScript/TypeScript、Go、Java、C、C++
- 基于 Tree-sitter 的 AST 解析与精确分块
- 混合检索（语义 + 关键词/grep）
- 集成 Qdrant 向量数据库
- 支持增量索引，提升更新效率
- 多种接口：CLI、HTTP API、MCP（用于 LLM 集成）
- 自动检测本地时区并用于时间戳显示

## 安装

```bash
# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install -e ".[dev]"
```

## 环境要求

- Python 3.10+
- Qdrant（本地可通过 Docker 自动启动，或使用云端 URL + API Key）
- 兼容 OpenAI 的 Embedding API（如 OpenAI、SiliconFlow 等）

## 基本使用

```bash
# 建立代码库索引
aci index /path/to/codebase

# 搜索代码
aci search "function that handles authentication"

# 带文件路径过滤搜索
aci search "parse config path:*.py"

# 排除某些路径
aci search "database connection -path:tests"

# 查看索引状态
aci status

# 增量更新索引
aci update

# 重置索引（删除 collection 与元数据）
aci reset

# 启动交互式 shell
aci shell

# 启动 HTTP 服务（FastAPI）
aci serve --host 0.0.0.0 --port 8000

# 也可通过 python -m 入口启动
uv run python -m aci serve  # 使用 uv 时

# 启动 MCP 服务（用于 LLM 集成）
aci-mcp
# 或
uv run aci-mcp
```

## 交互式 Shell 模式

ACI 提供交互式 shell 模式，你可以连续执行多条命令而无需每次重启程序。对于反复索引、搜索和调优查询的工作流非常高效。

### 启动 Shell

```bash
aci shell
```

启动后会进入 REPL（Read-Eval-Print Loop），包含：
- 命令历史（方向键上下浏览）
- 命令自动补全（Tab）
- 跨会话持久化历史

### 可用命令

| 命令 | 说明 |
|------|------|
| `index <path>` | 为目录建立语义索引 |
| `search <query>` | 搜索已索引代码库（支持修饰符） |
| `status` | 查看索引状态和统计 |
| `update <path>` | 增量更新索引 |
| `list` | 列出已索引仓库（`aci list --global` 查看全局注册表） |
| `reset` | 清空索引（需要确认） |
| `help` 或 `?` | 显示可用命令 |
| `exit`、`quit` 或 `q` | 退出 shell |

### 示例会话

```
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

## 搜索查询修饰符

搜索语句支持内联修饰符来过滤结果：

| 修饰符 | 说明 | 示例 |
|--------|------|------|
| `path:<pattern>` | 仅包含匹配路径的文件 | `path:*.py`, `path:src/**` |
| `file:<pattern>` | `path:` 别名 | `file:handlers.py` |
| `-path:<pattern>` | 排除匹配路径的文件 | `-path:tests` |
| `exclude:<pattern>` | `-path:` 别名 | `exclude:fixtures` |

可组合多个排除条件：

```bash
aci search "database query -path:tests -path:fixtures"
```

## Artifact 类型过滤

ACI 会以多种粒度索引代码。你可以通过 `--type` / `-t` 按 Artifact 类型过滤结果：

| Artifact 类型 | 说明 |
|---------------|------|
| `chunk` | 代码分块（函数、类或固定长度块） |
| `function_summary` | 函数的自然语言摘要 |
| `class_summary` | 类的自然语言摘要 |
| `file_summary` | 文件级摘要（描述整体用途） |

```bash
# 仅搜索代码块
aci search "authentication" --type chunk

# 仅搜索摘要（适合高层问题）
aci search "what handles user login" --type function_summary --type class_summary

# 组合多种类型
aci search "config parsing" -t chunk -t file_summary
```

默认情况下（不指定 `--type`）会返回所有 Artifact 类型的结果。

## MCP 集成

ACI 支持 Model Context Protocol（MCP），使 LLM 可直接调用你的代码索引与搜索能力。

### MCP 快速开始

1. 配置 MCP 客户端（如 Kiro、Claude Desktop、Cursor）：

```json
{
  "mcpServers": {
    "aci": {
      "command": "uv",
      "args": ["run", "aci-mcp"],
      "cwd": "/path/to/your/project"
    }
  }
}
```

2. 确保工作目录存在 `.env` 且配置完整（参考 `../.env.example`）

3. 可用自然语言与代码库交互，例如：
   - “索引当前目录”
   - “搜索认证相关函数”
   - “查看当前索引状态”

### MCP 可用工具

| 工具 | 说明 |
|------|------|
| `index_codebase` | 为目录建立语义索引 |
| `search_code` | 使用自然语言搜索代码 |
| `get_index_status` | 获取索引统计和健康状态 |
| `update_index` | 增量更新索引 |
| `list_indexed_repos` | 列出已索引仓库 |

### MCP 测试

```bash
# 使用 MCP Inspector（Web UI）
npx @modelcontextprotocol/inspector uv run aci-mcp

# 使用 Python 脚本测试
uv run python tests/test_mcp_call/test_stdio.py

# 测试索引
uv run python tests/test_mcp_call/test_index_codebase.py
```

### 搜索质量评估脚本

使用独立质量评估脚本（不放入 `tests/`，避免 CI 不稳定）：

```bash
# 假设索引已存在
uv run python scripts/measure_mcp_search.py

# 先强制重建索引再评估
REINDEX=1 uv run python scripts/measure_mcp_search.py
```

### 调试模式

在 `.env` 中设置 `ACI_ENV=development` 以启用调试日志：

```
ACI_ENV=development
```

调试信息会输出到 stderr，并可在 MCP Inspector 的 notifications 中查看。

> **注意**：MCP 为了 stdio 兼容性采用单线程索引。若需更快地索引大型代码库，请使用 CLI：`uv run aci index .`

## 安全性

ACI 内置以下安全保护：

- **系统目录保护**：禁止索引系统目录（如 `/etc`、`/var`、`C:\Windows`），在 CLI、HTTP、MCP 全接口统一生效
- **敏感文件拒绝列表**：以下文件会被自动排除，且不受用户配置覆盖：
  - SSH 密钥与目录（`.ssh`、`id_rsa`、`id_ed25519` 等）
  - GPG 目录（`.gnupg`）
  - 证书和私钥（`*.pem`、`*.key`、`*.p12`、`*.pfx`、`*.crt`）
  - 环境变量文件（`.env`、`.env.*`）
  - 凭据文件（`.netrc`、`.npmrc`、`.pypirc`）

这些保护策略不能被用户配置覆盖。

## 配置

通过 `.env` 文件或环境变量进行配置。可先复制 `../.env.example`：

```bash
cp .env.example .env
```

关键配置项：

| 变量 | 说明 | 必填 |
|------|------|------|
| `ACI_EMBEDDING_API_KEY` | Embedding 服务 API Key | 是 |
| `ACI_EMBEDDING_API_URL` | Embedding API 地址 | 否（默认 OpenAI） |
| `ACI_EMBEDDING_MODEL` | 模型名称 | 否 |
| `ACI_VECTOR_STORE_URL` | Qdrant 基础 URL（优先于 host/port） | 否 |
| `ACI_VECTOR_STORE_API_KEY` | Qdrant API Key（Qdrant Cloud） | 否 |
| `ACI_VECTOR_STORE_HOST` | Qdrant 主机地址 | 否（默认 localhost） |
| `ACI_VECTOR_STORE_PORT` | Qdrant 端口 | 否（默认 6333） |
| `ACI_SERVER_HOST` | HTTP 服务主机地址 | 否（默认 0.0.0.0） |
| `ACI_SERVER_PORT` | HTTP 服务端口 | 否（默认 8000） |
| `ACI_ENV` | 运行环境（development/production） | 否 |

完整配置请查看 `../.env.example`。

CLI 和 HTTP 服务仅在目标为本地端点（`localhost` / `127.0.0.1`）时尝试自动启动本地 Qdrant Docker 容器。若使用云端 Qdrant（`ACI_VECTOR_STORE_URL`），不会启动 Docker。
