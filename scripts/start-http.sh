#!/usr/bin/env bash
# start-http.sh — 自动配置环境并启动 ACI HTTP 服务器 (FastAPI / uvicorn)
# 用法：bash scripts/start-http.sh [--host HOST] [--port PORT]
set -euo pipefail

# ---------- 路径 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ---------- 颜色输出 ----------
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[ACI]${NC} $*"; }
success() { echo -e "${GREEN}[ACI]${NC} $*"; }
warn()    { echo -e "${YELLOW}[ACI]${NC} $*"; }
error()   { echo -e "${RED}[ACI]${NC} $*" >&2; }

# ---------- 参数默认值 ----------
HOST="${ACI_SERVER_HOST:-0.0.0.0}"
PORT="${ACI_SERVER_PORT:-8000}"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    *) warn "未知参数 '$1'，已忽略"; shift ;;
  esac
done

echo ""
echo -e "${CYAN}================================================${NC}"
echo -e "${CYAN}  ACI — HTTP 服务器启动脚本${NC}"
echo -e "${CYAN}================================================${NC}"
echo ""

# ---------- 检查 uv ----------
if ! command -v uv &>/dev/null; then
  warn "未检测到 uv，正在安装..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$PATH"
  if ! command -v uv &>/dev/null; then
    error "uv 安装失败，请手动安装: https://docs.astral.sh/uv/"
    exit 1
  fi
  success "uv 安装成功"
else
  info "uv 已就绪: $(uv --version)"
fi

# ---------- 检查 .env ----------
if [[ ! -f ".env" ]]; then
  warn ".env 文件不存在，从 .env.example 复制..."
  if [[ -f ".env.example" ]]; then
    cp ".env.example" ".env"
    echo ""
    warn "=========================================================="
    warn "  请编辑 .env 文件，填写以下必填项后重新运行本脚本："
    warn "  - ACI_EMBEDDING_API_URL"
    warn "  - ACI_EMBEDDING_API_KEY"
    warn "  - ACI_EMBEDDING_MODEL"
    warn "=========================================================="
    echo ""
    read -rp "已打开 .env，配置完成后按 [Enter] 继续，或 Ctrl+C 退出..." _
  else
    error ".env.example 不存在，请手动创建 .env 文件"
    exit 1
  fi
else
  info ".env 已存在"
fi

# ---------- 检查必填环境变量 ----------
# shellcheck source=/dev/null
set -a; source ".env"; set +a
MISSING=()
[[ -z "${ACI_EMBEDDING_API_KEY:-}" || "${ACI_EMBEDDING_API_KEY}" == "your_embedding_api_key" ]] && MISSING+=("ACI_EMBEDDING_API_KEY")
[[ -z "${ACI_EMBEDDING_API_URL:-}" ]] && MISSING+=("ACI_EMBEDDING_API_URL")
if [[ ${#MISSING[@]} -gt 0 ]]; then
  error "以下环境变量未配置: ${MISSING[*]}"
  error "请编辑 .env 文件后重试"
  exit 1
fi

# ---------- 同步依赖 ----------
info "同步依赖（uv sync）..."
uv sync --quiet
success "依赖同步完成"

# ---------- 启动 HTTP 服务器 ----------
echo ""
success "启动 ACI HTTP 服务器 → http://${HOST}:${PORT}"
info "API 文档: http://${HOST}:${PORT}/docs"
info "按 Ctrl+C 停止服务"
echo ""
exec uv run aci serve --host "$HOST" --port "$PORT"
