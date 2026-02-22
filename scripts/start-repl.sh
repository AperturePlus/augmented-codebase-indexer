#!/usr/bin/env bash
# start-repl.sh — 自动配置环境并启动 ACI 交互式 REPL（shell 模式）
# 用法：bash scripts/start-repl.sh
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

echo ""
echo -e "${CYAN}================================================${NC}"
echo -e "${CYAN}  ACI — 交互式 REPL 启动脚本${NC}"
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
    read -rp "配置完成后按 [Enter] 继续，或 Ctrl+C 退出..." _
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

# ---------- 提示 ----------
echo ""
info "REPL 常用命令："
info "  index <path>   — 索引目录"
info "  search <query> — 语义搜索"
info "  status         — 查看索引状态"
info "  update <path>  — 增量更新索引"
info "  reset          — 清空索引"
info "  help / ?       — 查看帮助"
info "  exit / q       — 退出"
echo ""

# ---------- 启动 REPL ----------
success "进入 ACI 交互式 Shell..."
echo ""
exec uv run aci shell
