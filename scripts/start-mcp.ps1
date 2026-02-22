#Requires -Version 5.1
<#
.SYNOPSIS
    自动配置 ACI 环境并启动 MCP 服务器（供 LLM 客户端接入）

.DESCRIPTION
    1. 检查并安装 uv 包管理器
    2. 检查 .env 配置文件（不存在则从 .env.example 复制并提示填写）
    3. 验证必填环境变量
    4. 执行 uv sync 同步依赖
    5. 打印 MCP 客户端配置示例
    6. 启动 ACI MCP 服务器（stdio 传输）

.EXAMPLE
    .\scripts\start-mcp.ps1
#>
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------- 路径 ----------
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

# ---------- 颜色辅助函数 ----------
function Write-Info    { param($Msg) Write-Host "[ACI] $Msg" -ForegroundColor Cyan    }
function Write-Success { param($Msg) Write-Host "[ACI] $Msg" -ForegroundColor Green   }
function Write-Warn    { param($Msg) Write-Host "[ACI] $Msg" -ForegroundColor Yellow  }
function Write-Err     { param($Msg) Write-Host "[ACI] $Msg" -ForegroundColor Red     }

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  ACI — MCP 服务器启动脚本"                      -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# ---------- 检查 uv ----------
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCmd) {
    Write-Warn "未检测到 uv，正在安装..."
    try {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "User") + ";" + $env:PATH
        $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
        if (-not $uvCmd) { throw "安装后仍未找到 uv" }
        Write-Success "uv 安装成功"
    } catch {
        Write-Err "uv 安装失败: $_"
        Write-Err "请手动安装: https://docs.astral.sh/uv/"
        exit 1
    }
} else {
    Write-Info "uv 已就绪: $(uv --version)"
}

# ---------- 检查 .env ----------
if (-not (Test-Path ".env")) {
    Write-Warn ".env 文件不存在，从 .env.example 复制..."
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host ""
        Write-Warn "=========================================================="
        Write-Warn "  请编辑 .env 文件，填写以下必填项后重新运行本脚本："
        Write-Warn "  - ACI_EMBEDDING_API_URL"
        Write-Warn "  - ACI_EMBEDDING_API_KEY"
        Write-Warn "  - ACI_EMBEDDING_MODEL"
        Write-Warn "=========================================================="
        Write-Host ""
        Start-Process notepad.exe -ArgumentList (Resolve-Path ".env").Path -Wait
    } else {
        Write-Err ".env.example 不存在，请手动创建 .env 文件"
        exit 1
    }
} else {
    Write-Info ".env 已存在"
}

# ---------- 加载 .env 并验证必填项 ----------
$envContent = Get-Content ".env" -ErrorAction SilentlyContinue
$envVars = @{}
foreach ($line in $envContent) {
    if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
        $envVars[$Matches[1]] = $Matches[2].Trim('"').Trim("'")
    }
}

$missing = @()
if (-not $envVars["ACI_EMBEDDING_API_KEY"] -or $envVars["ACI_EMBEDDING_API_KEY"] -eq "your_embedding_api_key") {
    $missing += "ACI_EMBEDDING_API_KEY"
}
if (-not $envVars["ACI_EMBEDDING_API_URL"]) {
    $missing += "ACI_EMBEDDING_API_URL"
}
if ($missing.Count -gt 0) {
    Write-Err "以下环境变量未配置: $($missing -join ', ')"
    Write-Err "请编辑 .env 文件后重试"
    exit 1
}

# ---------- 同步依赖 ----------
Write-Info "同步依赖（uv sync）..."
uv sync --quiet
if ($LASTEXITCODE -ne 0) { Write-Err "依赖同步失败"; exit 1 }
Write-Success "依赖同步完成"

# ---------- 打印 MCP 客户端配置示例 ----------
Write-Host ""
Write-Info "MCP 客户端配置参考（粘贴到 Kiro / Claude Desktop / Cursor 等 MCP 客户端）："
Write-Host @"
{
  "mcpServers": {
    "aci": {
      "command": "uv",
      "args": ["run", "aci-mcp"],
      "cwd": "$($ProjectRoot.Replace('\','\\'))"
    }
  }
}
"@ -ForegroundColor DarkGray
Write-Host ""

# ---------- 启动 MCP 服务器 ----------
Write-Success "启动 ACI MCP 服务器（stdio 传输模式）..."
Write-Info "按 Ctrl+C 停止服务"
Write-Host ""

uv run aci-mcp
