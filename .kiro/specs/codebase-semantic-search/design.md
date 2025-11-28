# 概要设计: Augmented Codebase Indexer (Project ACI)

## Overview

Project ACI 是一个Python代码库语义检索工具，采用分层架构设计。系统使用 Tree-sitter 进行AST解析，Qdrant 作为向量数据库，通过 API 调用 Embedding 模型。

### 设计目标

- **可维护性**: 清晰的模块边界，依赖注入
- **可扩展性**: 插件式语言和模型扩展
- **性能**: 并行处理、批量操作
- **可测试性**: 核心逻辑与外部依赖分离

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      CLI Layer (Typer)                       │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Service Layer                           │
│    IndexingService  │  SearchService  │  EvaluationService   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                       Core Layer                             │
│  FileScanner │ AST_Parser │ Chunker │ Tokenizer │ ChangeDet  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Infrastructure Layer                       │
│      EmbeddingClient (API)  │  VectorStore (Qdrant)          │
└─────────────────────────────────────────────────────────────┘
```

### 架构原则

1. **依赖倒置**: 上层依赖抽象接口
2. **单一职责**: 每组件一个明确功能
3. **开闭原则**: 通过接口扩展

## Core Components

| 组件 | 职责 | 输入 | 输出 |
|------|------|------|------|
| FileScanner | 递归扫描目录，过滤文件 | 目录路径，扩展名，忽略模式 | ScannedFile 迭代器 |
| AST_Parser | Tree-sitter 解析代码结构 | 代码内容，语言 | ASTNode 列表 |
| Chunker | 将代码分割为语义块 | ScannedFile，ASTNode | CodeChunk 列表 |
| Tokenizer | Token 计数和截断 | 文本 | Token 数量 |
| EmbeddingClient | 调用 API 生成向量 | 文本批次 | 向量列表 |
| VectorStore | Qdrant 存储和检索 | 向量，元数据 | SearchResult |

## Data Flow

```
目录路径 → FileScanner → ScannedFile
                ↓
         AST_Parser → ASTNode
                ↓
           Chunker → CodeChunk
                ↓
      EmbeddingClient → Vector
                ↓
        VectorStore → 持久化存储

查询 → EmbeddingClient → Vector → VectorStore → SearchResult
```

## Key Data Models

### CodeChunk
```python
@dataclass
class CodeChunk:
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    content: str
    language: str
    chunk_type: str  # 'function', 'class', 'fixed'
    metadata: dict   # parent_class, function_name, imports
```

### SearchResult
```python
@dataclass
class SearchResult:
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    content: str
    score: float
    metadata: dict
```

## Error Handling Strategy

| 错误类型 | 处理策略 |
|----------|----------|
| 文件访问错误 | 记录日志，跳过，继续处理 |
| 解析错误 | 回退到固定大小分块 |
| API 错误 | 指数退避重试（最多3次） |
| 存储错误 | 重试后失败则报告 |

## Testing Strategy

- **属性测试**: Hypothesis 框架，每属性 ≥100 次迭代
- **单元测试**: pytest，覆盖率 ≥80%
- **集成测试**: Docker Qdrant 实例

## Component Dependencies

```
IndexingService
├── FileScannerInterface
├── ASTParserInterface
├── ChunkerInterface
│   └── TokenizerInterface
├── EmbeddingClientInterface
└── VectorStoreInterface

SearchService
├── EmbeddingClientInterface
├── VectorStoreInterface
└── RerankerInterface (可选)

EvaluationService
├── SearchService
└── EvaluationDatasetLoader
```

## Key Sequences

### 索引流程
```
1. CLI 调用 IndexingService.index_directory(path)
2. FileScanner.scan(path) → 生成 ScannedFile 流
3. 使用 ProcessPoolExecutor 并行处理文件（CPU密集型）:
   a. AST_Parser.parse(content, language) → ASTNode[]
   b. Chunker.chunk(file, nodes) → CodeChunk[]
4. 主线程收集 CodeChunk，批量处理（batch_size）
5. 使用 asyncio 调用 EmbeddingClient.embed_batch(texts) → Vector[]（IO密集型）
6. 使用 asyncio 调用 VectorStore.upsert(chunk_id, vector, payload)
7. 更新 IndexMetadata（SQLite）
```

### 并行策略说明                                                                                                                                                                          
- **CPU密集型操作**（文件扫描、AST解析、分块）: 使用 `ProcessPoolExecutor` + `max_workers`                                                                                                
- **Worker 初始化**: 使用 `executor(initializer=_init_worker)` 模式，在每个子进程启动时初始化全局 `TreeSitterParser` 和 `Chunker` 实例。这解决了 `tree-sitter` C扩展对象无法序列化（pickle）的问题，并避免了重复创建 Parser 的开销。
- **IO密集型操作**（API调用、数据库操作）: 使用 `asyncio` + `aiohttp`                                                                                                                     
- **协调方式**: 主进程使用 `asyncio.run()` 驱动，CPU任务通过 `loop.run_in_executor()` 桥接
### 搜索流程
```
1. CLI 调用 SearchService.search(query, limit, filter, use_rerank)
2. EmbeddingClient.embed_batch([query]) → query_vector
3. VectorStore.search(query_vector, recall_limit, filter) → candidates[]
   - recall_limit = limit * 5 (启用 rerank 时)
4. [可选] Reranker.rerank(query, candidates, limit) → reranked[]
5. 返回最终结果（Top-K）
```

**Re-ranking 策略:**
- 默认启用（如果配置了 Reranker）
- 向量召回 Top-50 → Cross-Encoder 精排 → Top-10
- 本地运行，无额外 API 成本

### 增量更新流程
```
1. CLI 调用 IndexingService.update_incremental(path)
2. FileScanner.scan(path) → 当前文件列表
3. 对比 IndexMetadata.file_hashes:
   a. 新文件: 索引并添加
   b. 修改文件: VectorStore.delete_by_file() → 重新索引
   c. 删除文件: VectorStore.delete_by_file()
4. 更新 IndexMetadata
```

## Related Documents

- [design-detail.md](./design-detail.md) - 详细接口定义、数据模型 Schema
- [design-properties.md](./design-properties.md) - 正确性属性详细说明
- [requirements.md](./requirements.md) - 需求规格

## Correctness Properties Summary

*完整属性定义见 [design-properties.md](./design-properties.md)*

| 类别 | 属性数量 | 覆盖需求 |
|------|----------|----------|
| 文件扫描与分块 | 9 | Req 1.1, 1.3, 1.4, 2.2-2.6 |
| 向量存储与检索 | 8 | Req 3.1, 3.3-3.4, 4.2-4.5 |
| 增量更新 | 5 | Req 5.1-5.3, 5.5 |
| 配置与评估 | 4 | Req 6.1, 7.5, 9.2-9.3 |
| **总计** | **26** | |

*包含 Re-ranking、智能拆分、元数据查询等扩展属性*
