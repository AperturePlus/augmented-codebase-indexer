# Project ACI 代码诊断

## 发现的问题
- **CLI 入口缺失（需求 8 全部不可用）**：`pyproject.toml:38` 将可执行脚本指向 `aci.cli:app`，但 `src/aci/cli/__init__.py:1` 只有注释，没有 Typer 应用或任何命令，运行 `aci` 会直接导入失败。
- **服务层未实现（需求 1/2/4/5/6/9 未覆盖）**：设计文档要求的 `IndexingService`、`SearchService`、`EvaluationService` 等均不存在，`src/aci/services/__init__.py:1` 只有注释，仓库内也找不到任何对应类或并行/进度/评估逻辑，导致扫描、分块、嵌入、搜索、增量更新、评测流程都无法编排。
- **基础设施层缺失（需求 3/4 无法满足）**：`src/aci/infrastructure/__init__.py:1` 仅有占位注释，没有 EmbeddingClient（批处理、退避重试）、VectorStore（Qdrant upsert/search/delete）或连接池实现，设计细节和正确性属性 9/10/11/12/13/14/14a/14b 全部缺席。
- **增量索引与元数据存储缺位（需求 5.1-5.5 和设计 5.1-5.3 未落地）**：代码库没有 IndexMetadataStore/SQLite、文件哈希对比、删除/更新旧 chunk 的逻辑，也没有变更检测或统计更新。当前仅在 `FileScanner` 计算 hash，后续无人使用，无法实现删除、增量添加或统计一致性。
- **配置模型与设计规范不一致（需求 7.1/7.2/7.5 偏差）**：`src/aci/core/config.py:55` 定义的配置缺少设计文档要求的 `embedding_dimension`、`qdrant_api_key`/`ACI_QDRANT_API_KEY` 等敏感项映射，VectorStore 使用 `host/port` 而非设计的 `qdrant_url`。缺失字段会导致与外部依赖的配置串联失败，也无法满足环境变量覆盖敏感信息的要求。
- **超长分块策略未实现设计中的语法友好拆分**：设计-detail 定义的 SmartChunkSplitter 应优先在空行/语句边界拆分并保留上下文。现有 `_split_oversized_node`（`src/aci/core/chunker.py:349`）只按逐行 token 计数硬切，未利用 AST、未加上下文前缀，无法满足正确性属性 7a（避免破坏语法结构）以及需求 2.5 的质量要求。
