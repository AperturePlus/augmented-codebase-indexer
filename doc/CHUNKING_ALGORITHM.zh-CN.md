# Chunking 算法原理（当前实现）

本文基于 `src/aci/core/chunker` 的当前代码实现，说明 ACI 在索引阶段如何把源码切分为可检索片段（chunks）。

## 1. 总体流程

`Chunker.chunk(file, ast_nodes)` 的主流程：

1. 先按语言抽取 import 列表（写入每个 chunk 的 metadata）。
2. 若存在 AST 节点：走 **语义切分（AST-based）**。
3. 若无 AST 节点：走 **固定行数切分（fixed-size fallback）**。
4. 若配置了 `summary_generator`，并行产出 function/class/file summary artifact。

## 2. AST 语义切分（优先路径）

当解析器能产出 AST 节点时：

- 每个 AST 节点（`function/class/method`）默认作为一个 chunk 候选。
- metadata 会补充结构化信息：
  - `function_name`
  - `class_name`
  - `parent_class`（method 场景）
  - `imports`、`file_hash`、`language`
- 若节点有 docstring，会先规范化，再以分隔符拼到 chunk 内容前缀，提高语义可检索性。

### Token 上限控制

对每个候选节点：

- `token_count <= max_tokens`：直接生成单个 chunk。
- `token_count > max_tokens`：交给 `SmartChunkSplitter` 做智能拆分。

## 3. SmartChunkSplitter 智能拆分策略

目标：在 token 约束下尽量不破坏代码语法/语义边界。

### 3.1 拆分优先级

在一个超大节点内部，优先在这些位置切分：

1. 空行
2. 语句边界（`def/class/if/for/while/try/except/return/...` 模式）
3. 缩进较低的行（块边界）
4. 实在不行按可容纳最大范围切

### 3.2 如何找“可容纳最大范围”

- 通过二分法 `_find_max_end_index` 找从 `start_idx` 开始，token 不超限的最远 `end_idx`。
- 再在 `[start_idx, end_idx]` 区间回溯挑“最佳切点”。

### 3.3 上下文补偿

拆分后会给后续子块加上下文前缀，避免脱离语境：

- 方法：`# Context: class <Parent>`
- 函数：`# Context: function <Name>`
- 类：`# Context: class <Name>`

此外：

- docstring 前缀只附加在首个子块。
- metadata 里标记 `is_partial / part_index / total_parts` 等字段。

## 4. 固定行数切分（fallback）

当某语言暂不支持 AST（或 AST 为空）时：

- 以 `fixed_chunk_lines`（默认 50 行）分块。
- 相邻块保留 `overlap_lines`（默认 5 行）重叠，降低跨块语义断裂。
- 每块仍会做 token 校验；若超限，持续从块尾减行直到不超限（至少保留 1 行）。
- chunk 类型标记为 `fixed`。

## 5. Import 抽取策略

chunking 前会先提取 import，并写入 metadata：

- Python：识别 `import ...` / `from ...`
- JS/TS：识别 `import ...` 和 `const ... require(...)`
- Go：支持 `import (...)` 块和单行 import
- 其他语言：空实现（返回空列表）

这让检索和后续总结模型可利用依赖上下文。

## 6. 输出数据形态

最终 `ChunkingResult` 包含两类产物：

- `chunks: list[CodeChunk]`
- `summaries: list[SummaryArtifact]`

其中 `CodeChunk` 是索引主对象，带有：

- 行号范围（1-based，含结束行）
- 原始/拆分后的内容
- chunk 类型（`function/class/method/fixed`）
- metadata（含 imports、符号名、分片标记等）

## 7. 设计取舍总结

当前算法是“**语义优先 + token 兜底 + 行切分回退**”的混合方案：

- 优点：
  - 尽量对齐语言结构（函数/类/方法），检索粒度更自然。
  - 超大节点可智能拆分，并保留上下文，降低语义损失。
  - 对不支持 AST 的语言仍可工作（工程可用性高）。
- 潜在限制：
  - 语句边界模式目前偏 Python 风格正则，对其他语言并非完全精确。
  - 固定切分路径主要靠行数和重叠，语义一致性弱于 AST 路径。

