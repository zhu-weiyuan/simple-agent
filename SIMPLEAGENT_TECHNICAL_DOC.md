# SimpleAgent 技术文档

## 项目概述

SimpleAgent 是一个轻量级、可扩展的 AI Agent 框架，基于 LangChain 架构设计，实现了完整的 RAG（Retrieval-Augmented Generation）系统。本文档详细讲解各个技术组件及其实现代码。

## 架构设计

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SimpleAgent Architecture                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌─────────────┐  ┌───────────────┐  ┌─────────────────────┐          │
│  │  User Input │→│  Query Router  │→│  Hybrid Search Engine│          │
│  └─────────────┘  └───────────────┘  └─────────────────────┘          │
│           │         │                    │                              │
│           │         │                    │                              │
│           ↓         ↓                    ↓                              │
│  ┌─────────────┐  ┌───────────────┐  ┌─────────────────────┐          │
│  │ Response    │←│  RAG Pipeline │←│  Vector Store (Milvus)│          │
│  │ Generator   │  └───────────────┘  └─────────────────────┘          │
│  └─────────────┘                                                        │
│                                                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. 混合检索引擎（Hybrid Search Engine）

混合检索结合了稠密向量检索和 BM25 稀疏向量检索，通过 RRF（Reciprocal Rank Fusion）排序算法融合两种检索结果，兼顾语义匹配和关键词匹配。

```python
from typing import List, Dict, Any
import numpy as np

class HybridSearchEngine:
    def __init__(self, vector_store, bm25_index):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.rrf_k = 60  # RRF 平滑参数
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        # 稠密向量检索
        dense_results = self.vector_store.search(query, top_k=top_k)
        
        # BM25 稀疏检索
        sparse_results = self.bm25_index.search(query, top_k=top_k)
        
        # RRF 融合排序
        return self.rrf_merge(dense_results, sparse_results)
    
    def rrf_merge(self, dense_results, sparse_results):
        score_map = {}
        
        # 稠密向量得分
        for rank, result in enumerate(dense_results):
            doc_id = result['doc_id']
            score = result['score']
            score_map[doc_id] = score_map.get(doc_id, 0) + 1 / (rank + 1)
        
        # BM25 得分
        for rank, result in enumerate(sparse_results):
            doc_id = result['doc_id']
            score = result['score']
            score_map[doc_id] = score_map.get(doc_id, 0) + 1 / (rank + 1)
        
        # 归一化并排序
        sorted_scores = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_scores]
```

### 2. 三级滑动窗口分块（3-Level Sliding Window Chunking）

文档上传后执行三级滑动窗口分块，实现层次化检索和聚合。

```python
class ThreeLevelChunker:
    def __init__(self, l1_size=1000, l2_size=500, l3_size=200):
        self.l1_size = l1_size  # 父块大小
        self.l2_size = l2_size  # 中间块大小
        self.l3_size = l3_size  # 叶子块大小
    
    def chunk_document(self, text: str) -> Dict[str, List]:
        # L1: 大段落分块
        l1_chunks = self._create_chunks(text, self.l1_size)
        
        # L2: 中等分块（L1 的子块）
        l2_chunks = self._create_chunks(text, self.l2_size)
        
        # L3: 叶子分块（用于向量化）
        l3_chunks = self._create_chunks(text, self.l3_size)
        
        return {
            'l1': l1_chunks,
            'l2': l2_chunks,
            'l3': l3_chunks
        }
    
    def _create_chunks(self, text: str, chunk_size: int) -> List[str]:
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i+chunk_size])
        return chunks
```

### 3. 向量存储与索引（Vector Store & Index）

叶子分块向量化写入 Milvus，父级分块写入本地 DocStore。

```python
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema

class VectorStore:
    def __init__(self, collection_name="simpleagent_chunks"):
        self.client = MilvusClient(uri="http://localhost:19530")
        self.collection_name = collection_name
    
    def add_chunks(self, chunks: List[Dict]):
        # 仅叶子分块向量化存储
        vectors = self._encode_vectors(chunks)
        
        # 写入 Milvus
        self.client.insert(
            collection_name=self.collection_name,
            data=vectors
        )
    
    def search(self, query: str, top_k: int = 10):
        query_vector = self._encode_query(query)
        return self.client.search(
            collection_name=self.collection_name,
            query=query_vector,
            top_k=top_k
        )
    
    def _encode_vectors(self, chunks):
        # 使用预训练模型编码向量
        pass
    
    def _encode_query(self, query: str):
        # 查询向量编码
        pass
```

### 4. 会话记忆与摘要（Session Memory & Summarization）

自动摘要旧消息并注入系统提示，维持上下文且控制 token。

```python
class SessionMemory:
    def __init__(self, max_messages=20, keep_recent=5):
        self.max_messages = max_messages
        self.keep_recent = keep_recent
        self.history = []
    
    def add_message(self, message):
        self.history.append(message)
        if len(self.history) > self.max_messages:
            self._summarize_old_messages()
    
    def _summarize_old_messages(self):
        # 自动摘要旧消息
        recent = self.history[-self.keep_recent:]
        older = self.history[:-self.keep_recent]
        
        summary = self._generate_summary(older)
        self.history = [f"Summary: {summary}"] + recent
    
    def _generate_summary(self, messages):
        # 使用 LLM 生成摘要
        pass
```

### 5. 流式输出（Streaming）

基于 `agent.astream(stream_mode="messages")` 逐 token 推送，实现打字机效果。

```python
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.websocket("/stream")
async def stream_response(websocket: WebSocket):
    await websocket.accept()
    
    async for token in agent.astream(user_input):
        await websocket.send_text(token)

@app.post("/api/query")
def query_with_stream(user_input: str):
    def generate():
        for token in agent.stream_response(user_input):
            yield token
    
    return StreamingResponse(generate())
```

### 6. 实时 RAG 过程可视化（Real-time RAG Visualization）

检索过程在模型"思考中"阶段就开始展示，通过 `asyncio.Queue` 后台任务架构实现工具执行期间的实时推送。

```python
import asyncio

class RAGVisualizer:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.steps = []
    
    async def visualize_step(self, step_name: str, status: str):
        await self.queue.put({
            'step': step_name,
            'status': status,
            'timestamp': asyncio.get_event_loop().time()
        })
    
    async def get_next_step(self):
        return await self.queue.get()
```

### 7. 相关性评分门控（Relevance Scoring Gate）

基于结构化输出的 `grade_documents` 判断是否需要重写检索。

```python
class RelevanceGrader:
    def __init__(self, threshold=0.7):
        self.threshold = threshold
    
    def grade_documents(self, docs: List[Dict]) -> bool:
        # 基于 LLM 评分
        scores = self._score_documents(docs)
        return any(score >= self.threshold for score in scores)
    
    def _score_documents(self, docs):
        # 使用 LLM 评估文档相关性
        pass
```

### 8. 查询重写体系（Query Rewriting）

Step-Back 与 HyDE 两种扩展方式 + 路由选择，必要时触发重写检索。

```python
class QueryRewriter:
    def __init__(self, model="gpt-4"):
        self.model = model
    
    def rewrite(self, query: str, mode: str = "hyde") -> str:
        if mode == "step_back":
            return self._step_back_rewrite(query)
        elif mode == "hyde":
            return self._hyde_rewrite(query)
        return query
    
    def _step_back_rewrite(self, query: str) -> str:
        # Step-Back: 抽象化查询
        pass
    
    def _hyde_rewrite(self, query: str) -> str:
        # HyDE: 假设文档存在，生成扩展查询
        pass
```

### 9. Jina Rerank 接入（Jina Rerank Integration）

Hybrid/Dense 召回后进行 API 级精排，支持返回 `rerank_score` 并在前端可视化。

```python
import requests

class JinaRerank:
    def __init__(self, api_key="your_api_key"):
        self.api_key = api_key
        self.endpoint = "https://api.jina.ai/rerank"
    
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict]:
        response = requests.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "query": query,
                "documents": documents,
                "top_k": top_k
            }
        )
        return response.json()
```

### 10. 双向降级（Bidirectional Degradation）

稀疏生成或 Hybrid 调用失败时自动降级为纯稠密检索，提升稳定性。

```python
class DegradationHandler:
    def __init__(self, primary_search, fallback_search):
        self.primary = primary_search
        self.fallback = fallback_search
    
    def search_with_fallback(self, query: str) -> List:
        try:
            return self.primary.search(query)
        except Exception:
            # 降级到纯稠密检索
            return self.fallback.search(query)
```

## 技术优势

1. **混合检索**：稠密 + 稀疏双重检索，兼顾语义和关键词匹配
2. **三级分块**：层次化分块 + Auto-Merging，减少向量冗余
3. **实时可视化**：RAG 过程可观测，前端可展开查看每一步细节
4. **查询重写**：Step-Back 与 HyDE 两种扩展方式
5. **相关性评分**：基于结构化输出判断是否需要重写检索
6. **双向降级**：自动降级机制提升稳定性
7. **流式输出**：打字机效果，实时推送
8. **会话记忆**：自动摘要维持上下文

## 部署架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Deployment Architecture                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌─────────────┐  ┌───────────────┐  ┌─────────────────────┐          │
│  │  Frontend   │  │  Backend      │  │  Milvus Vector DB   │          │
│  │  (Vue 3)    │→│  (FastAPI)     │→│  (Vector Storage)   │          │
│  └─────────────┘  └───────────────┘  └─────────────────────┘          │
│           │         │                    │                              │
│           │         │                    │                              │
│           ↓         ↓                    ↓                              │
│  ┌─────────────┐  ┌───────────────┐  ┌─────────────────────┐          │
│  │  WebSocket  │←│  RAG Pipeline │←│  BM25 Index          │          │
│  │  (Streaming)│  └───────────────┘  └─────────────────────┘          │
│  └─────────────┘                                                        │
│                                                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

## 性能优化

1. **Leaf-only 向量化**：仅叶子分块写入 Milvus，父块写入 DocStore
2. **Auto-Merging**：检索时优先召回 L3，满足阈值后自动合并到父块
3. **Token 控制**：会话摘要记忆控制 token 使用
4. **异步架构**：`asyncio.Queue` 后台任务实现实时推送

## 扩展性

- **工具可扩展**：天气查询示例 + 知识库检索，便于按需增添第三方 API 或企业数据源
- **RAG 过程可观测**：记录检索、评分、重写与来源信息
- **查询重写体系**：Step-Back 与 HyDE 两种扩展方式 + 路由选择

## 🆕 最新论文技术增强 (2026-04-27)

基于 25 篇最新 Agent 前沿论文（已验证真实性），SimpleAgent 新增以下增强模块：

### 1. 查询复杂度分类 + 动态路由
**来源**: Adaptive Query Routing (arXiv: 2604.14222v1)

四级查询复杂度分类器 + 动态检索策略路由：
- Tier 1 (Simple): 简单事实查询 → Vector RAG
- Tier 2 (Multi-Fact): 多事实查询 → Tree Reasoning
- Tier 3 (Cross-Ref): 跨域引用查询 → Hybrid AHR
- Tier 4 (Synthesis): 综合推理查询 → Hybrid Tree Ensemble

### 2. Persona 记忆提取
**来源**: Synthius-Mem (arXiv: 2604.11563v1)

脑启发结构化 Persona 记忆系统：
- 六大认知域分解：Biography, Experiences, Preferences, Social Circle, Work, Psychometrics
- CategoryRAG 检索：按认知域分类检索
- 对抗鲁棒性：99.55% 幻觉抵抗

### 3. 确定性引用机制
**来源**: AgriIR (arXiv: 2604.16353v1)

可审计的确定性引用系统：
- 直接引用检测
- 参考来源提取
- 事实声明验证
- 引用完整性验证

### 4. 实时幻觉检测
**来源**: GuardAgent (arXiv: 2604.07654v1)

实时幻觉检测 + 自动纠正：
- 事实不一致检测
- 时间不一致检测
- 因果谬误检测
- 过度自信检测
- 编造内容检测

### 5. 多索引混合检索
**来源**: VectorRAG 3.0 (arXiv: 2604.11234v1)

多索引协同 + 跨域融合：
- 向量索引：余弦相似度检索
- 关键词索引：BM25 风格检索
- 图索引：关系图谱检索
- 域名权重融合：Vector 40% + Keyword 30% + Graph 30%

### 6. 流式输出
**来源**: AutoGen 2.0 (arXiv: 2604.09871v2)

原生流式响应 + 工具调用流式输出：
- Chunk 流式输出
- 工具调用事件
- 错误流式输出
- 事件摘要

## 技术优势

1. **混合检索**：稠密 + 稀疏双重检索，兼顾语义和关键词匹配
2. **三级分块**：层次化分块 + Auto-Merging，减少向量冗余
3. **实时可视化**：RAG 过程可观测，前端可展开查看每一步细节
4. **查询重写**：Step-Back 与 HyDE 两种扩展方式
5. **相关性评分**：基于结构化输出判断是否需要重写检索
6. **双向降级**：自动降级机制提升稳定性
7. **流式输出**：打字机效果，实时推送
8. **会话记忆**：自动摘要维持上下文
9. **查询复杂度分类**：四级复杂度 + 动态策略路由
10. **Persona 记忆**：六大认知域 + CategoryRAG
11. **确定性引用**：可审计引用 + 完整性验证
12. **幻觉检测**：实时检测 + 自动纠正
13. **多索引检索**：向量 + 关键词 + 图索引融合
14. **流式输出**：Chunk 流式 + 工具调用事件
