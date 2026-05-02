# -*- coding: utf-8 -*-
"""
SimpleAgent Web 应用
提供 REST API 和 Web 界面来测试所有增强模块
"""
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import json
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from my_agent.enhanced import (
    QueryComplexityClassifier,
    DynamicRouter,
    QueryTier,
    RetrievalStrategy,
    PersonaExtractor,
    PersonaMemory,
    CategoryRAG,
    CognitiveDomain,
    HallucinationDetector,
    DeterministicCitation,
    MultiIndexRetrieval,
    VectorIndex,
    KeywordIndex,
    GraphIndex,
    StreamingOutput,
)

app = FastAPI(title="SimpleAgent Enhanced Web")

# 创建静态文件目录并挂载
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# 全局实例
router = DynamicRouter()
persona_memory = PersonaMemory()
persona_extractor = PersonaExtractor()
category_rag = CategoryRAG(persona_memory)
hallucination_detector = HallucinationDetector()
citation_system = DeterministicCitation()
multi_index = MultiIndexRetrieval()
streaming_output = StreamingOutput()

# 对话历史存储
chat_history_store: Dict[str, List[Dict[str, str]]] = {}

# 请求模型
class ChatRequest(BaseModel):
    message: str
    chatId: str = "default"
    stream: bool = False  # 是否启用流式输出

class TextRequest(BaseModel):
    """支持 message 或 text 字段的通用请求模型"""
    message: str | None = None
    text: str | None = None
    
    def get_text(self) -> str:
        return self.message or self.text or ""


# 聊天 API 端点（集成增强模块 + 流式输出）
@app.post("/api/chat")
async def chat(request: ChatRequest):
    """聊天 API - 调用本地模型 + 增强模块 + 流式输出"""
    try:
        from openai import OpenAI
        import os
        
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "your_key_here"),
            base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8080/v1")
        )
        
        if request.stream:
            # ── 流式输出模式 ──
            async def event_generator():
                try:
                    # 1. 构建带上下文的对话历史
                    chat_history = chat_history_store.get(request.chatId, [])
                    messages = [
                        {"role": "system", "content": (
                            "你是一个聪明、友好、有帮助的AI助手。请用中文回答用户的问题。"
                            "你的回答应该："
                            "1. 简洁明了，直接给出有用的回答；"
                            "2. 如果问题不清楚，主动询问细节；"
                            "3. 如果不确定，诚实地说不知道；"
                            "4. 使用自然的语气，像一个真正的朋友在聊天；"
                            "5. 适当使用emoji让对话更生动。"
                        )},
                    ]
                    # 加入最近的对话历史（最多6条用户/助手对话）
                    for msg in chat_history[-6:]:
                        messages.append({"role": msg["role"], "content": msg["content"]})
                    
                    # 2. 发送查询路由分析
                    enhanced_data = {}
                    try:
                        from my_agent.enhanced import (
                            DynamicRouter, PersonaExtractor, PersonaMemory,
                            CategoryRAG, HallucinationDetector, DeterministicCitation,
                            MultiIndexRetrieval
                        )
                        router = DynamicRouter()
                        analysis = router.route_query(request.message)
                        enhanced_data["router"] = {
                            "tier": analysis.tier.value,
                            "strategy": analysis.strategy.value,
                            "confidence": analysis.confidence,
                        }
                        
                        extractor = PersonaExtractor()
                        memory = PersonaMemory()
                        category_rag = CategoryRAG(memory)
                        facts = extractor.extract_facts(request.message, "chat")
                        for f in facts:
                            memory.add_fact(f)
                        if facts:
                            enhanced_data["persona"] = [f.fact for f in facts]
                    except Exception:
                        pass
                    
                    yield f"event: start\ndata: {json.dumps({'enhanced': enhanced_data}, ensure_ascii=False)}\n\n"
                    
                    # 流式获取 LLM 回复
                    stream = client.chat.completions.create(
                        model=os.getenv("OPENAI_MODEL", "E:/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"),
                        messages=messages,
                        temperature=0.5,
                        max_tokens=4096,
                        stream=True,
                    )
                    
                    full_reply = ""
                    for chunk in stream:
                        delta = chunk.choices[0].delta if chunk.choices else None
                        content = getattr(delta, 'content', None)
                        if content:
                            full_reply += content
                            yield f"event: chunk\ndata: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"
                    
                    # 后处理：去掉 <end_of_turn>，运行增强模块
                    full_reply = full_reply.replace("<end_of_turn>", "").strip()
                    
                    # 幻觉检测
                    enhanced_data["hallucination"] = {}
                    try:
                        hall_detector = HallucinationDetector()
                        hall_result = hall_detector.detect(full_reply)
                        enhanced_data["hallucination"] = {
                            "is_hallucination": hall_result.is_hallucination,
                            "confidence": hall_result.confidence,
                            "hallucination_type": hall_result.hallucination_type,
                            "correction_suggestion": hall_result.correction_suggestion,
                        }
                    except Exception:
                        pass
                    
                    # 引用提取
                    enhanced_data["citation"] = {"has_citation": False, "citations": []}
                    try:
                        cit_system = DeterministicCitation()
                        cit_result = cit_system.extract_citations(full_reply)
                        enhanced_data["citation"] = {
                            "has_citation": cit_result.has_citation,
                            "citations": [
                                {"source": c.source, "content": c.content, "confidence": c.confidence}
                                for c in cit_result.citations
                            ],
                        }
                    except Exception:
                        pass
                    
                    # 更新对话历史
                    if chat_history is not None:
                        chat_history.append({"role": "assistant", "content": full_reply})
                        chat_history_store[request.chatId] = chat_history
                    
                    # 发送完成信号
                    yield f"event: done\ndata: {json.dumps({'reply': full_reply, 'enhanced': enhanced_data}, ensure_ascii=False)}\n\n"
                    
                except Exception as e:
                    yield f"event: error\ndata: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
            
            from fastapi.responses import StreamingResponse
            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
            )
        
        else:
            # ── 传统模式（非流式）──
            chat_history = chat_history_store.get(request.chatId, [])
            messages = [
                {"role": "system", "content": (
                    "你是一个聪明、友好、有帮助的AI助手。请用中文回答用户的问题。"
                    "你的回答应该："
                    "1. 简洁明了，直接给出有用的回答；"
                    "2. 如果问题不清楚，主动询问细节；"
                    "3. 如果不确定，诚实地说不知道；"
                    "4. 使用自然的语气，像一个真正的朋友在聊天；"
                    "5. 适当使用emoji让对话更生动。"
                    "6. 如果用户之前问过相关问题，要参考之前的回答。"
                )},
            ]
            # 加入最近的对话历史（最多6条用户/助手对话）
            for msg in chat_history[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
            
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "E:/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"),
                messages=messages,
                temperature=0.5,
                max_tokens=4096,
            )
            
            reply = (response.choices[0].message.content or "无回复").replace("<end_of_turn>", "").strip()
            
            # 更新对话历史
            chat_history.append({"role": "assistant", "content": reply})
            chat_history_store[request.chatId] = chat_history
            
            # ── 运行增强模块分析 ──
            enhanced = {}
            try:
                from my_agent.enhanced import (
                    DynamicRouter, PersonaExtractor, PersonaMemory,
                    CategoryRAG, HallucinationDetector, DeterministicCitation,
                    MultiIndexRetrieval
                )
                
                # 1. 查询路由
                router = DynamicRouter()
                analysis = router.route_query(request.message)
                enhanced["router"] = {
                    "tier": analysis.tier.value,
                    "strategy": analysis.strategy.value,
                    "confidence": analysis.confidence,
                }
                
                # 2. Persona 记忆
                extractor = PersonaExtractor()
                memory = PersonaMemory()
                category_rag = CategoryRAG(memory)
                facts = extractor.extract_facts(request.message, "chat")
                for f in facts:
                    memory.add_fact(f)
                if facts:
                    enhanced["persona"] = [f.fact for f in facts]
                
                # 3. 幻觉检测
                hall_detector = HallucinationDetector()
                hall_result = hall_detector.detect(reply)
                enhanced["hallucination"] = {
                    "is_hallucination": hall_result.is_hallucination,
                    "confidence": hall_result.confidence,
                    "hallucination_type": hall_result.hallucination_type,
                    "correction_suggestion": hall_result.correction_suggestion,
                }
                
                # 4. 引用提取
                cit_system = DeterministicCitation()
                cit_result = cit_system.extract_citations(reply)
                enhanced["citation"] = {
                    "has_citation": cit_result.has_citation,
                    "citations": [
                        {"source": c.source, "content": c.content, "confidence": c.confidence}
                        for c in cit_result.citations
                    ],
                }
                
                # 5. 多索引检索
                multi_idx = MultiIndexRetrieval()
                results = multi_idx.search(request.message, top_k=3)
                if results:
                    enhanced["retrieval"] = [
                        {"domain": r.domain, "score": r.score, "content": r.document.content}
                        for r in results
                    ]
            except Exception:
                pass
            
            return {"reply": reply, "enhanced": enhanced}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────
#  增强模块独立 API 端点
# ──────────────────────────────────────

# ── 查询复杂度分类 ──
@app.post("/api/query/classify")
async def classify_query_endpoint(req: TextRequest):
    analysis = router.route_query(req.get_text())
    indicators = []
    try:
        cls = QueryComplexityClassifier()
        indicators = cls.get_complexity_indicators(req.message)
    except Exception:
        pass
    return {
        "tier": analysis.tier.value,
        "strategy": analysis.strategy.value,
        "confidence": analysis.confidence,
        "complexity_score": analysis.confidence,
        "indicators": indicators,
    }


# ── Persona 记忆提取 ──
@app.post("/api/persona/extract")
async def extract_persona_endpoint(req: TextRequest):
    facts = persona_extractor.extract_facts(req.get_text(), "chat")
    return {
        "facts": [
            {
                "domain": f.domain.value,
                "fact": f.fact,
                "confidence": f.confidence,
            }
            for f in facts
        ]
    }


# ── 幻觉检测 ──
@app.post("/api/hallucination/detect")
async def detect_hallucination_endpoint(req: TextRequest):
    result = hallucination_detector.detect(req.get_text())
    return {
        "is_hallucination": result.is_hallucination,
        "confidence": result.confidence,
        "hallucination_type": result.hallucination_type,
        "correction_suggestion": result.correction_suggestion,
        "evidence": result.evidence if hasattr(result, "evidence") else [],
    }


# ── 引用提取 ──
@app.post("/api/citation/extract")
async def extract_citation_endpoint(req: TextRequest):
    result = citation_system.extract_citations(req.get_text())
    return {
        "has_citation": result.has_citation,
        "citations": [
            {
                "source": c.source,
                "content": c.content,
                "confidence": c.confidence,
            }
            for c in (result.citations if hasattr(result, "citations") else [])
        ],
    }


# ── 多索引检索 ──
@app.post("/api/multi-index/search")
async def search_multi_index_endpoint(req: TextRequest):
    results = multi_index.search(req.get_text(), top_k=5)
    return {
        "query": req.message,
        "results": [
            {
                "domain": r.domain,
                "score": r.score,
                "content": r.document.content,
            }
            for r in (results if results else [])
        ],
    }


# ── 主页 ──
@app.get("/", response_class=HTMLResponse)
async def homepage():
    return FileResponse(str(static_dir / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
