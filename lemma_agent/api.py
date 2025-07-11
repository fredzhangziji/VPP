#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LEMMA API: 电力市场分析智能体API服务

基于FastAPI框架封装LEMMA智能体，提供HTTP API接口
"""

import json
import logging
import uuid
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from qwen_agent.llm.base import Message
from qwen_agent.agents import Assistant

# 导入配置和工具函数
from config import LLM_CONFIG, TOOLS, SYSTEM_MESSAGE, API_SESSION_CLEANUP_INTERVAL_SECONDS, API_SESSION_MAX_AGE_HOURS
from utils import extract_content, Colors

# 导入lemma_agent以触发工具注册
import lemma_agent

# 配置API日志
api_logger = logging.getLogger("lemma_api")
api_logger.setLevel(logging.INFO)
handler = logging.FileHandler("lemma_api.log")
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
api_logger.addHandler(handler)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    api_logger.info("应用启动，开始后台会话清理任务...")
    cleanup_task = asyncio.create_task(periodic_cleanup())
    yield
    api_logger.info("应用关闭，正在取消后台任务...")
    cleanup_task.cancel()

# 创建FastAPI应用
app = FastAPI(
    title="LEMMA API",
    description="电力市场分析智能体API服务",
    version="1.0.0",
    lifespan=lifespan,
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境建议限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 会话存储
active_sessions: Dict[str, Dict[str, Any]] = {}

# 请求模型
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str

# 初始化Agent实例
def initialize_agent():
    """初始化并返回一个新的Agent实例"""
    try:
        api_logger.info("初始化新的LEMMA Agent实例")
        agent = Assistant(
            llm=LLM_CONFIG,
            system_message=SYSTEM_MESSAGE,
            function_list=TOOLS
        )
        return agent
    except Exception as e:
        api_logger.error(f"Agent初始化失败: {e}", exc_info=True)
        raise RuntimeError(f"Agent初始化失败: {e}")

# 创建新会话或获取现有会话
def get_session(session_id: Optional[str] = None) -> tuple:
    """
    创建新会话或获取现有会话
    
    Args:
        session_id: 可选的会话ID
        
    Returns:
        tuple: (session_id, session_data)
    """
    if not session_id or session_id not in active_sessions:
        # 创建新会话
        new_session_id = session_id or str(uuid.uuid4())
        api_logger.info(f"创建新会话: {new_session_id}")
        
        session_data = {
            "agent": initialize_agent(),
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "sidecar_data": {}  # 为会话添加边车数据存储
        }
        active_sessions[new_session_id] = session_data
        return new_session_id, session_data
    
    # 获取现有会话
    session_data = active_sessions[session_id]
    session_data["last_active"] = datetime.now().isoformat()
    return session_id, session_data

# 清理过期会话
def cleanup_sessions():
    """清理超过指定时间的会话"""
    current_time = datetime.now()
    max_age = timedelta(hours=API_SESSION_MAX_AGE_HOURS)
    expired_sessions = []
    
    for session_id, session_data in active_sessions.items():
        last_active = datetime.fromisoformat(session_data["last_active"])
        if current_time - last_active > max_age:
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        api_logger.info(f"清理过期会话: {session_id}")
        del active_sessions[session_id]

# 后台任务，用于定期清理会话
async def periodic_cleanup():
    """后台任务，定期运行清理函数"""
    while True:
        await asyncio.sleep(API_SESSION_CLEANUP_INTERVAL_SECONDS)
        api_logger.info("开始执行定期会话清理任务...")
        cleanup_sessions()

@app.get("/")
async def root():
    return {"message": "LEMMA电力市场分析智能体API服务"}

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    处理用户对话请求并以流式方式返回响应
    
    返回格式为Server-Sent Events (SSE)，包含思考过程和回答内容的实时更新
    """
    try:
        # 获取或创建会话
        session_id, session_data = get_session(request.session_id)
        agent = session_data["agent"]
        messages = session_data["messages"]
        sidecar_data = session_data["sidecar_data"] # 获取sidecar数据
        
        # 添加用户消息
        user_message = Message(role="user", content=request.message)
        messages.append(user_message)
        
        api_logger.info(f"会话 {session_id} 接收到新消息: {request.message}")
        
        async def event_generator():
            try:
                # 调用agent.run()并流式处理响应，传入sidecar_data
                response_generator = agent.run(messages=messages, stream=True, sidecar_data=sidecar_data)
                
                last_assistant_content = ""
                final_assistant_content = ""
                
                for response_chunk in response_generator:
                    # 在每次循环迭代时都检查sidecar_data，实现解耦
                    if 'hourly_bidding_space_ratio' in sidecar_data:
                        detailed_data = sidecar_data.pop('hourly_bidding_space_ratio')
                        api_logger.info(f"Found and sending 'hourly_bidding_space_ratio' data from sidecar. Size: {len(detailed_data)} records.")
                        yield f"data: {json.dumps({'type': 'bidding_space_data', 'data': detailed_data}, ensure_ascii=False)}\n\n"

                    # 对于流式响应，处理每个chunk
                    if isinstance(response_chunk, list) and response_chunk:
                        latest_message = response_chunk[-1]
                        
                        if hasattr(latest_message, 'role') and latest_message.role == 'assistant':
                            current_content = latest_message.content or ""
                            # 始终记录最新的完整内容
                            final_assistant_content = current_content
                            
                            delta = ""
                            if current_content.startswith(last_assistant_content):
                                # 计算增量内容
                                delta = current_content[len(last_assistant_content):]
                            else:
                                # 内容出现非追加式变化，将当前全部内容作为增量
                                # 前端需要能够处理这种情况，例如直接替换之前的内容
                                api_logger.warning("检测到非追加式内容更新，将发送完整内容作为增量。")
                                delta = current_content
                            
                            if delta:
                                last_assistant_content = current_content
                                # 发送统一的流式内容事件
                                yield f"data: {json.dumps({'type': 'streaming_content', 'delta': delta}, ensure_ascii=False)}\n\n"
                        
                        # 检查是否有工具调用消息，记录日志并通知前端
                        elif hasattr(latest_message, 'role') and latest_message.role == 'tool':
                            tool_name = getattr(latest_message, 'name', '未知工具')
                            api_logger.info(f"Tool call reported by agent: {tool_name}")
                            # 发送通用工具调用事件
                            yield f"data: {json.dumps({'type': 'tool_call', 'name': tool_name}, ensure_ascii=False)}\n\n"
                            
                            # 注意：原先耦合在这里的sidecar检查逻辑已被移除并上提到循环顶部

                # 流式响应完成后，将最终的、完整的助手消息添加到会话历史
                if final_assistant_content:
                    assistant_message = Message(role="assistant", content=final_assistant_content)
                    messages.append(assistant_message)
                    
                    # 更新会话最后活跃时间
                    session_data["last_active"] = datetime.now().isoformat()
                
                # 发送完成事件
                yield f"data: {json.dumps({'type': 'done', 'session_id': session_id}, ensure_ascii=False)}\n\n"
                
            except Exception as e:
                api_logger.error(f"会话 {session_id} 处理时出错: {e}", exc_info=True)
                error_msg = f"处理请求时发生错误: {str(e)}"
                yield f"data: {json.dumps({'type': 'error', 'content': error_msg}, ensure_ascii=False)}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            },
        )
    
    except Exception as e:
        api_logger.error(f"处理聊天请求时出错: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理请求时发生错误: {str(e)}")

@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """获取会话信息"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")
    
    session_data = active_sessions[session_id]
    return {
        "session_id": session_id,
        "created_at": session_data["created_at"],
        "last_active": session_data["last_active"],
        "message_count": len(session_data["messages"])
    }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")
    
    del active_sessions[session_id]
    return {"message": f"会话 {session_id} 已删除"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 