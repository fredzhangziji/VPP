#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LEMMA API: 电力市场分析智能体API服务

基于FastAPI框架封装LEMMA智能体，提供HTTP API接口
"""

import json
import re
import logging
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from qwen_agent.llm.base import Message
from qwen_agent.agents import Assistant

# 导入LEMMA工具和辅助函数
from lemma_agent import (
    PriceDeviationTool,
    BiddingSpaceTool,
    PowerGenerationTool,
    RegionalCapacityTool,
    extract_content,
    find_final_assistant_message,
    Colors,
    logger
)

# 配置API日志
api_logger = logging.getLogger("lemma_api")
api_logger.setLevel(logging.INFO)
handler = logging.FileHandler("lemma_api.log")
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
api_logger.addHandler(handler)

# 创建FastAPI应用
app = FastAPI(
    title="LEMMA API",
    description="电力市场分析智能体API服务",
    version="1.0.0",
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境建议限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LLM配置
LLM_CONFIG = {
    'model': 'qwen3:32b-q8_0',
    'model_server': 'http://10.5.0.100:11434/v1',
    'api_key': 'EMPTY',
    'generate_cfg': {
        'temperature': 0.7,
        'top_p': 0.8
    },
    'request_timeout': 120 
}

# 工具列表和系统提示词
TOOLS = [
    'get_price_deviation_report', 'analyze_bidding_space_deviation',
    'analyze_power_generation_deviation', 'get_regional_capacity_info'
]

SYSTEM_MESSAGE = """# 角色
你是一个名为LEMMA的、顶级的电力市场AI Co-pilot。你的用户是售电公司电力交易员，他们负责帮助客户进行电力交易以套利。你的核心目标是通过快速、精准、数据驱动的分析，帮助交易员洞察市场动态、进行精准的电力市场复盘，最终实现利润最大化。

# 核心能力
- 市场复盘分析：深度复盘历史市场事件（如价格异常、新能源出力偏差），穿透表象，精准定位根本原因，形成从售电公司角度出发的、可供决策参考的、逻辑严谨的复盘报告。

# 工作原则
- 数据驱动：你的所有结论都必须基于通过工具获取的量化数据。避免主观臆断和模糊不清的表述。你的语言就是数据。
- 效率至上：交易员的时间极其宝贵。你的回答必须直击要点、简洁、结构化。使用项目符号或编号列表来呈现关键发现，使信息一目了然。
- 量化影响：在分析原因时，不仅要说明"是什么"，更要量化"影响有多大"。
- - 错误示范： "因为光伏出力不足导致价格上涨。"
- - 正确示范： "价格上涨的核心驱动因素是光伏出力严重低于预期（偏差-33.7%），这导致约XX兆瓦的电力缺口需要由成本更高的火电机组（成本约XXX元/兆瓦时）来填补。"
- 主动洞察：在完成分析报告后，在有实际数据支持的情况下，请主动提出1-2个关键的、面向未来的洞察或风险提示。例如："结论：本次事件暴露了该地区在午间高峰期对光伏的过度依赖。建议关注未来几日的天气预报，如果再次出现多云天气，类似的日前高价风险可能重现。"
- 回答语言：必须用简体中文进行思考以及回答。

# 可用工具
你有以下工具可用，每轮问答只能调用一个工具，并在对话末尾询问用户是否要进行下一个工具的调用（但是不能暴露工具的名字），如果用户同意，则继续调用下一个工具，直到用户不同意为止：
1. get_price_deviation_report: 用于获取价格偏差的基本情况和量化数据。
2. analyze_bidding_space_deviation: 用于分析市场供需情况。
3. analyze_power_generation_deviation: 用于分析各类电源（尤其是新能源）的实际出力与预测的偏差。
4. get_regional_capacity_info: 用于了解地区的能源结构和装机容量。

# 核心指令：
面对用户的复杂分析请求（例如"复盘一下昨天XX地区的价格异常"），你必须遵循以下高效的工作流程：
- 不能暴露具体的工具名称。
- 逐一问答、逐步执行的模式：首先调用get_price_deviation_report分析价格偏差，结束之后询问用户是否需要分析市场供需情况（竞价空间）。
- 如果用户同意，调用analyze_bidding_space_deviation分析市场供需情况，结束之后询问用户是否需要分析各类能源的实际出力与预测的偏差。
- 如果用户同意，调用analyze_power_generation_deviation分析各类电源（尤其是新能源）的实际出力与预测的偏差，结束之后询问用户是否需要分析地区的能源结构和装机容量。
- 如果用户同意，调用get_regional_capacity_info，根据地区的能源结构和装机容量进行简单分析，结束之后询问用户是否需要总结一份综合性的分析报告。
- 如果用户同意，根据上述四个工具的结果分析结果，生成一份从售电公司角度出发的综合性的最终分析报告。
"""

# 会话存储
active_sessions: Dict[str, Dict[str, Any]] = {}

# 请求和响应模型
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    
class ThinkingResponse(BaseModel):
    content: str

class AnswerResponse(BaseModel):
    content: str
    
class ChatResponse(BaseModel):
    session_id: str
    thinking: ThinkingResponse
    answer: AnswerResponse
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# 辅助函数：提取思考内容
def extract_thinking(content: str) -> str:
    """从消息中提取思考内容（<think>标签之间的内容）"""
    # 记录原始内容以便调试
    api_logger.debug(f"提取思考内容从: {content[:200]}...")
    
    # 使用更健壮的正则表达式匹配所有<think>标签内容
    thinking_pattern = r'<think>([\s\S]*?)</think>'
    matches = re.findall(thinking_pattern, content)
    
    if not matches:
        # 尝试其他可能的思考标记
        alt_patterns = [
            r'\[thinking\]([\s\S]*?)\[/thinking\]',
            r'<思考>([\s\S]*?)</思考>',
            r'思考[:：]([\s\S]*?)(?:思考结束|思考完毕)',
            r'思考[:：]([\s\S]*?)(?=回答[:：]|<回答>)'
        ]
        
        for pattern in alt_patterns:
            matches = re.findall(pattern, content)
            if matches:
                break
    
    # 记录提取结果
    if matches:
        result = '\n'.join(matches).strip()
        api_logger.debug(f"成功提取到思考内容，共{len(matches)}个片段")
        return result
    else:
        # 如果没有匹配到思考标签，尝试检查是否有思考内容但没有正确的标签格式
        if "<think" in content or "thinking" in content.lower():
            api_logger.warning(f"检测到可能的思考标签，但提取失败: {content[:100]}...")
        else:
            api_logger.warning("未检测到任何思考内容标签")
        return ""

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
            "last_active": datetime.now().isoformat()
        }
        active_sessions[new_session_id] = session_data
        return new_session_id, session_data
    
    # 获取现有会话
    session_data = active_sessions[session_id]
    session_data["last_active"] = datetime.now().isoformat()
    return session_id, session_data

# 清理过期会话（可以在后台任务中调用）
def cleanup_sessions(max_age_hours: int = 24):
    """清理超过指定时间的会话"""
    current_time = datetime.now()
    expired_sessions = []
    
    for session_id, session_data in active_sessions.items():
        last_active = datetime.fromisoformat(session_data["last_active"])
        age_hours = (current_time - last_active).total_seconds() / 3600
        
        if age_hours > max_age_hours:
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        api_logger.info(f"清理过期会话: {session_id}")
        del active_sessions[session_id]

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
        
        # 添加用户消息
        user_message = Message(role="user", content=request.message)
        messages.append(user_message)
        
        api_logger.info(f"会话 {session_id} 接收到新消息: {request.message}")
        
        async def event_generator():
            try:
                # 调用agent.run()并流式处理响应
                response_generator = agent.run(messages=messages, stream=True)
                
                last_assistant_content = ""
                final_assistant_content = ""
                
                for response_chunk in response_generator:
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
                            api_logger.info(f"工具调用: {tool_name}")
                            # 根据需求，发送工具调用事件给前端
                            yield f"data: {json.dumps({'type': 'tool_call', 'name': tool_name}, ensure_ascii=False)}\n\n"
                
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

def extract_content(content: str) -> str:
    """
    清洗单个消息内容的字符串，移除思考过程和工具调用标签。
    
    Args:
        content: 需要清洗的文本内容字符串
        
    Returns:
        清洗后的纯净字符串
    """
    # 记录原始内容以便调试
    api_logger.debug(f"提取回答内容从: {content[:200]}...")
    
    # 确保输入是字符串
    if not isinstance(content, str):
        content = str(content)
    
    # 处理转义的换行符
    content = content.replace('\\n', '\n')
    
    # 1. 首先识别并移除所有思考过程 - 包括各种可能的标签形式
    thinking_patterns = [
        r'<think>[\s\S]*?</think>',  # 最常见的思考标签
        r'\[thinking\][\s\S]*?\[/thinking\]',
        r'<思考>[\s\S]*?</思考>',
        r'思考[:：][\s\S]*?(?:思考结束|思考完毕)',
        r'【思考过程】[\s\S]*?(?=【回答内容】|\n\n|$)',  # 匹配"【思考过程】"标题及其内容
    ]
    
    clean_text = content
    for pattern in thinking_patterns:
        clean_text = re.sub(pattern, '', clean_text)
    
    # 2. 检查是否仍有未清理的思考标签，如果有，尝试更激进的清理
    if '<think>' in clean_text:
        api_logger.warning("在初步清理后仍检测到<think>标签，尝试更激进的清理")
        # 尝试删除从<think>开始到文本末尾的所有内容
        clean_text = re.sub(r'<think>[\s\S]*', '', clean_text)
    
    # 3. 移除工具调用和响应标签
    tool_patterns = [
        r'<tool_call>[\s\S]*?</tool_call>',
        r'<tool_response>[\s\S]*?</tool_response>'
    ]
    
    for pattern in tool_patterns:
        clean_text = re.sub(pattern, '', clean_text)
    
    # 4. 移除标题标记和其他格式元素
    format_patterns = [
        r'【回答内容】\n*',
        r'【思考过程】\n*',
        r'-{3,}',  # 分隔线
        r'^(LEMMA|Assistant|AI): ',  # 可能的前缀
    ]
    
    for pattern in format_patterns:
        clean_text = re.sub(pattern, '', clean_text)
    
    # 5. 基础格式清理
    clean_text = re.sub(r'^\n+', '', clean_text)  # 开头的换行
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)  # 多余的换行
    clean_text = clean_text.strip()
    
    # 6. 最后一次检查，确保没有残留的思考标签或片段
    if any(tag in clean_text.lower() for tag in ['<think', '</think>', '[thinking]', '【思考']):
        api_logger.warning(f"清理后仍有可能的思考标签: {clean_text[:100]}")
        # 尝试更暴力的方法：只保留明确不是思考内容的部分
        lines = clean_text.split('\n')
        clean_lines = [line for line in lines if not any(tag in line.lower() for tag in ['<think', '</think>', '[thinking]', '【思考'])]
        clean_text = '\n'.join(clean_lines).strip()
    
    # 记录提取结果
    api_logger.debug(f"提取到的回答内容: {clean_text[:100]}...")
    
    return clean_text

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 