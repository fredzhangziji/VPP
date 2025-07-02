#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LEMMA: 电力市场分析智能体

基于Qwen-Agent框架开发的电力市场分析智能体，可以分析电价偏差事件，连接到远程Ollama服务。
具有以下核心工具：
1. 价格偏差分析工具
2. 竞价空间分析工具 
3. 电力生成偏差分析工具
4. 区域容量信息工具
"""

import json
import sys
import time
import logging
import re
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.agents import Assistant
from qwen_agent.llm.base import Message

# 终端颜色设置
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    BLUE = '\033[34m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    RED = '\033[31m'
    CYAN = '\033[36m'

# 降低httpx库的日志级别
logging.getLogger("httpx").setLevel(logging.WARNING)

# 配置日志
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lemma_agent.log"),  # 文件日志
        logging.StreamHandler()  # 终端日志
    ]
)
logger = logging.getLogger("LEMMA")
# 只将ERROR及以上级别的日志输出到终端
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setLevel(logging.ERROR)

@register_tool('get_price_deviation_report')
class PriceDeviationTool(BaseTool):
    """用于获取电价偏差报告的工具。"""
    
    description = "获取特定日期的电价偏差报告，包括市场价格、预测价格和实际价格之间的偏差情况。"
    
    parameters = [{
        "name": "date",
        "type": "string",
        "description": "分析日期，格式为YYYY-MM-DD或描述性日期如'昨天'、'6月24日'等",
        "required": True
    }, {
        "name": "region",
        "type": "string",
        "description": "电力市场区域，例如'呼包东'、'呼包西'、'内蒙全省'等",
        "required": False
    }]
    
    def call(self, params: str, **kwargs) -> str:
        """执行电价偏差报告获取操作。"""
        logger.debug(f"PriceDeviationTool被调用，参数: {params}")
        print(f"{Colors.CYAN}[工具] 获取电价偏差报告...{Colors.RESET}")
        
        # 解析参数
        try:
            parsed_params = json.loads(params)
            date = parsed_params.get("date", "")
            region = parsed_params.get("region", "全国")
        except Exception as e:
            logger.error(f"参数解析错误: {e}")
            return json.dumps({"error": f"参数解析错误: {e}"}, ensure_ascii=False)
        
        # TODO: 实现与数据库的连接和查询
        # 当前返回模拟数据
        mock_report = {
            "date": date,
            "region": region,
            "market_price": 456.78,
            "forecast_price": 420.25,
            "actual_price": 510.35,
            "deviation_percentage": 21.5,
            "peak_hours_deviation": 28.7,
            "valley_hours_deviation": 15.2,
            "analysis_summary": "该日期电价出现较大偏差，特别是在用电高峰期。实际价格比预测高出约21.5%，超出正常波动范围。"
        }
        
        return json.dumps(mock_report, ensure_ascii=False)

@register_tool('analyze_bidding_space_deviation')
class BiddingSpaceTool(BaseTool):
    """用于分析竞价空间偏差的工具。"""
    
    description = "分析特定日期和区域的电力市场竞价空间偏差，包括供应曲线、需求曲线和出清价格的变化。"
    
    parameters = [{
        "name": "date",
        "type": "string",
        "description": "分析日期，格式为YYYY-MM-DD或描述性日期如'昨天'、'6月24日'等",
        "required": True
    }, {
        "name": "region",
        "type": "string",
        "description": "电力市场区域，例如'呼包东'、'呼包西'、'内蒙全省'等",
        "required": True
    }, {
        "name": "time_period",
        "type": "string",
        "description": "分析的时段，例如'全天'、'高峰期'、'15:00-18:00'等",
        "required": False
    }]
    
    def call(self, params: str, **kwargs) -> str:
        """执行竞价空间偏差分析操作。"""
        logger.debug(f"BiddingSpaceTool被调用，参数: {params}")
        print(f"{Colors.CYAN}[工具] 分析竞价空间偏差...{Colors.RESET}")
        
        # 解析参数
        try:
            parsed_params = json.loads(params)
            date = parsed_params.get("date", "")
            region = parsed_params.get("region", "全国")
            time_period = parsed_params.get("time_period", "全天")
        except Exception as e:
            logger.error(f"参数解析错误: {e}")
            return json.dumps({"error": f"参数解析错误: {e}"}, ensure_ascii=False)
        
        # TODO: 实现与数据库的连接和查询
        # 当前返回模拟数据
        mock_analysis = {
            "date": date,
            "region": region,
            "time_period": time_period,
            "supply_curve_shift": "上移约15%",
            "demand_curve_shift": "基本稳定，小时尖峰需求增加8%",
            "clearing_price_change": "+85.4元/MWh",
            "bid_concentration_index": 0.68,
            "unusual_bidding_patterns": True,
            "key_market_participants_impact": "某些大型新能源供应商的出力显著低于预期，导致供应曲线上移",
            "analysis_summary": "该日期的竞价空间出现明显异常，主要表现为供应曲线整体上移，而需求相对稳定，导致出清价格显著上升。这可能与新能源出力不足有关。"
        }
        
        return json.dumps(mock_analysis, ensure_ascii=False)

@register_tool('analyze_power_generation_deviation')
class PowerGenerationTool(BaseTool):
    """用于分析电力生成偏差的工具。"""
    
    description = "分析特定日期和区域的电力生成偏差情况，尤其是新能源(风电、光伏)的实际出力与预测出力的差异。"
    
    parameters = [{
        "name": "date",
        "type": "string",
        "description": "分析日期，格式为YYYY-MM-DD或描述性日期如'昨天'、'6月24日'等",
        "required": True
    }, {
        "name": "region",
        "type": "string",
        "description": "电力市场区域，例如'呼包东'、'呼包西'、'内蒙全省'等",
        "required": False
    }, {
        "name": "energy_type",
        "type": "string",
        "description": "能源类型，如'风电'、'光伏'、'水电'、'火电'或'全部'",
        "required": False
    }]
    
    def call(self, params: str, **kwargs) -> str:
        """执行电力生成偏差分析操作。"""
        logger.debug(f"PowerGenerationTool被调用，参数: {params}")
        print(f"{Colors.CYAN}[工具] 分析电力生成偏差...{Colors.RESET}")
        
        # 解析参数
        try:
            parsed_params = json.loads(params)
            date = parsed_params.get("date", "")
            region = parsed_params.get("region", "全国")
            energy_type = parsed_params.get("energy_type", "全部")
        except Exception as e:
            logger.error(f"参数解析错误: {e}")
            return json.dumps({"error": f"参数解析错误: {e}"}, ensure_ascii=False)
        
        # TODO: 实现与数据库的连接和查询
        # 当前返回模拟数据
        mock_analysis = {
            "date": date,
            "region": region,
            "energy_type": energy_type,
            "forecast_generation": {
                "wind": 12500,  # 单位: MWh
                "solar": 18700,
                "hydro": 8500,
                "thermal": 45000
            },
            "actual_generation": {
                "wind": 8200,  # 单位: MWh
                "solar": 12400,
                "hydro": 8300,
                "thermal": 48000
            },
            "deviation_percentage": {
                "wind": -34.4,
                "solar": -33.7,
                "hydro": -2.4,
                "thermal": +6.7
            },
            "weather_impact": "该日期出现大面积云层覆盖和弱风天气，显著影响了风电和光伏发电效率",
            "analysis_summary": "新能源出力严重不足，风电和光伏分别比预期低了34.4%和33.7%，导致火电增发以弥补缺口。这是价格偏差的主要原因之一。"
        }
        
        return json.dumps(mock_analysis, ensure_ascii=False)


@register_tool('get_regional_capacity_info')
class RegionalCapacityTool(BaseTool):
    """用于获取区域发电容量信息的工具。"""
    
    description = "获取特定区域的发电容量结构信息，包括不同类型能源的装机容量、占比以及近期变化趋势。"
    
    parameters = [{
        "name": "region",
        "type": "string",
        "description": "电力市场区域，例如'华东'、'华北'、'南方'等",
        "required": True
    }, {
        "name": "year",
        "type": "string",
        "description": "查询年份，例如'2024'、'今年'等",
        "required": False
    }]
    
    def call(self, params: str, **kwargs) -> str:
        """执行区域容量信息获取操作。"""
        logger.debug(f"RegionalCapacityTool被调用，参数: {params}")
        print(f"{Colors.CYAN}[工具] 获取区域容量信息...{Colors.RESET}")
        
        # 解析参数
        try:
            parsed_params = json.loads(params)
            region = parsed_params.get("region", "")
            year = parsed_params.get("year", "2024")
        except Exception as e:
            logger.error(f"参数解析错误: {e}")
            return json.dumps({"error": f"参数解析错误: {e}"}, ensure_ascii=False)
        
        # TODO: 实现与数据库的连接和查询
        # 当前返回模拟数据
        mock_info = {
            "region": region,
            "year": year,
            "total_capacity": 158500,  # 单位: MW
            "capacity_by_type": {
                "wind": 28500,  # 单位: MW
                "solar": 42300,
                "hydro": 15700,
                "thermal": 68000,
                "nuclear": 4000
            },
            "percentage_by_type": {
                "wind": 18.0,
                "solar": 26.7,
                "hydro": 9.9,
                "thermal": 42.9,
                "nuclear": 2.5
            },
            "year_on_year_change": {
                "wind": +21.5,
                "solar": +35.8,
                "hydro": +2.1,
                "thermal": -5.3,
                "nuclear": 0.0
            },
            "new_energy_penetration": 44.7,  # 风电+光伏占比
            "analysis_summary": "该区域新能源占比接近45%，而且仍在快速增长。风电和光伏的装机容量在过去一年分别增长了21.5%和35.8%。高新能源渗透率使得电网对天气条件更加敏感，导致价格波动加剧。"
        }
        
        return json.dumps(mock_info, ensure_ascii=False)

def extract_content(response):
    """
    从响应中提取纯文本内容，去除思考过程。
    支持各种可能的响应格式，包括列表、字典、字符串和嵌套内容。
    参考了Qwen-Agent的消息处理方式。
    
    Args:
        response: 各种可能的响应格式
        
    Returns:
        str: 提取的纯文本内容
    """
    content = ""
    
    # 处理字典类型响应
    if isinstance(response, dict):
        if 'content' in response:
            content = response['content']
        elif 'message' in response and isinstance(response['message'], dict):
            if 'content' in response['message']:
                content = response['message']['content']
    
    # 处理具有content属性的对象
    elif hasattr(response, 'content'):
        if isinstance(response.content, str):
            content = response.content
        elif hasattr(response.content, 'content'):  # 处理嵌套content
            content = response.content.content
    
    # 处理列表类型响应 - 通常由多个消息组成
    elif isinstance(response, list):
        # 优先查找最后一个元素中的内容(通常包含最终回答)
        if len(response) > 0:
            for i in range(len(response)-1, -1, -1):  # 从后往前查找
                item = response[i]
                # 处理字典类型元素
                if isinstance(item, dict) and 'content' in item:
                    if 'role' in item and item['role'] == 'assistant':  # 优先选择助手角色的消息
                        content = item['content']
                        break
                    elif not content:  # 如果还没找到内容，先保存着
                        content = item['content']
                
                # 处理字符串类型元素 - 可能直接包含文本或XML标签
                elif isinstance(item, str):
                    # 优先查找不含工具调用/响应的内容
                    if '<tool_call>' not in item and '<tool_response>' not in item:
                        if not content:  # 如果还没找到内容
                            content = item
                    elif not content:  # 如果还没找到内容，可以先保存工具相关内容
                        content = item
    
    # 直接处理字符串类型响应
    elif isinstance(response, str):
        content = response
    
    # 确保content是字符串类型
    if not isinstance(content, str):
        content = str(content)
    
    # 处理消息格式
    message_patterns = [
        r"\[Message\({.*?'content':\s*'(.*?)'}.*?\)\]",  # 单引号格式
        r"\[Message\({.*?'content':\s*\"(.*?)\".*?\)\]",  # 双引号格式
        r"Message\({'role':.*?'content':\s*['\"](.+?)['\"].*?\}\)"  # 第三种格式
    ]
    
    for pattern in message_patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            content = match.group(1)
            break
    
    # 处理转义的换行符
    content = content.replace('\\n', '\n')
    
    # 移除思考过程 - 更全面的模式匹配
    # 1. 移除<think>标签及其内容
    answer = re.sub(r'<think>[\s\S]*?</think>', '', content)
    # 2. 处理不闭合的think标签
    answer = re.sub(r'<think>[\s\S]*', '', answer)
    # 3. 处理其他思考格式
    answer = re.sub(r'\[thinking\][\s\S]*?\[/thinking\]', '', answer)
    answer = re.sub(r'thinking:[\s\S]*?thinking end', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'好的，用户问的是.*?。首先', '首先', answer, flags=re.DOTALL)
    answer = re.sub(r'用户问的是.*?。(我需要|我将)', r'\1', answer, flags=re.DOTALL)
    answer = re.sub(r'接下来，我应该.*?工具', '使用工具', answer, flags=re.DOTALL)
    answer = re.sub(r'因此，(步骤|流程)可能是：.*?首先', '首先', answer, flags=re.DOTALL)
    answer = re.sub(r'现在需要按照步骤调用工具.*?首先', '首先', answer, flags=re.DOTALL)
    
    # 移除工具调用和响应
    answer = re.sub(r'<tool_call>[\s\S]*?</tool_call>', '', answer)
    answer = re.sub(r'<tool_response>[\s\S]*?</tool_response>', '', answer)
    
    # 清理格式
    # 1. 移除LEMMA前缀
    answer = re.sub(r'^LEMMA: ', '', answer)
    # 2. 清理多余换行
    answer = re.sub(r'^\n+', '', answer)
    answer = re.sub(r'\n{3,}', '\n\n', answer)
    answer = answer.strip()
    
    # 最后检查think标签
    if '<think>' in answer or '</think>' in answer:
        answer = re.sub(r'<think>[\s\S]*?</think>', '', answer)
        answer = re.sub(r'<think>[\s\S]*', '', answer)
        answer = re.sub(r'[\s\S]*</think>', '', answer)
    
    # 如果提取结果为空，但看起来应该有内容(包含字母或数字字符)
    if not answer.strip() and re.search(r'[a-zA-Z0-9]', content):
        # 检查是否完整是工具调用
        if (('<tool_call>' in content and '</tool_call>' in content) or 
            ('<tool_response>' in content and '</tool_response>' in content)):
            # 此时确实是纯工具调用，不需要进行内容提取
            return ""
        
        # 试图提取任何可能的文本内容，跳过XML标签
        text_fragments = re.findall(r'>([^<>]+)<', content)
        if text_fragments:
            return ' '.join(text_fragments)
    
    return answer

def extract_tool_responses(messages: List[Message]) -> Dict[str, Any]:
    """
    一个更健壮的辅助函数，用于从消息历史中提取并分类所有工具的响应。

    Args:
        messages: 包含所有对话历史的消息列表。

    Returns:
        一个字典，键是工具名，值是该工具返回的已解析的JSON数据。
    """
    tool_outputs = {}
    # Qwen-Agent中，工具调用后，框架通常会添加一个 role='tool' 的消息
    # 我们遍历这个列表，寻找这样的消息
    for msg in messages:
        # Qwen-Agent 的标准做法是将工具调用的结果放在 role='tool' 的消息中
        if msg.role == 'tool':
            try:
                # msg.name 是工具名, msg.content 是返回的JSON字符串
                tool_name = msg.name
                tool_data = json.loads(msg.content)
                tool_outputs[tool_name] = tool_data
                logger.info(f"成功提取到工具 '{tool_name}' 的响应。")
            except (json.JSONDecodeError, AttributeError) as e:
                logger.warning(f"解析工具响应时出错: {e} - 消息内容: {getattr(msg, 'content', 'N/A')}")
    
    return tool_outputs

def generate_final_analysis(messages: List[Message]) -> str:
    """
    根据所有工具调用的结果，生成一个综合性的最终分析报告 (优化版)。
    
    Args:
        messages: 所有消息的历史记录。
    
    Returns:
        生成的分析报告字符串。
    """
    logger.info("开始生成最终分析报告...")
    start_time = time.time()
    
    # 第一步：调用辅助函数，结构化地提取所有工具的输出
    tool_results = extract_tool_responses(messages)
    
    if not tool_results:
        logger.warning("未能从历史记录中提取到任何有效的工具响应。")
        return "未能收集到足够的数据来生成分析报告。"

    # 第二步：基于提取好的数据，构建分析报告的各个部分
    analysis_parts = []
    
    # 价格偏差报告
    price_data = tool_results.get('get_price_deviation_report')
    if price_data:
        analysis_parts.append(
            f"根据价格偏差报告，在日期 {price_data.get('date', 'N/A')}，"
            f"{price_data.get('region', '该地区')}的实际电价与预测存在 {price_data.get('deviation_percentage', 0)}% 的显著偏差。"
            f"“{price_data.get('analysis_summary', '')}”"
        )
        
    # 竞价空间分析
    bidding_data = tool_results.get('analyze_bidding_space_deviation')
    if bidding_data:
        analysis_parts.append(
            f"竞价空间分析显示，偏差主要源于供应侧：{bidding_data.get('analysis_summary', '')}"
        )

    # 发电出力分析
    generation_data = tool_results.get('analyze_power_generation_deviation')
    if generation_data:
        wind_dev = generation_data.get('deviation_percentage', {}).get('wind', 0)
        solar_dev = generation_data.get('deviation_percentage', {}).get('solar', 0)
        analysis_parts.append(
            "发电出力分析找到了问题的核心：新能源出力严重不足。"
            f"风电实际出力比预期低了 {abs(wind_dev)}%，"
            f"光伏则低了 {abs(solar_dev)}%，这直接导致了供应紧张。"
        )

    # 区域容量结构分析
    capacity_data = tool_results.get('get_regional_capacity_info')
    if capacity_data:
        analysis_parts.append(
            "从区域能源结构来看，问题的深层原因在于："
            f"{capacity_data.get('analysis_summary', '未能获取到该地区的能源结构总结。')}"
        )

    # 第三步：组合所有分析部分，生成最终报告
    if not analysis_parts:
        final_report = "虽然调用了工具，但未能从返回数据中构建出有效的分析结论。"
        logger.warning(final_report)
    else:
        # 使用编号和换行符来组织报告，使其更清晰
        final_report = "综合分析报告如下：\n\n"
        for i, part in enumerate(analysis_parts, 1):
            final_report += f"{i}. {part}\n\n"
        final_report = final_report.strip()
    
    logger.info(f"生成分析报告完成，耗时: {time.time() - start_time:.2f}秒")
    return final_report


def typewriter_print(content, previous_content=""):
    """
    打印机风格的流式输出，不显示思考过程
    
    Args:
        content: 当前内容
        previous_content: 之前打印的内容
    
    Returns:
        返回当前打印的内容，用于下次比较
    """
    # 先提取内容
    clean_content = extract_content(content)
    
    # 如果提取的内容为空但原始内容不为空，可能是工具调用或未识别格式
    # 此时不输出任何内容，等待实际结果
    if not clean_content.strip() and content:
        return previous_content
    
    # 如果内容没有变化，直接返回
    if clean_content == previous_content:
        return previous_content
    
    # 计算上一次输出占用的行数
    if previous_content:
        # 安全起见，直接使用清屏方式重新打印
        # 这种方式不依赖于ANSI转义序列的行数计算，更可靠
        if os.name == 'nt':
            os.system('cls')
        else:
            os.system('clear')
    
    # 打印新内容
    prefix = f"{Colors.GREEN}LEMMA: {Colors.RESET}"
    print(f"{prefix}{clean_content}", flush=True)
    
    return clean_content


def print_welcome_banner():
    """打印欢迎横幅"""
    banner = f"""
{Colors.GREEN}╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║  {Colors.BOLD}LEMMA - 电力市场分析智能体{Colors.RESET}{Colors.GREEN}                                        ║
║                                                                    ║
║  基于Qwen-Agent框架和远程Ollama服务 (qwen3:32b)                    ║
║                                                                    ║
║  {Colors.YELLOW}核心功能:{Colors.GREEN}                                                         ║
║  • 价格偏差分析           • 竞价空间分析                           ║
║  • 电力生成偏差分析       • 区域容量信息                           ║
║                                                                    ║
║  {Colors.CYAN}输入'exit'或'quit'退出对话{Colors.GREEN}                                        ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝{Colors.RESET}
"""
    print(banner)


def get_session_id():
    """生成唯一的会话ID"""
    return datetime.now().strftime("%Y%m%d%H%M%S")

if __name__ == "__main__":
    # ... 您之前的代码，直到 get_session_id() 函数定义结束 ...

    # 清屏和打印欢迎横幅保持不变
    os.system('cls' if os.name == 'nt' else 'clear')
    print_welcome_banner()

    session_id = get_session_id()
    logger.info(f"开始新会话，ID: {session_id}")

    # LLM配置保持不变
    llm_cfg = {
        'model': 'qwen3:32b',  # 已根据您的要求更新
        'model_server': 'http://10.5.0.100:11434/v1',
        'api_key': 'EMPTY',
        'generate_cfg': {
            'temperature': 0.7,
            'top_p': 0.8
        },
        'request_timeout': 120  # 建议将超时时间设置得更长，以应对复杂的多步调用
    }

    # 工具列表和系统提示词保持不变
    tools = [
        'get_price_deviation_report', 'analyze_bidding_space_deviation',
        'analyze_power_generation_deviation', 'get_regional_capacity_info'
    ]
    system_message = """你是LEMMA，一个专注于电力市场分析的专业智能助手...（您的系统提示词内容）"""

    # Agent初始化，使用我们之前确认的 Assistant 类
    try:
        print(f"{Colors.BLUE}初始化LEMMA Agent...{Colors.RESET}")
        agent = Assistant(llm=llm_cfg,
                          system_message=system_message,
                          function_list=tools)
        print(f"{Colors.GREEN}初始化成功！{Colors.RESET}")
    except Exception as e:
        logger.error(f"Agent初始化失败: {e}", exc_info=True)
        sys.exit(1)

    # 初始化消息历史
    messages = []

    # =================================================================
    # ↓↓↓ 这是重构后的、更简洁、更可靠的交互循环 ↓↓↓
    # =================================================================
    while True:
        try:
            user_input = input(f"\n{Colors.BOLD}{Colors.YELLOW}用户: {Colors.RESET}")
        except (KeyboardInterrupt, EOFError):
            print(f"\n{Colors.GREEN}再见！感谢使用LEMMA电力市场分析智能体！{Colors.RESET}")
            break

        if user_input.lower() in ['exit', 'quit']:
            print(f"\n{Colors.GREEN}再见！感谢使用LEMMA电力市场分析智能体！{Colors.RESET}")
            break

        messages.append(Message(role="user", content=user_input))

        print(f"\n{Colors.BLUE}LEMMA思考中...{Colors.RESET}")

        response_generator = agent.run(messages=messages, stream=True)

        full_response_content = ""
        last_chunk_printed = ""
        final_message_object = None

        try:
            for response_chunk in response_generator:
                # Qwen-Agent的流式输出通常是一个消息列表
                if isinstance(response_chunk, list) and response_chunk:
                    # 我们只关心最新的消息，通常是列表的最后一个
                    latest_message = response_chunk[-1]
                    
                    # 检查是否是助手正在生成内容
                    if latest_message.role == 'assistant':
                        new_content = latest_message.content
                        # 计算增量部分并以流式打印
                        if new_content.startswith(last_chunk_printed):
                            delta = new_content[len(last_chunk_printed):]
                            print(f"{Colors.GREEN}{delta}{Colors.RESET}", end='', flush=True)
                            last_chunk_printed = new_content
                        else: # 如果内容不是增量的，直接覆盖打印
                            # 使用回车符\r和清行符\x1b[K来覆盖当前行
                            sys.stdout.write('\r\x1b[K')
                            print(f"{Colors.GREEN}LEMMA: {new_content}{Colors.RESET}", end='', flush=True)
                            last_chunk_printed = new_content

                        full_response_content = new_content

                    # 检查是否有工具调用，并在后台打印出来用于调试
                    elif latest_message.role == 'tool' or (isinstance(latest_message.content, str) and '<tool_call>' in latest_message.content):
                         # 工具调用时，在思考提示后换行，让界面更整洁
                        if not last_chunk_printed.endswith('\n'):
                            print()
                        print(f"{Colors.CYAN}[工具] Agent正在调用工具... ({latest_message.tool_name if hasattr(latest_message, 'tool_name') else ''}){Colors.RESET}")
                        last_chunk_printed = "" # 重置已打印内容，准备接收新一轮的助手回复
            
            # 循环结束后，打印一个换行符，让格式更美观
            print()

            # 整个循环结束后，我们需要判断最终状态
            # 使用 extract_content 函数来判断最终内容是否是干净的回答
            clean_final_answer = extract_content(full_response_content)

            if clean_final_answer:
                # 如果有干净的回答，说明Agent自己完成了总结
                final_message_object = Message(role='assistant', content=clean_final_answer)
                logger.info("Agent已生成最终回答。")
            else:
                # 如果最后一步仍然是工具调用，或者内容为空，说明Agent没能自己总结
                # 此时，我们调用您编写的 generate_final_analysis 函数来强制生成总结
                logger.warning("Agent未生成最终文本回答，尝试手动生成分析报告。")
                manual_summary = generate_final_analysis(messages + [Message(role='assistant', content=full_response_content)])
                if manual_summary:
                    print(f"{Colors.BOLD}{Colors.GREEN}LEMMA (分析总结):{Colors.RESET}\n{manual_summary}")
                    final_message_object = Message(role='assistant', content=manual_summary)
                else:
                    logger.error("手动生成分析报告也失败了。")
                    print(f"{Colors.RED}无法生成最终分析报告。{Colors.RESET}")

            # 将最终有效的回复（无论是Agent自生成的还是我们手动总结的）加入历史记录
            if final_message_object:
                messages.append(final_message_object)

        except Exception as e:
            logger.error(f"处理流式响应时出错: {e}", exc_info=True)
            print(f"{Colors.RED}\n处理过程中发生错误，请查看日志 lemma_agent.log。{Colors.RESET}")