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

注意：虽然elu/elu_mldev是SSH管理凭证，用于登录到服务器10.5.0.100，
但这些凭证不应用于API调用，因为Ollama的API在默认配置下无需认证。
SSH凭证仅用于管理服务器，而不是API访问。
"""

import json
import sys
import time
import logging
import re
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

# 降低httpx库的日志级别
logging.getLogger("httpx").setLevel(logging.WARNING)

# 导入Qwen-Agent相关库
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.agents import Assistant
from qwen_agent.llm.base import Message  # 导入Qwen-Agent的消息类

# 终端颜色 - 更简单的设置
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    BLUE = '\033[34m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    RED = '\033[31m'
    CYAN = '\033[36m'

# 配置日志
logging.basicConfig(
    level=logging.INFO,
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
        "description": "电力市场区域，例如'华东'、'华北'、'南方'等",
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
        "description": "电力市场区域，例如'华东'、'华北'、'南方'等",
        "required": False
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
        "description": "电力市场区域，例如'华东'、'华北'、'南方'等",
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
    支持各种可能的响应格式，并去除<think>标签内容。
    """
    content = ""
    
    # 处理各种可能的响应格式
    if isinstance(response, dict) and 'content' in response:
        content = response['content']
    elif hasattr(response, 'content') and isinstance(response.content, str):
        content = response.content
    elif isinstance(response, list) and len(response) > 0:
        # 如果是列表，尝试从第一个元素获取内容
        item = response[0]
        if isinstance(item, dict) and 'content' in item:
            content = item['content']
        elif hasattr(item, 'content') and isinstance(item.content, str):
            content = item.content
    elif isinstance(response, str):
        content = response
    
    # 确保content是字符串
    if not isinstance(content, str):
        content = str(content)
    
    # 处理Message格式响应 - 增强识别能力
    message_pattern = r"\[Message\({.*?'content':\s*'(.*?)'}.*?\)\]"
    message_match = re.search(message_pattern, content, re.DOTALL)
    if message_match:
        content = message_match.group(1)
    else:
        # 尝试其他Message格式
        message_pattern2 = r"\[Message\({.*?'content':\s*\"(.*?)\".*?\)\]"
        message_match2 = re.search(message_pattern2, content, re.DOTALL)
        if message_match2:
            content = message_match2.group(1)
        # 再尝试一种格式
        message_pattern3 = r"Message\({'role':.*?'content':\s*['\"](.+?)['\"].*?\}\)"
        message_match3 = re.search(message_pattern3, content, re.DOTALL)
        if message_match3:
            content = message_match3.group(1)
    
    # 处理转义的换行符
    content = content.replace('\\n', '\n')
    
    # 先移除所有<think>标签及其内容
    # 使用贪婪匹配确保捕获所有内容
    answer = re.sub(r'<think>[\s\S]*?</think>', '', content)
    
    # 如果存在不闭合的think标签，也要处理
    answer = re.sub(r'<think>[\s\S]*', '', answer)
    
    # 尝试其他可能的思考格式
    answer = re.sub(r'\[thinking\][\s\S]*?\[/thinking\]', '', answer)
    answer = re.sub(r'thinking:[\s\S]*?thinking end', '', answer, flags=re.IGNORECASE)
    
    # 清除可能存在的"LEMMA: "前缀
    answer = re.sub(r'^LEMMA: ', '', answer)
    # 清除多余的换行
    answer = re.sub(r'^\n+', '', answer)
    answer = re.sub(r'\n{3,}', '\n\n', answer)  # 超过2个换行符的替换为2个
    answer = answer.strip()
    
    # 最后一次检查，确保没有残留的think相关内容
    if '<think>' in answer or '</think>' in answer:
        answer = re.sub(r'<think>[\s\S]*?</think>', '', answer)
        answer = re.sub(r'<think>[\s\S]*', '', answer)
        answer = re.sub(r'[\s\S]*</think>', '', answer)
    
    return answer


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
    # 清屏
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # 打印欢迎横幅
    print_welcome_banner()
    
    # 生成会话ID
    session_id = get_session_id()
    logger.debug(f"开始新会话，ID: {session_id}")
    
    # 配置LLM
    llm_cfg = {
        'model': 'qwen3:32b',
        'model_server': 'http://10.5.0.100:11434/v1',  # Ollama服务端点
        'api_key': 'EMPTY',  # Ollama不需要API密钥
        'generate_cfg': {
            'temperature': 0.7,
            'top_p': 0.8
        }
    }
    
    # 定义要使用的工具列表
    tools = [
        'get_price_deviation_report',
        'analyze_bidding_space_deviation',
        'analyze_power_generation_deviation',
        'get_regional_capacity_info'
    ]
    
    # 定义系统提示词
    system_message = """你是LEMMA，一个专注于电力市场分析的专业智能助手。
你的专长是分析电价偏差事件，找出价格异常的原因，并提供专业的分析报告。

在分析问题时，你应该：
1. 首先考虑使用价格偏差报告工具(get_price_deviation_report)来获取基本情况
2. 根据初步发现，进一步使用竞价空间分析工具(analyze_bidding_space_deviation)查看市场供需情况
3. 如果发现与新能源有关的异常，使用电力生成偏差工具(analyze_power_generation_deviation)分析新能源出力
4. 最后，可以使用区域容量信息工具(get_regional_capacity_info)了解该地区的能源结构特点

你的回答应该专业、客观、有逻辑性，并且基于工具提供的数据进行分析。避免主观臆断，重点关注数据支持的结论。
"""
    
    try:
        print(f"{Colors.BLUE}初始化LEMMA Agent...{Colors.RESET}")
        # 使用Qwen-Agent提供的Assistant类
        agent = Assistant(
            llm=llm_cfg,
            system_message=system_message,
            function_list=tools
        )
        print(f"{Colors.GREEN}初始化成功！{Colors.RESET}")
    except Exception as e:
        logger.error(f"Agent初始化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 初始化消息历史 - 使用Message对象
    messages = []
    
    # 开始交互循环
    while True:
        # 获取用户输入
        try:
            user_input = input(f"\n{Colors.BOLD}{Colors.YELLOW}用户: {Colors.RESET}")
        except (KeyboardInterrupt, EOFError):
            print(f"\n{Colors.GREEN}再见！感谢使用LEMMA电力市场分析智能体！{Colors.RESET}")
            break
        
        # 检查是否退出
        if user_input.lower() in ['exit', 'quit']:
            print(f"\n{Colors.GREEN}再见！感谢使用LEMMA电力市场分析智能体！{Colors.RESET}")
            break
        
        # 使用Message对象创建消息
        user_message = Message(role="user", content=user_input)
        messages.append(user_message)
        
        # 显示状态
        print(f"\n{Colors.BLUE}LEMMA思考中...{Colors.RESET}")
        
        try:
            # 收集所有响应
            responses = []
            previous_content = ""
            final_content = ""
            
            # 使用try-finally确保即使出错也能正确显示
            try:
                # 注意：agent.run始终返回生成器，即使stream=False
                for response in agent.run(messages=messages, stream=True):  # 确保开启流式传输
                    responses.append(response)
                    
                    # 提取内容
                    if isinstance(response, dict) and 'content' in response:
                        current_content = response['content']
                    elif hasattr(response, 'content'):
                        current_content = response.content
                    elif isinstance(response, str):
                        current_content = response
                    else:
                        current_content = str(response)
                    
                    # 流式打印响应（返回清理后的内容）
                    previous_content = typewriter_print(current_content, previous_content)
                    
                    # 更新最终内容（已清除思考过程）
                    final_content = extract_content(current_content)
            finally:
                # 确保最后打印一个换行
                if final_content:
                    print()
            
            # 将响应添加到消息历史
            if final_content:
                assistant_message = Message(role="assistant", content=final_content)
                messages.append(assistant_message)
                logger.debug("成功添加助手回复到历史")
            else:
                logger.warning(f"无法从响应中提取有效内容")
        
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}操作已取消{Colors.RESET}")
            continue
        except Exception as e:
            logger.error(f"处理过程中出错: {e}")
            import traceback
            traceback.print_exc()
            print(f"{Colors.RED}请重试或尝试其他问题{Colors.RESET}") 