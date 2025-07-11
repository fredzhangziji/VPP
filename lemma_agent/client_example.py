#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LEMMA API 客户端示例

演示如何使用LEMMA API进行对话（流式接口）。
"""

import asyncio
import json
import os
import time
import sys
import logging
from typing import Dict, List, Optional
from datetime import datetime

import aiohttp
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich.live import Live

# 设置服务器地址和端口
SERVER_URL = "http://127.0.0.1:8000"

# 配置日志
logging.basicConfig(
    filename='client_example.log',
    level=logging.INFO,
    format='%(asctime)s - RAW: %(message)s',
    encoding='utf-8'
)

# 初始化Rich控制台
console = Console()

# 当前会话ID
current_session_id = None

# 控制是否显示思考过程的标志
show_thinking = True

# 控制是否清除屏幕的标志
clear_screen = False

# 调试模式标志
DEBUG_MODE = False

async def stream_chat(message: str, session_id: Optional[str] = None) -> Dict:
    """
    向服务器发送消息，并以流式方式接收响应
    
    Args:
        message: 用户消息
        session_id: 会话ID，如果为None则创建新会话
        
    Returns:
        包含会话ID的字典
    """
    global current_session_id
    
    payload = {
        "message": message,
        "session_id": session_id
    }
    
    # 清屏或打印空行以准备显示新对话
    if clear_screen:
        os.system('cls' if os.name == 'nt' else 'clear')
        # 如果清屏，则需要重新显示用户输入
        console.print(f"\n[bold blue]用户:[/bold blue] {message}\n")
    else:
        # 在不清屏模式下，用户输入已经显示，我们只添加一个换行来分隔。
        console.print()

    # 准备接收AI回复的变量
    full_content = ""
    is_first_content_chunk = True
    thinking_messages = []
    start_time = time.time()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{SERVER_URL}/chat", json=payload) as response:
                if response.status != 200:
                    console.print(f"[bold red]错误:[/bold red] 服务器返回状态码 {response.status}")
                    error_text = await response.text()
                    console.print(f"错误详情: {error_text}")
                    return {"session_id": current_session_id}
                
                # 读取SSE流
                async for line in response.content:
                    line = line.decode('utf-8')
                    
                    # 将原始响应写入日志
                    if line.strip():
                        logging.info(line.strip())
                    
                    if not line.strip() or not line.startswith("data: "):
                        continue
                    
                    # 解析SSE数据
                    try:
                        data = json.loads(line[6:])
                        
                        if DEBUG_MODE:
                            console.print(f"[dim]调试: 收到事件: {data}[/dim]")
                        
                        event_type = data.get("type")
                        
                        if event_type == "streaming_content":
                            delta = data.get("delta", "")
                            if delta:
                                if is_first_content_chunk:
                                    # 如果有思考过程，打印换行与思考过程分隔
                                    if thinking_messages:
                                        console.print()
                                    console.print("[bold green]LEMMA:[/bold green] ", end="")
                                    is_first_content_chunk = False
                                
                                console.print(delta, end="")
                                full_content += delta
                        
                        elif event_type == "tool_call":
                            if show_thinking:
                                tool_name = data.get('name', '未知工具')
                                think_msg = f"[bold yellow]LEMMA 正在思考... (调用工具: {tool_name})[/bold yellow]"
                                console.print(think_msg)
                                thinking_messages.append(think_msg)
                        
                        elif event_type == "bidding_space_data":
                            detailed_data = data.get('data', {})
                            if DEBUG_MODE:
                                console.print(f"[dim]收到竞价空间详细数据: {json.dumps(detailed_data, indent=2)}[/dim]")
                            
                            # 使用Rich Panel美化输出
                            if detailed_data and 'x' in detailed_data:
                                output = "[bold magenta]竞价空间及相关指标偏差率 (%):[/bold magenta]\n"
                                # 提取除时间轴'x'之外的所有数据系列
                                series_keys = [k for k in detailed_data.keys() if k != 'x']
                                
                                # 准备表头
                                headers = ["时间"] + [detailed_data[k].get('name', k) for k in series_keys]
                                table_rows = " | ".join(headers) + "\n"
                                table_rows += " | ".join(["---"] * len(headers)) + "\n"

                                # 准备每一行的数据
                                for i, time_stamp in enumerate(detailed_data['x']):
                                    row_data = [time_stamp]
                                    for key in series_keys:
                                        # 检查数据是否存在且长度匹配
                                        if i < len(detailed_data[key].get('data', [])):
                                            row_data.append(f"{detailed_data[key]['data'][i]:.2f}")
                                        else:
                                            row_data.append("N/A")
                                    table_rows += " | ".join(row_data) + "\n"
                                
                                output += f"```markdown\n{table_rows}```"
                                console.print(Panel(Markdown(output), title="详细数据", border_style="cyan", expand=False))
                        
                        elif event_type == "done":
                            current_session_id = data.get("session_id")
                            if DEBUG_MODE:
                                elapsed_time = time.time() - start_time
                                console.print(f"\n[dim]对话完成 | 会话ID: {current_session_id} | 耗时: {elapsed_time:.2f}秒[/dim]")
                            break # 结束循环
                            
                        elif event_type == "error":
                            console.print(f"\n[bold red]错误:[/bold red] {data.get('content', '未知错误')}")
                            
                    except json.JSONDecodeError as e:
                        if DEBUG_MODE:
                            console.print(f"[bold red]解析JSON错误:[/bold red] {e} | 原始数据: {line}")
        
        console.print() # 确保在流式输出后换行

        return {"session_id": current_session_id}
        
    except aiohttp.ClientError as e:
        console.print(f"\n[bold red]连接错误:[/bold red] {e}")
        return {"session_id": current_session_id}
    except Exception as e:
        console.print(f"\n[bold red]异常:[/bold red] {e}")
        return {"session_id": current_session_id}

async def main():
    """主函数：处理用户输入并与服务器交互"""
    global current_session_id, show_thinking, clear_screen, DEBUG_MODE
    
    # 检查命令行参数
    if "--debug" in sys.argv:
        DEBUG_MODE = True
        console.print("[bold yellow]调试模式已启用[/bold yellow]")
    
    if "--no-thinking" in sys.argv:
        show_thinking = False
        console.print("[yellow]已禁用思考过程显示[/yellow]")
    
    if "--clear" in sys.argv:
        clear_screen = True
    
    console.print(Panel.fit("LEMMA 电力市场分析智能体", title="欢迎", border_style="green", padding=(1, 2)))
    console.print("\n输入 'q' 退出，'c' 创建新会话，'d' 切换调试模式，'t' 切换思考显示\n")
    
    while True:
        try:
            # 获取用户输入
            user_input = console.input("[bold blue]用户:[/bold blue] ")
            
            # 检查特殊命令
            if user_input.lower() == 'q':
                console.print("[yellow]再见！[/yellow]")
                break
            elif user_input.lower() == 'c':
                current_session_id = None
                console.print("[yellow]已创建新会话[/yellow]")
                continue
            elif user_input.lower() == 'd':
                DEBUG_MODE = not DEBUG_MODE
                console.print(f"[yellow]调试模式: {'开启' if DEBUG_MODE else '关闭'}[/yellow]")
                continue
            elif user_input.lower() == 't':
                show_thinking = not show_thinking
                console.print(f"[yellow]思考过程显示: {'开启' if show_thinking else '关闭'}[/yellow]")
                continue
            elif not user_input.strip():
                continue
            
            # 发送消息到服务器
            result = await stream_chat(user_input, current_session_id)
            current_session_id = result.get("session_id")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]用户中断，程序退出[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[bold red]发生错误:[/bold red] {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]程序已退出[/yellow]") 