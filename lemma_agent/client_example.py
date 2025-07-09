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
    done = False
    start_time = time.time()
    last_update = start_time
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{SERVER_URL}/chat", json=payload) as response:
                if response.status != 200:
                    console.print(f"[bold red]错误:[/bold red] 服务器返回状态码 {response.status}")
                    error_text = await response.text()
                    console.print(f"错误详情: {error_text}")
                    return {"session_id": current_session_id}
                
                # 使用Rich的Live显示来创建动态更新的区域
                with Live(console=console, auto_refresh=False) as live:
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
                                full_content += data.get("delta", "")
                                last_update = time.time()
                            
                            elif event_type == "tool_call":
                                tool_name = data.get('name', '未知工具')
                                full_content += f"\n\n[bold yellow]LEMMA 正在思考... (调用工具: {tool_name})[/bold yellow]\n"
                                last_update = time.time()
                                
                            elif event_type == "done":
                                done = True
                                current_session_id = data.get("session_id")
                                
                                if DEBUG_MODE:
                                    elapsed_time = time.time() - start_time
                                    console.print(f"\n[dim]对话完成 | 会话ID: {current_session_id} | 耗时: {elapsed_time:.2f}秒[/dim]")
                                
                            elif event_type == "error":
                                console.print(f"\n[bold red]错误:[/bold red] {data.get('content', '未知错误')}")
                                
                            # 统一更新显示，将所有内容放入一个带标题的面板中
                            live.update(Panel(Text.from_markup(full_content), title="[bold green]LEMMA[/bold green]", border_style="green"), refresh=True)

                        except json.JSONDecodeError as e:
                            if DEBUG_MODE:
                                console.print(f"[bold red]解析JSON错误:[/bold red] {e} | 原始数据: {line}")
        
        # 确保在Live上下文之外，最终内容被正确打印
        # Live组件在退出时会打印最后的状态，所以这里不需要再次打印full_content
        console.print()

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