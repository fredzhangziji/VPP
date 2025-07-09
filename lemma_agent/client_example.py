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
    
    # 清屏显示新对话
    if clear_screen:
        os.system('cls' if os.name == 'nt' else 'clear')
    
    # 显示用户消息
    console.print(f"\n[bold blue]用户:[/bold blue] {message}\n")
    
    # 准备接收AI回复的变量
    thinking_content = ""
    answer_content = ""
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
                with Live("", refresh_per_second=10) as live:
                    # 读取SSE流
                    async for line in response.content:
                        line = line.decode('utf-8')
                        
                        if not line.strip() or not line.startswith("data: "):
                            continue
                        
                        # 解析SSE数据
                        try:
                            data = json.loads(line[6:])  # 去掉 "data: " 前缀
                            
                            if DEBUG_MODE:
                                console.print(f"[dim]调试: 收到事件类型: {data.get('type')}[/dim]")
                            
                            # 处理不同类型的事件
                            if data.get("type") == "thinking" and show_thinking:
                                # 思考内容增量更新
                                thinking_content += data.get("content", "")
                                current_display = f"[yellow]【思考过程】[/yellow]\n{thinking_content}\n\n"
                                
                                if answer_content:
                                    current_display += f"[green]【回答内容】[/green]\n{answer_content}"
                                
                                live.update(current_display)
                                last_update = time.time()
                                
                            elif data.get("type") == "answer":
                                # 回答内容增量更新
                                answer_content += data.get("content", "")
                                current_display = ""
                                
                                if show_thinking and thinking_content:
                                    current_display += f"[yellow]【思考过程】[/yellow]\n{thinking_content}\n\n"
                                
                                current_display += f"[green]【回答内容】[/green]\n{answer_content}"
                                live.update(current_display)
                                last_update = time.time()
                                
                            elif data.get("type") == "done":
                                # 对话完成
                                done = True
                                current_session_id = data.get("session_id")
                                
                                if DEBUG_MODE:
                                    elapsed_time = time.time() - start_time
                                    console.print(f"\n[dim]对话完成 | 会话ID: {current_session_id} | 耗时: {elapsed_time:.2f}秒[/dim]")
                                    console.print("=" * 50)
                                
                            elif data.get("type") == "error":
                                # 错误信息
                                console.print(f"\n[bold red]错误:[/bold red] {data.get('content', '未知错误')}")
                        
                        except json.JSONDecodeError as e:
                            if DEBUG_MODE:
                                console.print(f"[bold red]解析JSON错误:[/bold red] {e} | 原始数据: {line}")
                        
                        # 检查是否长时间没有更新，显示等待消息
                        current_time = time.time()
                        if not done and current_time - last_update > 3 and current_time - start_time > 5:
                            live.update(f"{live.renderable}\n[dim]正在生成回复...[/dim]")
                            last_update = current_time
        
        # 输出最终结果
        if not answer_content and thinking_content:
            console.print("\n[bold red]警告:[/bold red] 接收到思考过程但没有回答内容")
            
        if DEBUG_MODE:
            console.print(f"\n思考字符数: {len(thinking_content)} | 回答字符数: {len(answer_content)}")
        
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