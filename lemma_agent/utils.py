#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for LEMMA Agent and API
"""

import re
import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Any, Dict

from qwen_agent.llm.base import Message

# Configure logger for utilities
util_logger = logging.getLogger("lemma_utils")
util_logger.setLevel(logging.INFO)

class Colors:
    """Class to hold terminal color codes."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    BLUE = '\033[34m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    RED = '\033[31m'
    CYAN = '\033[36m'

def parse_date(date_str: str) -> tuple:
    """
    Parses a date string and returns a start and end time for queries.
    Handles formats like 'YYYY-MM-DD' and descriptive terms like '昨天'.
    """
    if date_str == "昨天":
        target_date = datetime.now() - timedelta(days=1)
    elif date_str == "今天":
        target_date = datetime.now()
    elif date_str == "前天":
        target_date = datetime.now() - timedelta(days=2)
    elif re.match(r'(\d+)月(\d+)日', date_str):
        match = re.match(r'(\d+)月(\d+)日', date_str)
        month, day = int(match.group(1)), int(match.group(2))
        current_year = datetime.now().year
        target_date = datetime(current_year, month, day)
    else:
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            util_logger.warning(f"Could not parse date '{date_str}', defaulting to yesterday.")
            target_date = datetime.now() - timedelta(days=1)
    
    start_time = target_date.replace(hour=0, minute=0, second=0)
    end_time = target_date.replace(hour=23, minute=59, second=59)
    
    return start_time.strftime("%Y-%m-%d %H:%M:%S"), end_time.strftime("%Y-%m-%d %H:%M:%S")

def extract_content(content: str) -> str:
    """
    Cleans a message content string by removing thought processes and tool call tags.
    """
    if not isinstance(content, str):
        content = str(content)
    
    content = content.replace('\\n', '\n')
    
    thinking_patterns = [
        r'<think>[\s\S]*?</think>',
        r'\[thinking\][\s\S]*?\[/thinking\]',
        r'<思考>[\s\S]*?</思考>',
        r'思考[:：][\s\S]*?(?:思考结束|思考完毕)',
        r'【思考过程】[\s\S]*?(?=【回答内容】|\n\n|$)',
    ]
    
    clean_text = content
    for pattern in thinking_patterns:
        clean_text = re.sub(pattern, '', clean_text)
    
    if '<think>' in clean_text:
        util_logger.warning("Detected <think> tag after initial cleaning, attempting aggressive cleaning.")
        clean_text = re.sub(r'<think>[\s\S]*', '', clean_text)
    
    tool_patterns = [
        r'<tool_call>[\s\S]*?</tool_call>',
        r'<tool_response>[\s\S]*?</tool_response>'
    ]
    
    for pattern in tool_patterns:
        clean_text = re.sub(pattern, '', clean_text)
    
    format_patterns = [
        r'【回答内容】\n*',
        r'【思考过程】\n*',
        r'-{3,}',
        r'^(LEMMA|Assistant|AI): ',
    ]
    
    for pattern in format_patterns:
        clean_text = re.sub(pattern, '', clean_text)
    
    clean_text = re.sub(r'^\n+', '', clean_text)
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
    clean_text = clean_text.strip()
    
    if any(tag in clean_text.lower() for tag in ['<think', '</think>', '[thinking]', '【思考']):
        util_logger.warning(f"Possible thought tags remaining after cleaning: {clean_text[:100]}")
        lines = clean_text.split('\n')
        clean_lines = [line for line in lines if not any(tag in line.lower() for tag in ['<think', '</think>', '[thinking]', '【思考'])]
        clean_text = '\n'.join(clean_lines).strip()
    
    return clean_text

def find_final_assistant_message(response: Any) -> Optional[Message]:
    """
    Finds the final assistant message object from the complex response of agent.run().
    """
    if isinstance(response, list):
        for item in reversed(response):
            if hasattr(item, 'role') and item.role == 'assistant':
                if hasattr(item, 'content') and isinstance(item.content, str):
                    if '<tool_call>' not in item.content:
                        return item
            elif isinstance(item, dict) and item.get('role') == 'assistant':
                content = item.get('content', '')
                if isinstance(content, str) and '<tool_call>' not in content:
                    return Message(role='assistant', content=content)
    
    elif hasattr(response, 'role') and response.role == 'assistant':
        if hasattr(response, 'content') and '<tool_call>' not in response.content:
            return response
    
    elif isinstance(response, dict) and response.get('role') == 'assistant':
        content = response.get('content', '')
        if isinstance(content, str) and '<tool_call>' not in content:
            return Message(role='assistant', content=content)
    
    return None

def extract_tool_responses(messages: List[Message]) -> Dict[str, Any]:
    """
    Extracts and categorizes all tool responses from the message history.
    """
    tool_outputs = {}
    for msg in messages:
        if msg.role == 'tool':
            try:
                tool_name = msg.name
                tool_data = json.loads(msg.content)
                tool_outputs[tool_name] = tool_data
                util_logger.info(f"Successfully extracted response from tool '{tool_name}'.")
            except (json.JSONDecodeError, AttributeError) as e:
                util_logger.warning(f"Error parsing tool response: {e} - Content: {getattr(msg, 'content', 'N/A')}")
    
    return tool_outputs 