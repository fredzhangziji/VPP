#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LEMMA Agent and API Configuration File
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Database Configuration ---
DB_CONFIG_VPP_SERVICE = {
    'host': os.getenv('DB_HOST', '10.5.0.10'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', 'kunyu2023rds'),
    'database': os.getenv('DB_DATABASE', 'vpp_service'),
    'port': int(os.getenv('DB_PORT', 3306))
}

# --- LLM Configuration ---
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

# --- Agent Tools Configuration ---
TOOLS = [
    'get_price_deviation_report', 
    'analyze_bidding_space_deviation',
    'analyze_power_generation_deviation', 
    'get_regional_capacity_info'
]

# --- System Prompt Configuration ---
def load_system_prompt(file_path="system_prompt.md"):
    """Loads the system prompt from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: System prompt file not found at {file_path}. Using a default message.")
        return "You are a helpful assistant."

SYSTEM_MESSAGE = load_system_prompt()

# --- API Configuration ---
API_SESSION_CLEANUP_INTERVAL_SECONDS = 3600  # 1 hour
API_SESSION_MAX_AGE_HOURS = 24  # Sessions expire after 24 hours of inactivity 