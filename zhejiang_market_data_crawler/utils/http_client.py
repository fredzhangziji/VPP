"""
HTTP客户端，封装请求操作
"""

import time
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from utils.logger import setup_logger
from utils.config import REQUEST_TIMEOUT, MAX_RETRIES, REQUEST_INTERVAL

logger = setup_logger('http_client')

class HttpClient:
    """HTTP客户端，封装请求操作"""
    
    def __init__(self, timeout=None, max_retries=None, interval=None):
        """
        初始化HTTP客户端
        
        Args:
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            interval: 请求间隔时间（秒）
        """
        self.timeout = timeout or REQUEST_TIMEOUT
        self.max_retries = max_retries or MAX_RETRIES
        self.interval = interval or REQUEST_INTERVAL
        self.last_request_time = 0
        self.session = self._create_session()
    
    def _create_session(self):
        """创建会话并配置重试策略"""
        session = requests.Session()
        
        # 配置重试策略
        retry_strategy = Retry(
            total=self.max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            backoff_factor=1
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # 设置通用请求头
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        })
        
        return session
    
    def _wait_for_interval(self):
        """等待请求间隔"""
        now = time.time()
        elapsed = now - self.last_request_time
        
        if elapsed < self.interval:
            # 添加一点随机性，避免请求过于规律
            sleep_time = self.interval - elapsed + random.uniform(0, 1.5)
            logger.debug(f"等待 {sleep_time:.2f} 秒后发送下一个请求")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get(self, url, params=None, headers=None, **kwargs):
        """
        发送GET请求
        
        Args:
            url: 请求URL
            params: 请求参数
            headers: 请求头
            **kwargs: 其他参数
        
        Returns:
            response: 响应对象
        """
        self._wait_for_interval()
        
        try:
            logger.info(f"发送GET请求: {url}")
            response = self.session.get(
                url,
                params=params,
                headers=headers,
                timeout=self.timeout,
                **kwargs
            )
            response.raise_for_status()
            logger.info(f"GET请求成功: {url}, 状态码: {response.status_code}")
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"GET请求失败: {url}, 错误: {e}")
            raise
    
    def post(self, url, data=None, json=None, headers=None, **kwargs):
        """
        发送POST请求
        
        Args:
            url: 请求URL
            data: 表单数据
            json: JSON数据
            headers: 请求头
            **kwargs: 其他参数
        
        Returns:
            response: 响应对象
        """
        self._wait_for_interval()
        
        try:
            logger.info(f"发送POST请求: {url}")
            # 记录请求头信息（去除敏感信息）
            safe_headers = headers.copy() if headers else {}
            if 'Cookie' in safe_headers:
                safe_headers['Cookie'] = "***隐藏***"
            if 'Authorization' in safe_headers:
                safe_headers['Authorization'] = "***隐藏***"
            logger.debug(f"请求头: {safe_headers}")
            
            # 记录请求数据前几个字符
            if data:
                logger.debug(f"请求数据(前100字符): {str(data)[:100]}...")
            
            response = self.session.post(
                url,
                data=data,
                json=json,
                headers=headers,
                timeout=self.timeout,
                **kwargs
            )
            logger.info(f"POST请求成功: {url}, 状态码: {response.status_code}")
            
            # 记录响应头和部分响应内容
            logger.debug(f"响应头: {dict(response.headers)}")
            content_type = response.headers.get('Content-Type', '')
            if 'application/json' in content_type or 'text/' in content_type:
                logger.debug(f"响应内容(前200字符): {response.text[:200]}...")
            
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"POST请求失败: {url}, 错误: {e}")
            raise
    
    def download_file(self, url, local_path, params=None, headers=None, **kwargs):
        """
        下载文件
        
        Args:
            url: 文件URL
            local_path: 本地保存路径
            params: 请求参数
            headers: 请求头
            **kwargs: 其他参数
        
        Returns:
            success: 是否下载成功
        """
        self._wait_for_interval()
        
        try:
            logger.info(f"下载文件: {url} -> {local_path}")
            response = self.session.get(
                url,
                params=params,
                headers=headers,
                timeout=self.timeout,
                stream=True,
                **kwargs
            )
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"文件下载成功: {local_path}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"文件下载失败: {url}, 错误: {e}")
            return False
        except IOError as e:
            logger.error(f"文件保存失败: {local_path}, 错误: {e}")
            return False
    
    def close(self):
        """关闭会话"""
        self.session.close()
        logger.debug("HTTP会话已关闭")
    
    def __enter__(self):
        """支持with语句"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持with语句"""
        self.close()


# 单例模式，提供全局访问点
http_client = HttpClient()


def get(url, params=None, headers=None, **kwargs):
    """
    发送GET请求的便捷函数
    
    Args:
        url: 请求URL
        params: 请求参数
        headers: 请求头
        **kwargs: 其他参数
    
    Returns:
        response: 响应对象
    """
    return http_client.get(url, params=params, headers=headers, **kwargs)


def post(url, data=None, json=None, headers=None, **kwargs):
    """
    发送POST请求的便捷函数
    
    Args:
        url: 请求URL
        data: 表单数据
        json: JSON数据
        headers: 请求头
        **kwargs: 其他参数
    
    Returns:
        response: 响应对象
    """
    return http_client.post(url, data=data, json=json, headers=headers, **kwargs)


def download_file(url, local_path, params=None, headers=None, **kwargs):
    """
    下载文件的便捷函数
    
    Args:
        url: 文件URL
        local_path: 本地保存路径
        params: 请求参数
        headers: 请求头
        **kwargs: 其他参数
    
    Returns:
        success: 是否下载成功
    """
    return http_client.download_file(url, local_path, params=params, headers=headers, **kwargs) 