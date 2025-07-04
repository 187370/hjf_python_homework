import openai
import requests
import json
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random
from search import search_searxng


class LLMAnalyzer:
    """
    大语言模型分析器，用于股票市场分析和情感分析
    """

    def __init__(
        self,
        api_keys: List[str] = None,
        base_url: str = "http://10.129.80.218:3000/v1",
        model: str = "deepseek-v3-250324",
    ):
        """
        初始化LLM分析器

        Args:
            api_keys: DeepSeek API密钥列表，支持多个密钥以实现并行请求
            base_url: API基础URL
            model: 使用的模型名称
        """
        # 兼容性处理：如果传入单个字符串，转换为列表
        if isinstance(api_keys, str):
            self.api_keys = [api_keys]
        elif isinstance(api_keys, list):
            self.api_keys = api_keys
        else:
            self.api_keys = [
                ""
            ]  # 默认密钥

        self.base_url = base_url
        self.model = model

        # 为每个API密钥创建客户端
        self.clients = []
        for api_key in self.api_keys:
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
            self.clients.append(client)

        # 线程锁用于API密钥轮转
        self._lock = threading.Lock()
        self._current_key_index = 0

     
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _clean_json_response(self, response: str) -> str:
        """
        清理LLM响应中的代码块标记和其他非JSON内容

        Args:
            response: 原始响应文本

        Returns:
            清理后的JSON字符串
        """
        # 移除```json和```标记
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        # 移除首尾空白字符
        cleaned = cleaned.strip()

        # 尝试找到JSON对象的开始和结束
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            cleaned = cleaned[start_idx : end_idx + 1]

        return cleaned

    def _get_next_client(self):
        """获取下一个可用的客户端，实现API密钥轮转"""
        with self._lock:
            client = self.clients[self._current_key_index]
            self._current_key_index = (self._current_key_index + 1) % len(self.clients)
            return client

    def _make_request(
        self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 1000
    ) -> str:
        """
        向大语言模型发送请求（串行版本，也废弃了）

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            模型的响应文本
        """
        client = self._get_next_client()

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"API请求失败: {e}")
            return ""

    def _make_parallel_request(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        client_index: int = 0,
    ) -> str:
        """
        并行请求的单个调用，使用指定的客户端

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            client_index: 客户端索引

        Returns:
            模型的响应文本
        """
        try:
            client = self.clients[client_index % len(self.clients)]
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"并行API请求失败 (客户端 {client_index}): {e}")
            return ""

    def batch_analyze_stocks_parallel(self, analysis_tasks: List[Dict]) -> List[Dict]:
        """
        并行处理多个股票的分析任务
        Args:
            analysis_tasks: 分析任务列表，每个任务包含股票代码和相关数据
        Returns:
            分析结果列表
        """
        self.logger.info(f"开始并行分析 {len(analysis_tasks)} 个股票...")

        results = []

        # 使用线程池进行并行处理
        with ThreadPoolExecutor(
            max_workers=min(len(self.clients), len(analysis_tasks))
        ) as executor:
            # 提交所有任务
            future_to_task = {}

            for i, task in enumerate(analysis_tasks):
                client_index = i % len(self.clients)

                if task["task_type"] == "sentiment":
                    future = executor.submit(
                        self._analyze_sentiment_single,
                        task["stock_symbol"],
                        task["recent_data"],
                        task.get("news_headlines"),
                        task.get("current_date"),
                        task.get("pnl_history"),
                        client_index,
                    )
                elif task["task_type"] == "trading_signal":
                    future = executor.submit(
                        self._generate_trading_signal_single,
                        task["stock_symbol"],
                        task["technical_indicators"],
                        task["market_sentiment"],
                        task.get("holding_info"),
                        task.get("latest_price"),
                        client_index,
                    )

                future_to_task[future] = task

    
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(
                        {
                            "stock_symbol": task["stock_symbol"],
                            "task_type": task["task_type"],
                            "result": result,
                            "success": True,
                        }
                    )
                except Exception as e:
                    self.logger.error(f"任务失败 {task['stock_symbol']}: {e}")
                    results.append(
                        {
                            "stock_symbol": task["stock_symbol"],
                            "task_type": task["task_type"],
                            "result": None,
                            "success": False,
                            "error": str(e),
                        }
                    )

        self.logger.info(
            f"并行分析完成，成功: {sum(1 for r in results if r['success'])}/{len(results)}"
        )
        return results

    def _analyze_sentiment_single(
        self,
        stock_symbol: str,
        recent_data: pd.DataFrame,
        news_headlines: List[str] = None,
        current_date: Optional[str] = None,
        pnl_history: Optional[List[Tuple[str, float]]] = None,
        client_index: int = 0,
    ) -> Dict[str, Any]:
        """
        单个股票情感分析（并行版本）
        """
        if recent_data.empty:
            return {
                "sentiment_score": 0.5,
                "confidence": 0.5,
                "reasoning": "无数据",
                "risk_level": "中",
            }

        # 计算基本统计信息
        latest_price = recent_data["Close"].iloc[-1]
        price_change = (
            (latest_price - recent_data["Close"].iloc[0])
            / recent_data["Close"].iloc[0]
            * 100
        )
        volatility = recent_data["Close"].pct_change().std() * 100
        volume_trend = (
            recent_data["Volume"].iloc[-5:].mean()
            / recent_data["Volume"].iloc[:-5].mean()
            if len(recent_data) >= 10
            else 1
        )

        # 30天历史价格变化趋势
        price_history = []
        if len(recent_data) >= 30:
            for i in range(-30, 0, 1):  # 每5天采样一次
                idx = max(0, len(recent_data) + i)
                if idx < len(recent_data):
                    price_history.append(
                        f"{recent_data.iloc[idx]['Date'].strftime('%m-%d')}: ${recent_data.iloc[idx]['Close']:.2f}"
                    )

        # 公司背景信息
        company_info = self._get_company_context(stock_symbol)

        search_summaries = []
        if current_date:
            try:
                search_results = search_searxng(
                    f"{stock_symbol} 股票",
                    current_date=str(current_date),
                    time_range=30,
                    category="news",
                    engines="google,bing",
                    return_results=True,
                )
                for r in search_results[:3]:
                    summary = r.get("content") or r.get("title")
                    if summary:
                        search_summaries.append(summary.strip())
            except Exception as e:
                self.logger.warning(f"搜索信息获取失败 {stock_symbol}: {e}")
   
        prompt = f"""
今天是 {current_date or '未知日期'}，请结合该日期前的市场信息分析股票 {stock_symbol} 的情绪和前景。
紧密联系{current_date}日期附近有关该公司的新闻（可以用你的世界知识），分析详细理由，并按指定格式回复

=== 公司背景 ===
{company_info}

=== 技术分析摘要 ===
- 最新价格: ${latest_price:.2f}
- 期间价格变化: {price_change:.2f}%
- 价格波动率: {volatility:.2f}%
- 成交量趋势比率: {volume_trend:.2f}

=== 30天价格历史轨迹 ===
{chr(10).join(price_history) if price_history else "数据不足"}
"""
        if pnl_history:
            pnl_text = ", ".join(f"{d}: {p:+.2f}" for d, p in pnl_history)
            prompt += f"\n=== 历史盈亏(单股) ===\n{pnl_text}"

        if search_summaries:
            prompt += "\n=== 相关新闻摘要 ===\n" + "\n".join(
                f"- {s}" for s in search_summaries
            )

        prompt += """

=== 市场环境分析 ===
请结合以上信息，从以下维度进行综合分析：
1. 技术面分析（价格趋势、成交量确认、波动性）
2. 基本面考量（行业地位、业务前景、宏观影响）
3. 市场情绪（投资者信心、风险偏好、流动性）

请以JSON格式回复：
{{
    "sentiment_score": 0.7,
    "confidence": 0.8,
    "reasoning": "详细分析理由，包含技术面和基本面见解",
    "key_factors": ["关键因素1", "关键因素2", "关键因素3"],
    "risk_level": "低/中/高",
    "price_momentum": "上升/下降/震荡",
    "volume_confirmation": "强/中/弱"
}}
"""

        messages = [
            {
                "role": "system",
                "content": "你是一位经验丰富的金融分析师，擅长股票市场分析和情感分析。",
            },
            {"role": "user", "content": prompt},
        ]

        response = self._make_parallel_request(
            messages, temperature=0.3, client_index=client_index
        )
        try:
            cleaned_response = self._clean_json_response(response)
            result = json.loads(cleaned_response)
            return result
        except json.JSONDecodeError:
            self.logger.warning(f"无法解析LLM响应为JSON: {response}")
            return {
                "sentiment_score": 0.5,
                "confidence": 0.5,
                "reasoning": response,
                "key_factors": [],
                "risk_level": "中",
            }

    def _generate_trading_signal_single(
        self,
        stock_symbol: str,
        technical_indicators: Dict[str, float],
        market_sentiment: Dict[str, Any],
        holding_info: Optional[Dict[str, Any]] = None,
        latest_price: Optional[float] = None,
        client_index: int = 0,
    ) -> Dict[str, Any]:
        """
        单个股票交易信号生成（并行版本）
        """
        # 提取市场情感中的详细分析理由和关键因素
        sentiment_reasoning = market_sentiment.get("reasoning", "")
        key_factors = market_sentiment.get("key_factors", [])
        key_factors_text = ", ".join(key_factors) if key_factors else "无明确关键因素"

        holding_text = ""
        if holding_info:
            shares = holding_info.get("shares", 0)
            value = holding_info.get("value", 0.0)
            weight_pct = holding_info.get("weight_pct", 0.0)
            cash = holding_info.get("cash", 0.0)
            holding_text = f"当前持仓 {shares} 股, 价值约 ${value:.2f}, 占组合 {weight_pct:.2%}; 当前现金 ${cash:.2f}\n"

        price_text = (
            f"当前股价: ${latest_price:.2f}\n" if latest_price is not None else ""
        )
        prompt = f"""
作为量化交易分析师，请为股票 {stock_symbol} 生成交易信号：
{holding_text}{price_text}

技术指标：
- RSI: {technical_indicators.get('rsi', 50):.2f}
- MACD: {technical_indicators.get('macd', 0):.4f}
- SMA比率: {technical_indicators.get('sma_ratio', 1):.4f}
- ATR: {technical_indicators.get('atr', 0):.4f}
- 成交量比率: {technical_indicators.get('volume_ratio', 1):.4f}

市场情感：
- 情感评分: {market_sentiment.get('sentiment_score', 0.5):.3f}
- 置信度: {market_sentiment.get('confidence', 0.5):.3f}
- 风险级别: {market_sentiment.get('risk_level', '中')}

市场情感分析理由：
{sentiment_reasoning}

关键影响因素：
{key_factors_text}

请基于技术分析和情感分析，提供交易建议。

请以JSON格式回复：
{{
    "signal": "买入/卖出/持有",
    "confidence": 0.8,
    "target_weight": 0.05,
    "stop_loss": 0.95,
    "take_profit": 1.10,
    "holding_period": 10,
    "trade_shares": 100,
    "reasoning": "详细理由"
}}
"""
        messages = [
            {
                "role": "system",
                "content": "你是一位专业的量化交易分析师，擅长结合技术分析和基本面分析生成交易信号。",
            },
            {"role": "user", "content": prompt},
        ]

        response = self._make_parallel_request(
            messages, temperature=0.2, client_index=client_index
        )
        try:
            result = json.loads(self._clean_json_response(response))
            return result
        except json.JSONDecodeError:
            self.logger.warning(f"无法解析LLM响应为JSON: {response}")
            return {
                "signal": "持有",
                "confidence": 0.5,
                "target_weight": 0.0,
                "stop_loss": 0.95,
                "take_profit": 1.05,
                "holding_period": 5,
                "trade_shares": 0,
                "reasoning": response,
            }

    def analyze_market_sentiment(
        self,
        stock_symbol: str,
        recent_data: pd.DataFrame,
        news_headlines: List[str] = None,
        current_date: Optional[str] = None,
        pnl_history: Optional[List[Tuple[str, float]]] = None,
    ) -> Dict[str, Any]:
        """
        分析市场情感和股票前景（串行版本，已废弃）

        Args:
            stock_symbol: 股票代码
            recent_data: 最近的股票数据
            news_headlines: 相关新闻标题列表

        Returns:
            分析结果字典
        """
        # 准备股票数据摘要
        if len(recent_data) > 0:
            latest_price = recent_data["Close"].iloc[-1]
            price_change = (
                (latest_price - recent_data["Close"].iloc[0])
                / recent_data["Close"].iloc[0]
                * 100
            )
            volatility = recent_data["Close"].pct_change().std() * 100
            volume_trend = (
                recent_data["Volume"].rolling(5).mean().iloc[-1]
                / recent_data["Volume"].rolling(20).mean().iloc[-1]
            )
        else:
            return {"sentiment_score": 0.5, "confidence": 0.0, "reasoning": "数据不足"}

        # 获取搜索信息
        search_summaries = []
        if current_date:
            try:
                search_results = search_searxng(
                    f"{stock_symbol} 股票",
                    current_date=str(current_date),
                    time_range=30,
                    category="news",
                    engines="google,bing",
                    return_results=True,
                )
                for r in search_results[:3]:
                    summary = r.get("content") or r.get("title")
                    if summary:
                        search_summaries.append(summary.strip())
            except Exception as e:
                self.logger.warning(f"搜索信息获取失败 {stock_symbol}: {e}")

        # 构建提示词
        prompt = f"""
今天是 {current_date or '未知日期'}，请分析股票 {stock_symbol} 的市场情感和投资前景。

股票数据摘要：
- 最新收盘价: ${latest_price:.2f}
- 近期价格变化: {price_change:.2f}%
- 价格波动率: {volatility:.2f}%
- 成交量趋势比率: {volume_trend:.2f}

"""
        if pnl_history:
            pnl_text = ", ".join(f"{d}: {p:+.2f}" for d, p in pnl_history)
            prompt += f"历史盈亏(单股): {pnl_text}\n"

        if search_summaries:
            prompt += "相关新闻摘要:\n" + "\n".join(f"- {s}" for s in search_summaries) + "\n"

        if news_headlines:
            prompt += f"\n相关新闻标题：\n"
            for i, headline in enumerate(news_headlines[:5], 1):
                prompt += f"{i}. {headline}\n"

        prompt += """
请提供以下分析：
1. 市场情感评分 (0-1，0为极度悲观，1为极度乐观)
2. 分析置信度 (0-1)
3. 详细分析理由

请以JSON格式回复：
{
    "sentiment_score": 0.7,
    "confidence": 0.8,
    "reasoning": "详细分析理由",
    "key_factors": ["因素1", "因素2"],
    "risk_level": "低/中/高"
}
"""
        messages = [
            {
                "role": "system",
                "content": "你是一位经验丰富的金融分析师，擅长股票市场分析和情感分析。",
            },
            {"role": "user", "content": prompt},
        ]

        response = self._make_request(messages, temperature=0.3)
        try:
           
            cleaned_response = self._clean_json_response(response)
            result = json.loads(cleaned_response)
            return result
        except json.JSONDecodeError:
           
            self.logger.warning(f"无法解析LLM响应为JSON: {response}")
            return {
                "sentiment_score": 0.5,
                "confidence": 0.5,
                "reasoning": response,
                "key_factors": [],
                "risk_level": "中",
            }

    def analyze_stock_relationships(
        self, stock_symbols: List[str], correlation_matrix: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        分析股票之间的关系和行业关联

        Args:
            stock_symbols: 股票代码列表
            correlation_matrix: 相关性矩阵

        Returns:
            关系分析结果，包含行业聚类和有向依赖关系
        """
   
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  
                    high_corr_pairs.append(
                        {
                            "stock1": correlation_matrix.columns[i],
                            "stock2": correlation_matrix.columns[j],
                            "correlation": corr_value,
                        }
                    )

        prompt = f"""
作为金融行业专家，请分析以下股票组合的行业关系和投资组合建议：

股票列表: {', '.join(stock_symbols)}

高相关性股票对：
"""

        for pair in high_corr_pairs:
            prompt += (
                f"- {pair['stock1']} vs {pair['stock2']}: {pair['correlation']:.3f}\n"
            )

        prompt += """
请提供：
1. 这些股票的主要行业分布
2. 投资组合多样化建议
3. 潜在的系统性风险
4. 建议的权重分配策略
5. 哪些股票在行业内起领涨作用，哪些可能跟随，请给出有向依赖关
   系描述

请以JSON格式回复：
{
    "industry_clusters": {"Technology": ["AAPL", "MSFT"]},
    "diversification_score": 0.8,
    "systematic_risks": ["风险1", "风险2"],
    "weight_suggestions": {"AAPL": 0.1, "MSFT": 0.15},
    "directed_relations": [{"from": "AAPL", "to": "MSFT", "reason": "AAPL领涨"}],
    "overall_strategy": "整体策略建议"
}
"""
        messages = [
            {
                "role": "system",
                "content": "你是一位专业的投资组合经理，擅长行业分析和风险管理。",
            },
            {"role": "user", "content": prompt},
        ]

        response = self._make_request(messages, temperature=0.3, max_tokens=1500)
        try:
            result = json.loads(self._clean_json_response(response))
            return result
        except json.JSONDecodeError:
            self.logger.warning(f"无法解析LLM响应为JSON: {response}")
            return {
                "industry_clusters": {},
                "industry_analysis": response,
                "diversification_score": 0.5,
                "systematic_risks": [],
                "weight_suggestions": {},
                "directed_relations": [],
                "overall_strategy": "需要进一步分析",
            }

    def generate_trading_signals(
        self,
        stock_symbol: str,
        technical_indicators: Dict[str, float],
        market_sentiment: Dict[str, Any],
        holding_info: Optional[Dict[str, Any]] = None,
        latest_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        基于技术指标和市场情感生成交易信号（串行版本，已废弃）

        Args:
            stock_symbol: 股票代码
            technical_indicators: 技术指标字典
            market_sentiment: 市场情感分析结果

        Returns:
            交易信号建议
        """
        holding_text = ""
        if holding_info:
            shares = holding_info.get("shares", 0)
            value = holding_info.get("value", 0.0)
            weight_pct = holding_info.get("weight_pct", 0.0)
            cash = holding_info.get("cash", 0.0)
            holding_text = f"当前持仓 {shares} 股, 价值约 ${value:.2f}, 占组合 {weight_pct:.2%}; 当前现金 ${cash:.2f}\n"
        price_text = (
            f"当前股价: ${latest_price:.2f}\n" if latest_price is not None else ""
        )
        prompt = f"""
作为量化交易专家，请为股票 {stock_symbol} 生成交易信号。
{holding_text}{price_text}

技术指标：
"""

        for indicator, value in technical_indicators.items():
            prompt += f"- {indicator}: {value:.4f}\n"

        prompt += f"""
市场情感分析：
- 情感评分: {market_sentiment.get('sentiment_score', 0.5):.3f}
- 置信度: {market_sentiment.get('confidence', 0.5):.3f}
- 风险级别: {market_sentiment.get('risk_level', '中')}

请基于技术分析和情感分析，提供交易建议。

请以JSON格式回复：
{{
    "signal": "买入/卖出/持有",
    "confidence": 0.8,
    "target_weight": 0.05,
    "stop_loss": 0.95,
    "take_profit": 1.10,
    "holding_period": 10,
    "trade_shares": 100,
    "reasoning": "详细理由"
}}
"""
        messages = [
            {
                "role": "system",
                "content": "你是一位专业的量化交易分析师，擅长结合技术分析和基本面分析生成交易信号。",
            },
            {"role": "user", "content": prompt},
        ]

        response = self._make_request(messages, temperature=0.2)
        try:
            result = json.loads(self._clean_json_response(response))
            return result
        except json.JSONDecodeError:
            self.logger.warning(f"无法解析LLM响应为JSON: {response}")
            return {
                "signal": "持有",
                "confidence": 0.5,
                "target_weight": 0.0,
                "stop_loss": 0.95,
                "take_profit": 1.05,
                "holding_period": 5,
                "trade_shares": 0,
                "reasoning": response,
            }

#     def analyze_risk_factors(
#         self, portfolio_data: Dict[str, pd.DataFrame], market_data: pd.DataFrame
#     ) -> Dict[str, Any]:
#         """
#         分析投资组合的风险因素

#         Args:
#             portfolio_data: 投资组合中各股票的历史数据
#             market_data: 市场整体数据

#         Returns:
#             风险分析结果
#         """
#         # 计算投资组合的基本风险指标
#         portfolio_returns = []
#         for symbol, data in portfolio_data.items():
#             if len(data) > 1:
#                 returns = data["Close"].pct_change().dropna()
#                 portfolio_returns.append(returns)

#         if portfolio_returns:
#             portfolio_returns = pd.concat(portfolio_returns, axis=1)
#             portfolio_vol = portfolio_returns.std().mean()
#             portfolio_correlation = portfolio_returns.corr().mean().mean()
#         else:
#             portfolio_vol = 0
#             portfolio_correlation = 0

#         prompt = f"""
# 作为风险管理专家，请分析当前投资组合的风险状况：

# 投资组合统计：
# - 股票数量: {len(portfolio_data)}
# - 平均波动率: {portfolio_vol:.4f}
# - 平均相关性: {portfolio_correlation:.4f}

# 请评估：
# 1. 系统性风险水平
# 2. 非系统性风险
# 3. 流动性风险
# 4. 集中度风险
# 5. 风险管理建议

# 回复格式：
# {
#     "systematic_risk": "低/中/高",
#     "idiosyncratic_risk": "低/中/高", 
#     "liquidity_risk": "低/中/高",
#     "concentration_risk": "低/中/高",
#     "overall_risk_score": 0.6,
#     "risk_recommendations": ["建议1", "建议2"],
#     "stress_test_scenarios": ["场景1", "场景2"]
# }
# """
#         messages = [
#             {
#                 "role": "system",
#                 "content": "你是一位专业的投资风险管理专家，擅长投资组合风险评估。",
#             },
#             {"role": "user", "content": prompt},
#         ]

#         response = self._make_request(messages, temperature=0.3)
#         try:
#             result = json.loads(self._clean_json_response(response))
#             return result
#         except json.JSONDecodeError:
#             self.logger.warning(f"无法解析LLM响应为JSON: {response}")
#             return {
#                 "systematic_risk": "中",
#                 "idiosyncratic_risk": "中",
#                 "liquidity_risk": "中",
#                 "concentration_risk": "中",
#                 "overall_risk_score": 0.5,
#                 "risk_recommendations": [response],
#                 "stress_test_scenarios": [],
#             }

    def _get_company_context(self, stock_symbol: str) -> str:
        """
        获取公司背景信息（基于股票代码推断）

        Args:
            stock_symbol: 股票代码

        Returns:
            公司背景信息字符串
        """
        company_info = {
            "AAPL": {
                "name": "Apple Inc.",
                "industry": "科技硬件",
                "description": "全球领先的消费电子产品制造商，主要产品包括iPhone、iPad、Mac等",
                "market_cap": "大盘股",
                "key_factors": ["创新能力", "品牌价值", "生态系统", "全球供应链"],
            },
            "MSFT": {
                "name": "Microsoft Corporation",
                "industry": "软件服务",
                "description": "全球最大的软件公司之一，专注于云计算、生产力软件和企业服务",
                "market_cap": "大盘股",
                "key_factors": [
                    "云计算领导地位",
                    "Office套件",
                    "Azure平台",
                    "企业客户基础",
                ],
            },
            "GOOGL": {
                "name": "Alphabet Inc.",
                "industry": "互联网技术",
                "description": "谷歌母公司，主营搜索引擎、在线广告、云计算和人工智能",
                "market_cap": "大盘股",
                "key_factors": ["搜索垄断地位", "广告收入", "AI技术", "多元化业务"],
            },
            "AMZN": {
                "name": "Amazon.com Inc.",
                "industry": "电商与云服务",
                "description": "全球最大的电商平台和云服务提供商",
                "market_cap": "大盘股",
                "key_factors": ["电商领导地位", "AWS云服务", "物流网络", "创新文化"],
            },
            "TSLA": {
                "name": "Tesla Inc.",
                "industry": "电动汽车",
                "description": "电动汽车和清洁能源公司的领导者",
                "market_cap": "大盘股",
                "key_factors": ["电动车技术", "自动驾驶", "能源存储", "CEO影响力"],
            },
            "NVDA": {
                "name": "NVIDIA Corporation",
                "industry": "半导体",
                "description": "图形处理器和人工智能芯片的领先制造商",
                "market_cap": "大盘股",
                "key_factors": ["GPU技术", "AI计算", "数据中心", "游戏市场"],
            },
            "JPM": {
                "name": "JPMorgan Chase & Co.",
                "industry": "金融服务",
                "description": "美国最大的银行之一，提供投资银行和零售银行服务",
                "market_cap": "大盘股",
                "key_factors": ["银行业务", "投资银行", "风险管理", "金融科技"],
            },
            "JNJ": {
                "name": "Johnson & Johnson",
                "industry": "医疗保健",
                "description": "多元化医疗保健公司，涵盖制药、医疗设备和消费品",
                "market_cap": "大盘股",
                "key_factors": ["药物研发", "医疗设备", "品牌组合", "稳定收益"],
            },
            "V": {
                "name": "Visa Inc.",
                "industry": "金融技术",
                "description": "全球领先的数字支付技术公司",
                "market_cap": "大盘股",
                "key_factors": ["支付网络", "数字化转型", "全球覆盖", "交易费用"],
            },
            "PG": {
                "name": "Procter & Gamble Co.",
                "industry": "消费品",
                "description": "全球最大的消费品公司之一",
                "market_cap": "大盘股",
                "key_factors": ["品牌组合", "全球分销", "创新研发", "稳定分红"],
            },
            "AABA": {
                "name": "Altaba Inc.",
                "industry": "互联网投资",
                "description": "前雅虎公司资产管理实体",
                "market_cap": "中大盘股",
                "key_factors": ["资产价值", "投资组合", "清算进程"],
            },
            "AXP": {
                "name": "American Express Company",
                "industry": "金融服务",
                "description": "全球知名信用卡和旅行服务提供商",
                "market_cap": "大盘股",
                "key_factors": ["品牌声誉", "支付网络", "客户忠诚度"],
            },
            "BA": {
                "name": "The Boeing Company",
                "industry": "航空航天",
                "description": "全球领先的飞机和防务产品制造商",
                "market_cap": "大盘股",
                "key_factors": ["商用飞机", "国防合同", "技术创新"],
            },
            "CAT": {
                "name": "Caterpillar Inc.",
                "industry": "工程机械",
                "description": "全球最大的重型机械和发动机制造商之一",
                "market_cap": "大盘股",
                "key_factors": ["全球基建", "矿业需求", "品牌影响力"],
            },
            "CSCO": {
                "name": "Cisco Systems, Inc.",
                "industry": "网络设备",
                "description": "领先的企业级网络和通信解决方案供应商",
                "market_cap": "大盘股",
                "key_factors": ["路由器交换机", "企业客户", "云和安全业务"],
            },
            "CVX": {
                "name": "Chevron Corporation",
                "industry": "能源",
                "description": "全球综合性的石油和天然气公司",
                "market_cap": "大盘股",
                "key_factors": ["油气储量", "价格波动", "全球运营"],
            },
            "DIS": {
                "name": "The Walt Disney Company",
                "industry": "传媒娱乐",
                "description": "涵盖影视制作、主题公园和流媒体业务",
                "market_cap": "大盘股",
                "key_factors": ["内容版权", "品牌资产", "全球旅游业"],
            },
            "GE": {
                "name": "General Electric Company",
                "industry": "工业制造",
                "description": "多元化工业集团，业务涉足电力、航空等领域",
                "market_cap": "大盘股",
                "key_factors": ["产业多元化", "技术实力", "业务重组"],
            },
            "GS": {
                "name": "The Goldman Sachs Group, Inc.",
                "industry": "金融服务",
                "description": "全球领先的投资银行和资产管理公司",
                "market_cap": "大盘股",
                "key_factors": ["投行业务", "交易收入", "资产管理"],
            },
            "HD": {
                "name": "The Home Depot, Inc.",
                "industry": "零售",
                "description": "全球最大的家居装修零售连锁企业",
                "market_cap": "大盘股",
                "key_factors": ["住宅市场", "供应链", "品牌忠诚"],
            },
            "IBM": {
                "name": "International Business Machines Corporation",
                "industry": "信息技术",
                "description": "历史悠久的IT和咨询服务提供商",
                "market_cap": "大盘股",
                "key_factors": ["企业服务", "混合云", "人工智能"],
            },
            "INTC": {
                "name": "Intel Corporation",
                "industry": "半导体",
                "description": "全球知名的CPU和芯片制造商",
                "market_cap": "大盘股",
                "key_factors": ["制程工艺", "PC市场", "数据中心"],
            },
            "KO": {
                "name": "The Coca-Cola Company",
                "industry": "饮料",
                "description": "全球最大的碳酸饮料和饮品公司",
                "market_cap": "大盘股",
                "key_factors": ["品牌影响", "产品多元", "全球分销"],
            },
            "MCD": {
                "name": "McDonald's Corporation",
                "industry": "餐饮",
                "description": "全球最大的快餐连锁企业",
                "market_cap": "大盘股",
                "key_factors": ["门店规模", "品牌知名度", "供应链管理"],
            },
            "MMM": {
                "name": "3M Company",
                "industry": "工业用品",
                "description": "多元化科技和工业产品制造商",
                "market_cap": "大盘股",
                "key_factors": ["创新能力", "工业材料", "全球市场"],
            },
            "MRK": {
                "name": "Merck & Co., Inc.",
                "industry": "制药",
                "description": "全球领先的医药和疫苗研发生产企业",
                "market_cap": "大盘股",
                "key_factors": ["新药管线", "专利保护", "全球销售"],
            },
            "NKE": {
                "name": "Nike, Inc.",
                "industry": "运动服饰",
                "description": "全球知名的运动鞋服品牌",
                "market_cap": "大盘股",
                "key_factors": ["品牌营销", "创新设计", "全球供应链"],
            },
            "PFE": {
                "name": "Pfizer Inc.",
                "industry": "制药",
                "description": "大型跨国制药企业，产品涵盖多种治疗领域",
                "market_cap": "大盘股",
                "key_factors": ["药品研发", "疫苗业务", "全球市场"],
            },
            "TRV": {
                "name": "The Travelers Companies, Inc.",
                "industry": "保险",
                "description": "美国主要的商业和个人财产保险公司",
                "market_cap": "大盘股",
                "key_factors": ["保费收入", "风险管理", "投资收益"],
            },
            "UNH": {
                "name": "UnitedHealth Group Incorporated",
                "industry": "医疗保险",
                "description": "美国最大的医疗保险和健康服务公司",
                "market_cap": "大盘股",
                "key_factors": ["医疗管理", "保险会员", "多元化服务"],
            },
            "UTX": {
                "name": "United Technologies Corporation",
                "industry": "工业制造",
                "description": "提供航空航天和建筑技术的多元化集团",
                "market_cap": "大盘股",
                "key_factors": ["航空发动机", "电梯业务", "全球客户"],
            },
            "VZ": {
                "name": "Verizon Communications Inc.",
                "industry": "电信",
                "description": "美国主要的通信和网络服务提供商",
                "market_cap": "大盘股",
                "key_factors": ["无线网络", "宽带服务", "规模优势"],
            },
            "WMT": {
                "name": "Walmart Inc.",
                "industry": "零售",
                "description": "全球最大的连锁零售商",
                "market_cap": "大盘股",
                "key_factors": ["价格优势", "供应链", "全球采购"],
            },
            "XOM": {
                "name": "Exxon Mobil Corporation",
                "industry": "能源",
                "description": "全球领先的石油和天然气开采与炼化公司",
                "market_cap": "大盘股",
                "key_factors": ["油气储量", "炼化能力", "全球运营"],
            },
        }

        if stock_symbol in company_info:
            info = company_info[stock_symbol]
            return f"""
公司名称: {info['name']}
所属行业: {info['industry']}
业务描述: {info['description']}
市值规模: {info['market_cap']}
关键优势: {', '.join(info['key_factors'])}
            """.strip()
        else:
            # 对于未知股票，返回通用信息
            return f"""
股票代码: {stock_symbol}
注意: 该股票的详细公司信息暂未收录，建议关注其所属行业趋势和基本面数据。
分析时请重点关注技术指标和市场表现。
            """.strip()
