import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import ta
from datetime import datetime, timedelta
import warnings
import networkx as nx
from collections import defaultdict
import json
from llm_analyzer import LLMAnalyzer
import logging

warnings.filterwarnings("ignore")


class EnhancedTradingStrategy:
    """
    增强版交易策略，集成大语言模型分析
    """

    def __init__(
        self, stock_pool, llm_api_keys, llm_base_url="http://10.129.80.218:3000/v1"
    ):
        """
        初始化增强版交易策略

        Args:
            stock_pool: 允许交易的股票代码列表
            llm_api_keys: 大语言模型API密钥列表（支持多个密钥）
            llm_base_url: 大语言模型API基础URL
        """
        self.stock_pool = stock_pool
        self.models = {}  # 存储每个股票的预测模型
        self.scalers = {}  # 存储每个股票的数据缩放器
        self.clusters = None  # 存储聚类结果
        self.cluster_models = {}  # 每个类别的模型
        self.cluster_scalers = {}  # 每个类别的scaler
        self.stock2cluster = {}  # 股票到类别的映射
        self.n_clusters = 5  # 聚类类别数
        self.window_size = 20  # 时间窗口大小
        self.prediction_days = 5  # 预测未来天数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 新增：LLM分析器和关系网络（支持多API密钥）
        if isinstance(llm_api_keys, str):
            llm_api_keys = [llm_api_keys]
        self.llm_analyzer = LLMAnalyzer(llm_api_keys, llm_base_url)
        self.stock_network = nx.Graph()  # 股票关系网络
        self.sentiment_history = {}  # 情感分析历史
        self.llm_signals = {}  # LLM生成的交易信号
        self.risk_assessments = {}  # 风险评估结果

        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _build_model(self, input_dim):
        """构建LSTM模型"""

        class LSTM(nn.Module):
            def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1):
                super(LSTM, self).__init__()
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers

                self.lstm = nn.LSTM(
                    input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2
                )

                self.fc = nn.Sequential(
                    nn.Linear(hidden_dim, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, output_dim),
                )

            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(
                    x.device
                )
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(
                    x.device
                )

                out, _ = self.lstm(x, (h0, c0))
                out = self.fc(out[:, -1, :])
                return out

        model = LSTM(input_dim=input_dim).to(self.device)
        return model

    def _create_features(self, df):
        """创建技术指标特征，增加LLM情感特征"""
        # 基础特征：价格、交易量
        df["returns"] = df["Close"].pct_change()
        df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
        df["log_volume"] = np.log(df["Volume"] + 1)  # 趋势指标
        df["sma_10"] = (
            ta.trend.sma_indicator(df["Close"], window=10) if len(df) >= 10 else np.nan
        )
        df["sma_20"] = (
            ta.trend.sma_indicator(df["Close"], window=20) if len(df) >= 20 else np.nan
        )
        df["sma_50"] = (
            ta.trend.sma_indicator(df["Close"], window=50) if len(df) >= 50 else np.nan
        )
        df["macd"] = ta.trend.macd_diff(df["Close"]) if len(df) >= 26 else np.nan

        # ADX需要更多数据，使用try-except保护
        try:
            if len(df) >= 20:  # ADX通常需要更多数据点
                df["adx"] = ta.trend.adx(df["High"], df["Low"], df["Close"])
            else:
                df["adx"] = np.nan
        except (IndexError, ValueError):
            df["adx"] = np.nan  # 波动性指标
        try:
            if len(df) >= 14:
                df["atr"] = ta.volatility.average_true_range(
                    df["High"], df["Low"], df["Close"]
                )
            else:
                df["atr"] = np.nan
        except (IndexError, ValueError):
            df["atr"] = np.nan

        try:
            if len(df) >= 20:
                df["bollinger_high"] = ta.volatility.bollinger_hband(df["Close"])
                df["bollinger_low"] = ta.volatility.bollinger_lband(df["Close"])
            else:
                df["bollinger_high"] = np.nan
                df["bollinger_low"] = np.nan
        except (IndexError, ValueError):
            df["bollinger_high"] = np.nan
            df["bollinger_low"] = np.nan

        # 动量指标
        try:
            if len(df) >= 14:
                df["rsi"] = ta.momentum.rsi(df["Close"])
                df["stoch"] = ta.momentum.stoch(df["High"], df["Low"], df["Close"])
                df["cci"] = ta.trend.cci(df["High"], df["Low"], df["Close"])
            else:
                df["rsi"] = 50.0  # RSI默认值50为中性
                df["stoch"] = 50.0  # 默认值50
                df["cci"] = 0.0  # CCI默认值0为中性
        except (IndexError, ValueError):
            df["rsi"] = 50.0
            df["stoch"] = 50.0
            df["cci"] = 0.0

        # 新增：集成LLM情感特征
        stock_symbol = df["Name"].iloc[0] if "Name" in df.columns else "UNKNOWN"
        if stock_symbol in self.sentiment_history:
            # 使用最近的情感分析结果
            recent_sentiment = (
                self.sentiment_history[stock_symbol][-1]
                if self.sentiment_history[stock_symbol]
                else 0.5
            )
            df["llm_sentiment"] = recent_sentiment
        else:
            df["llm_sentiment"] = 0.5  # 中性情感

        # 额外特征 - 基于聚类结果添加
        if self.clusters is not None and stock_symbol in self.clusters:
            cluster_id = self.clusters[stock_symbol]
            df["cluster"] = cluster_id
        else:
            df["cluster"] = 0

        return df

    def build_stock_relationship_network(self, all_data):
        """
        构建股票关系网络，基于相关性和LLM分析

        Args:
            all_data: 所有股票的历史数据
        """
        self.logger.info("构建股票关系网络...")

        # 计算股票收益率相关性
        returns_data = {}
        for stock, data in all_data.items():
            if len(data) > 20:  # 确保有足够的数据
                returns = data["Close"].pct_change().dropna()
                returns_data[stock] = returns

        if len(returns_data) < 2:
            return

        # 创建相关性矩阵
        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr()

        # 使用LLM分析股票关系
        try:
            relationship_analysis = self.llm_analyzer.analyze_stock_relationships(
                list(returns_data.keys()), correlation_matrix
            )
            self.logger.info(f"LLM关系分析: {relationship_analysis}")
        except Exception as e:
            self.logger.error(f"LLM关系分析失败: {e}")
            relationship_analysis = {}

        # 构建网络图
        self.stock_network.clear()

        # 添加节点
        for stock in returns_data.keys():
            self.stock_network.add_node(stock)

        # 添加边（基于相关性阈值）
        correlation_threshold = 0.3
        for i, stock1 in enumerate(correlation_matrix.columns):
            for j, stock2 in enumerate(correlation_matrix.columns):
                if i < j:  # 避免重复
                    corr = correlation_matrix.loc[stock1, stock2]
                    if abs(corr) > correlation_threshold:
                        self.stock_network.add_edge(
                            stock1, stock2, weight=abs(corr), correlation=corr
                        )

        self.logger.info(
            f"股票关系网络构建完成: {self.stock_network.number_of_nodes()}个节点, {self.stock_network.number_of_edges()}条边"
        )

    def analyze_market_sentiment_for_all_stocks(self, all_data, current_date):
        """
        为所有股票分析市场情感

        Args:
            all_data: 所有股票的历史数据
            current_date: 当前日期
        """
        self.logger.info(f"分析 {current_date} 的市场情感...")

        for stock_symbol in self.stock_pool:
            if stock_symbol not in all_data:
                continue

            # 获取最近的数据
            stock_data = all_data[stock_symbol]
            current_data = stock_data[stock_data["Date"] <= current_date]

            if len(current_data) < 10:  # 数据不足
                continue

            recent_data = current_data.tail(30)  # 最近30天的数据
            print(recent_data)
            try:
                # 使用LLM分析情感
                sentiment_analysis = self.llm_analyzer.analyze_market_sentiment(
                    stock_symbol, recent_data
                )

                # 存储情感分析结果
                if stock_symbol not in self.sentiment_history:
                    self.sentiment_history[stock_symbol] = []

                self.sentiment_history[stock_symbol].append(
                    {
                        "date": current_date,
                        "sentiment_score": sentiment_analysis.get(
                            "sentiment_score", 0.5
                        ),
                        "confidence": sentiment_analysis.get("confidence", 0.5),
                        "reasoning": sentiment_analysis.get("reasoning", ""),
                        "risk_level": sentiment_analysis.get("risk_level", "中"),
                    }
                )

                # 只保留最近的分析结果（避免内存过度使用）
                if len(self.sentiment_history[stock_symbol]) > 100:
                    self.sentiment_history[stock_symbol] = self.sentiment_history[
                        stock_symbol
                    ][-50:]

            except Exception as e:
                self.logger.error(f"股票 {stock_symbol} 情感分析失败: {e}")

    def generate_llm_trading_signals(self, all_data, current_date):
        """
        基于LLM生成交易信号

        Args:
            all_data: 所有股票的历史数据
            current_date: 当前日期
        """
        self.logger.info(f"生成 {current_date} 的LLM交易信号...")

        for stock_symbol in self.stock_pool:
            if stock_symbol not in all_data:
                continue

            # 获取技术指标
            stock_data = all_data[stock_symbol]
            current_data = stock_data[stock_data["Date"] <= current_date]

            if len(current_data) < 20:
                continue

            # 创建特征
            enhanced_data = self._create_features(current_data.copy())
            latest_data = enhanced_data.iloc[-1]

            # 准备技术指标
            technical_indicators = {
                "rsi": latest_data.get("rsi", 50),
                "macd": latest_data.get("macd", 0),
                "sma_ratio": (
                    latest_data.get("Close", 0) / latest_data.get("sma_20", 1)
                    if latest_data.get("sma_20", 0) != 0
                    else 1
                ),
                "atr": latest_data.get("atr", 0),
                "volume_ratio": (
                    latest_data.get("Volume", 0)
                    / enhanced_data["Volume"].rolling(20).mean().iloc[-1]
                    if enhanced_data["Volume"].rolling(20).mean().iloc[-1] != 0
                    else 1
                ),
            }

            # 获取情感分析结果
            market_sentiment = {}
            if (
                stock_symbol in self.sentiment_history
                and self.sentiment_history[stock_symbol]
            ):
                latest_sentiment = self.sentiment_history[stock_symbol][-1]
                market_sentiment = {
                    "sentiment_score": latest_sentiment["sentiment_score"],
                    "confidence": latest_sentiment["confidence"],
                    "risk_level": latest_sentiment["risk_level"],
                }

            try:
                # 生成交易信号
                trading_signal = self.llm_analyzer.generate_trading_signals(
                    stock_symbol, technical_indicators, market_sentiment
                )

                self.llm_signals[stock_symbol] = {
                    "date": current_date,
                    "signal": trading_signal.get("signal", "持有"),
                    "confidence": trading_signal.get("confidence", 0.5),
                    "target_weight": trading_signal.get("target_weight", 0.0),
                    "reasoning": trading_signal.get("reasoning", ""),
                }

            except Exception as e:
                self.logger.error(f"股票 {stock_symbol} 交易信号生成失败: {e}")

    def enhanced_make_decision(self, current_data, date, next_trading_date=None):
        """
        增强版决策函数，结合LLM分析结果

        Args:
            current_data: 当前可用的股票数据
            date: 当前日期
            next_trading_date: 下一个交易日期（可选）

        Returns:
            交易决策列表
        """
        decisions = []

        # 首先进行传统的技术分析
        traditional_decisions = self.make_decision(current_data, date)

        # 结合LLM信号进行增强
        for decision in traditional_decisions:
            stock_symbol = decision["action"]["symbol"]

            # 获取LLM信号
            llm_signal = self.llm_signals.get(stock_symbol, {})
            sentiment_data = self.sentiment_history.get(stock_symbol, [])

            # 调整决策
            enhanced_decision = decision.copy()

            if llm_signal and "signal" in llm_signal:
                llm_action = llm_signal.get("signal", "持有")
                llm_confidence = llm_signal.get("confidence", 0.5)
                original_action = decision["action"]["type"]

                # 信号冲突处理
                if llm_action == "卖出" and original_action == "buy":
                    # LLM建议卖出但技术分析建议买入，降低买入量
                    if llm_confidence > 0.7:
                        enhanced_decision["action"]["shares"] = int(
                            decision["action"]["shares"] * 0.3
                        )
                        enhanced_decision[
                            "reason"
                        ] += f" [LLM警告: {llm_signal.get('reasoning', '')}]"
                elif llm_action == "买入" and original_action == "buy":
                    # 两者都建议买入，可能增加买入量
                    if llm_confidence > 0.7:
                        enhanced_decision["action"]["shares"] = int(
                            decision["action"]["shares"] * 1.3
                        )
                        enhanced_decision[
                            "reason"
                        ] += f" [LLM确认: {llm_signal.get('reasoning', '')}]"
                elif llm_action == "买入" and original_action == "sell":
                    # LLM建议买入但技术分析建议卖出，减少卖出量
                    if llm_confidence > 0.7:
                        enhanced_decision["action"]["shares"] = int(
                            decision["action"]["shares"] * 0.5
                        )
                        enhanced_decision[
                            "reason"
                        ] += f" [LLM反对: {llm_signal.get('reasoning', '')}]"

            # 情感调整
            if sentiment_data:
                latest_sentiment = sentiment_data[-1]
                sentiment_score = latest_sentiment["sentiment_score"]
                risk_level = latest_sentiment["risk_level"]

                # 根据情感调整交易量
                if risk_level == "高" and enhanced_decision["action"]["type"] == "buy":
                    enhanced_decision["action"]["shares"] = int(
                        enhanced_decision["action"]["shares"] * 0.7
                    )
                    enhanced_decision["reason"] += f" [高风险调整]"
                elif (
                    risk_level == "低" and enhanced_decision["action"]["type"] == "buy"
                ):
                    enhanced_decision["action"]["shares"] = int(
                        enhanced_decision["action"]["shares"] * 1.1
                    )
                    enhanced_decision["reason"] += f" [低风险增持]"
                decisions.append(enhanced_decision)

        return decisions

    def make_decision(self, current_data, date):
        """基于技术分析的决策函数（参考原始策略逻辑）"""
        decisions = []

        # 计算信号强度和决策
        for stock_symbol, data_list in current_data.items():
            if not data_list or stock_symbol not in self.stock_pool:
                continue

            # 转换数据格式
            df = pd.DataFrame(
                data_list,
                columns=["Date", "Open", "High", "Low", "Close", "Volume", "Name"],
            )
            df["Date"] = pd.to_datetime(df["Date"])

            if len(df) < self.window_size:
                continue

            # 创建特征
            enhanced_df = self._create_features(df)
            latest_data = enhanced_df.iloc[-1]

            # 计算综合信号强度
            signal = self._calculate_signal_strength(enhanced_df, latest_data)
            price = latest_data.get("Close", 0)

            # 根据信号强度生成决策
            if signal > 2:  # 强买信号
                decisions.append(
                    {
                        "action": {
                            "type": "buy",
                            "symbol": stock_symbol,
                            "shares": min(
                                2000, int(signal * 400)
                            ),  # 根据信号强度调整数量
                            "price": price,
                        },
                        "reason": f"强买入信号 (信号强度={signal:.2f})",
                        "confidence": min(0.9, 0.5 + signal * 0.1),
                    }
                )
            elif signal > 0.5:  # 弱买信号
                decisions.append(
                    {
                        "action": {
                            "type": "buy",
                            "symbol": stock_symbol,
                            "shares": min(1000, int(signal * 500)),
                            "price": price,
                        },
                        "reason": f"买入信号 (信号强度={signal:.2f})",
                        "confidence": 0.5 + signal * 0.1,
                    }
                )
            elif signal < -2:  # 强卖信号
                decisions.append(
                    {
                        "action": {
                            "type": "sell",
                            "symbol": stock_symbol,
                            "shares": min(2000, int(abs(signal) * 400)),
                            "price": price,
                        },
                        "reason": f"强卖出信号 (信号强度={signal:.2f})",
                        "confidence": min(0.9, 0.5 + abs(signal) * 0.1),
                    }
                )
            elif signal < -0.5:  # 弱卖信号
                decisions.append(
                    {
                        "action": {
                            "type": "sell",
                            "symbol": stock_symbol,
                            "shares": min(1000, int(abs(signal) * 500)),
                            "price": price,
                        },
                        "reason": f"卖出信号 (信号强度={signal:.2f})",
                        "confidence": 0.5 + abs(signal) * 0.1,
                    }
                )

        return decisions

    def _calculate_signal_strength(self, df, latest_data):
        """计算技术分析信号强度（参考原始策略）"""
        signal = 0

        # RSI信号
        rsi = latest_data.get("rsi", 50)
        if rsi < 20:  # 严重超卖
            signal += 3
        elif rsi < 30:  # 超卖
            signal += 2
        elif rsi < 40:  # 弱超卖
            signal += 1
        elif rsi > 80:  # 严重超买
            signal -= 3
        elif rsi > 70:  # 超买
            signal -= 2
        elif rsi > 60:  # 弱超买
            signal -= 1
        # MACD信号
        macd = latest_data.get("macd", 0)
        if pd.isna(macd):
            macd = 0
        macd_prev = (
            df["macd"].iloc[-2]
            if len(df) > 1 and not pd.isna(df["macd"].iloc[-2])
            else 0
        )
        if macd > 0 and macd > macd_prev:  # MACD向上穿越零轴
            signal += 2
        elif macd > 0:  # MACD在零轴之上
            signal += 1
        elif macd < 0 and macd < macd_prev:  # MACD向下穿越零轴
            signal -= 2
        elif macd < 0:  # MACD在零轴之下
            signal -= 1
        # 移动平均信号
        price = latest_data.get("Close", 0)
        sma_10 = latest_data.get("sma_10", price)
        sma_20 = latest_data.get("sma_20", price)
        sma_50 = latest_data.get("sma_50", price)

        # 处理NaN值
        if pd.isna(sma_10):
            sma_10 = price
        if pd.isna(sma_20):
            sma_20 = price
        if pd.isna(sma_50):
            sma_50 = price

        if sma_10 > sma_20 > sma_50:  # 多头排列
            signal += 2
        elif sma_10 > sma_20:  # 短期上升
            signal += 1
        elif sma_10 < sma_20 < sma_50:  # 空头排列
            signal -= 2
        elif sma_10 < sma_20:  # 短期下降
            signal -= 1
        # 布林带信号
        bollinger_high = latest_data.get("bollinger_high", price * 1.02)
        bollinger_low = latest_data.get("bollinger_low", price * 0.98)

        # 处理NaN值
        if pd.isna(bollinger_high):
            bollinger_high = price * 1.02
        if pd.isna(bollinger_low):
            bollinger_low = price * 0.98
        if price < bollinger_low:  # 价格突破下轨
            signal += 1.5
        elif price > bollinger_high:  # 价格突破上轨
            signal -= 1.5

        # 成交量信号
        volume = latest_data.get("Volume", 0)
        avg_volume = (
            df["Volume"].rolling(20).mean().iloc[-1] if len(df) >= 20 else volume
        )
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        if volume_ratio > 2:  # 成交量放大
            signal = signal * 1.2  # 增强信号
        elif volume_ratio < 0.5:  # 成交量萎缩
            signal = signal * 0.8  # 减弱信号

        return signal

    def get_analysis_summary(self):
        """
        获取分析总结

        Returns:
            分析总结字典
        """
        summary = {
            "sentiment_analysis_count": sum(
                len(history) for history in self.sentiment_history.values()
            ),
            "trading_signals_count": len(self.llm_signals),
            "network_nodes": self.stock_network.number_of_nodes(),
            "network_edges": self.stock_network.number_of_edges(),
            "recent_sentiments": {},
        }

        # 获取最近的情感分析
        for stock, history in self.sentiment_history.items():
            if history:
                latest = history[-1]
                summary["recent_sentiments"][stock] = {
                    "sentiment_score": latest["sentiment_score"],
                    "risk_level": latest["risk_level"],
                }

        return summary

    def save_analysis_results(self, filepath):
        """
        保存分析结果到文件

        Args:
            filepath: 保存路径
        """
        results = {
            "sentiment_history": self.sentiment_history,
            "llm_signals": self.llm_signals,
            "network_info": {
                "nodes": list(self.stock_network.nodes()),
                "edges": [(u, v, d) for u, v, d in self.stock_network.edges(data=True)],
            },
            "analysis_summary": self.get_analysis_summary(),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)

        self.logger.info(f"分析结果已保存到: {filepath}")

    def analyze_market_sentiment_for_all_stocks_parallel(self, all_data, current_date):
        """
        并行分析所有股票的市场情感

        Args:
            all_data: 所有股票的历史数据
            current_date: 当前日期
        """
        self.logger.info(f"并行分析 {current_date} 的所有股票市场情感...")

        # 准备分析任务
        analysis_tasks = []
        for stock_symbol in self.stock_pool:
            if stock_symbol not in all_data:
                continue

            stock_data = all_data[stock_symbol]
            current_data = stock_data[stock_data["Date"] <= current_date]

            if len(current_data) < 10:  # 需要足够的历史数据
                continue

            # 获取最近的数据用于分析
            recent_data = current_data.tail(30)

            analysis_tasks.append(
                {
                    "task_type": "sentiment",
                    "stock_symbol": stock_symbol,
                    "recent_data": recent_data,
                    "news_headlines": None,  # 可以添加新闻数据
                }
            )

        # 执行并行分析
        results = self.llm_analyzer.batch_analyze_stocks_parallel(analysis_tasks)

        # 保存结果
        for result in results:
            if result["success"] and result["result"]:
                stock_symbol = result["stock_symbol"]
                sentiment_data = result["result"]

                # 添加日期信息
                sentiment_data["date"] = current_date

                # 保存到历史记录
                if stock_symbol not in self.sentiment_history:
                    self.sentiment_history[stock_symbol] = []
                self.sentiment_history[stock_symbol].append(sentiment_data)

                self.logger.info(
                    f"{stock_symbol} 情感分析完成: 评分={sentiment_data.get('sentiment_score', 0.5):.3f}"
                )

    def generate_llm_trading_signals_parallel(self, all_data, current_date):
        """
        并行生成所有股票的LLM交易信号

        Args:
            all_data: 所有股票的历史数据
            current_date: 当前日期
        """
        self.logger.info(f"并行生成 {current_date} 的LLM交易信号...")

        # 准备分析任务
        analysis_tasks = []

        for stock_symbol in self.stock_pool:
            if stock_symbol not in all_data:
                continue

            # 获取技术指标
            stock_data = all_data[stock_symbol]
            current_data = stock_data[stock_data["Date"] <= current_date]

            if len(current_data) < 20:
                continue

            # 创建特征
            enhanced_data = self._create_features(current_data.copy())
            latest_data = enhanced_data.iloc[-1]

            # 准备技术指标
            technical_indicators = {
                "rsi": latest_data.get("rsi", 50),
                "macd": latest_data.get("macd", 0),
                "sma_ratio": (
                    latest_data.get("Close", 0) / latest_data.get("sma_20", 1)
                    if latest_data.get("sma_20", 0) != 0
                    else 1
                ),
                "atr": latest_data.get("atr", 0),
                "volume_ratio": (
                    latest_data.get("Volume", 0)
                    / enhanced_data["Volume"].rolling(20).mean().iloc[-1]
                    if enhanced_data["Volume"].rolling(20).mean().iloc[-1] != 0
                    else 1
                ),
            }

            # 获取情感分析结果
            market_sentiment = {}
            if (
                stock_symbol in self.sentiment_history
                and self.sentiment_history[stock_symbol]
            ):
                market_sentiment = self.sentiment_history[stock_symbol][-1]
            else:
                market_sentiment = {
                    "sentiment_score": 0.5,
                    "confidence": 0.5,
                    "risk_level": "中",
                }

            analysis_tasks.append(
                {
                    "task_type": "trading_signal",
                    "stock_symbol": stock_symbol,
                    "technical_indicators": technical_indicators,
                    "market_sentiment": market_sentiment,
                }
            )

        # 执行并行分析
        results = self.llm_analyzer.batch_analyze_stocks_parallel(analysis_tasks)

        # 保存结果
        for result in results:
            if result["success"] and result["result"]:
                stock_symbol = result["stock_symbol"]
                signal_data = result["result"]

                # 添加日期信息
                signal_data["date"] = current_date

                # 保存交易信号
                self.llm_signals[stock_symbol] = signal_data

                self.logger.info(
                    f"{stock_symbol} 交易信号: {signal_data.get('signal', '持有')} (置信度: {signal_data.get('confidence', 0.5):.3f})"
                )
