import requests
import json
import time
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import logging
from typing import List, Dict, Optional

class MarketDataCollector:
    """
    市场数据收集器，用于获取股票相关的新闻和市场信息
    """
    
    def __init__(self):
        """初始化市场数据收集器"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_stock_info(self, symbol: str) -> Dict:
        """
        获取股票基本信息
        
        Args:
            symbol: 股票代码
            
        Returns:
            股票信息字典
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'company_name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 1.0),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
                'description': info.get('longBusinessSummary', '')
            }
        except Exception as e:
            self.logger.error(f"获取股票信息失败 {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e)
            }
    
    def get_market_news_headlines(self, symbol: str, max_articles: int = 10) -> List[str]:
        """
        获取股票相关的新闻标题（模拟数据）
        
        Args:
            symbol: 股票代码
            max_articles: 最大文章数
            
        Returns:
            新闻标题列表
        """
        # 由于真实的新闻API需要额外的key，这里提供一些模拟的新闻标题
        # 在实际应用中，可以集成真实的新闻API如NewsAPI, Alpha Vantage等
        
        simulated_news = {
            'AAPL': [
                "Apple Reports Strong Quarterly Earnings Beat Expectations",
                "New iPhone Launch Expected to Drive Q4 Revenue Growth", 
                "Apple Services Revenue Reaches Record High",
                "Analysts Upgrade Apple Stock Price Target",
                "Apple CEO Announces New Sustainability Initiative"
            ],
            'MSFT': [
                "Microsoft Cloud Revenue Surges 20% Year-over-Year",
                "Azure Growth Continues to Outpace Competition",
                "Microsoft AI Integration Boosts Productivity Suite",
                "Strong Enterprise Demand Drives Revenue Growth",
                "Microsoft Announces Strategic Partnership Expansion"
            ],
            'GOOGL': [
                "Google Search Revenue Shows Resilient Growth",
                "YouTube Advertising Revenue Exceeds Expectations",
                "Google Cloud Gains Market Share in Enterprise",
                "Regulatory Concerns Weigh on Stock Performance",
                "AI Innovations Drive Future Growth Prospects"
            ],
            'AMZN': [
                "Amazon Web Services Maintains Market Leadership",
                "E-commerce Growth Slows as Markets Normalize",
                "Amazon Logistics Network Expansion Continues",
                "Prime Membership Growth Drives Recurring Revenue",
                "Sustainability Initiatives Show Progress"
            ],
            'TSLA': [
                "Tesla Delivers Record Number of Vehicles",
                "Supercharger Network Expansion Accelerates", 
                "Battery Technology Breakthrough Announced",
                "European Factory Ramps Up Production",
                "Autonomous Driving Features Show Improvement"
            ]
        }
        
        # 为其他股票生成通用新闻
        default_news = [
            f"{symbol} Shows Strong Technical Indicators",
            f"Institutional Investors Increase {symbol} Holdings",
            f"{symbol} Sector Analysis Shows Positive Outlook",
            f"Market Volatility Creates Opportunities for {symbol}",
            f"{symbol} Dividend Policy Under Review"
        ]
        
        news_list = simulated_news.get(symbol, default_news)
        return news_list[:max_articles]
    
    def get_sector_performance(self, sector: str) -> Dict:
        """
        获取行业表现数据
        
        Args:
            sector: 行业名称
            
        Returns:
            行业表现字典
        """
        # 行业ETF映射
        sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV', 
            'Financial': 'XLF',
            'Consumer Cyclical': 'XLY',
            'Communication': 'XLC',
            'Industrial': 'XLI',
            'Consumer Defensive': 'XLP',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE',
            'Materials': 'XLB',
            'Energy': 'XLE'
        }
        
        etf_symbol = sector_etfs.get(sector, 'SPY')  # 默认使用SPY
        
        try:
            ticker = yf.Ticker(etf_symbol)
            hist = ticker.history(period="1mo")
            
            if len(hist) > 0:
                recent_performance = (hist['Close'][-1] - hist['Close'][0]) / hist['Close'][0] * 100
                volatility = hist['Close'].pct_change().std() * 100
                
                return {
                    'sector': sector,
                    'etf_symbol': etf_symbol,
                    'monthly_performance': recent_performance,
                    'volatility': volatility,
                    'last_price': hist['Close'][-1]
                }
        except Exception as e:
            self.logger.error(f"获取行业表现失败 {sector}: {e}")
        
        return {
            'sector': sector,
            'error': '无法获取行业数据'
        }
    
    def get_economic_indicators(self) -> Dict:
        """
        获取经济指标（模拟数据）
        
        Returns:
            经济指标字典
        """
        # 在实际应用中，可以集成FRED API等获取真实经济数据
        return {
            'vix': 20.5,  # 波动率指数
            'ten_year_yield': 4.2,  # 十年期国债收益率
            'dollar_index': 103.5,  # 美元指数
            'oil_price': 75.8,  # 原油价格
            'gold_price': 1950.0,  # 黄金价格
            'unemployment_rate': 3.7,  # 失业率
            'inflation_rate': 3.2,  # 通胀率
            'gdp_growth': 2.1  # GDP增长率
        }
    
    def analyze_market_sentiment_from_data(self, symbol: str) -> Dict:
        """
        基于收集的数据分析市场情感
        
        Args:
            symbol: 股票代码
            
        Returns:
            情感分析结果
        """
        try:
            # 获取股票信息
            stock_info = self.get_stock_info(symbol)
            
            # 获取新闻标题
            news_headlines = self.get_market_news_headlines(symbol)
            
            # 获取行业表现
            sector = stock_info.get('sector', 'Technology')
            sector_performance = self.get_sector_performance(sector)
            
            # 获取经济指标
            economic_indicators = self.get_economic_indicators()
            
            # 简单的情感评分算法
            sentiment_score = 0.5  # 基础分数
            confidence = 0.5
            
            # 基于PE比率调整
            pe_ratio = stock_info.get('pe_ratio', 20)
            if pe_ratio and pe_ratio > 0:
                if pe_ratio < 15:
                    sentiment_score += 0.1  # 低PE可能被低估
                elif pe_ratio > 30:
                    sentiment_score -= 0.1  # 高PE可能被高估
            
            # 基于行业表现调整
            if 'monthly_performance' in sector_performance:
                sector_perf = sector_performance['monthly_performance']
                if sector_perf > 5:
                    sentiment_score += 0.15
                elif sector_perf < -5:
                    sentiment_score -= 0.15
            
            # 基于新闻情感（简单关键词分析）
            positive_keywords = ['strong', 'growth', 'beat', 'high', 'increase', 'surge', 'record']
            negative_keywords = ['weak', 'decline', 'miss', 'low', 'decrease', 'fall', 'concern']
            
            positive_count = 0
            negative_count = 0
            
            for headline in news_headlines:
                headline_lower = headline.lower()
                positive_count += sum(1 for keyword in positive_keywords if keyword in headline_lower)
                negative_count += sum(1 for keyword in negative_keywords if keyword in headline_lower)
            
            if positive_count > negative_count:
                sentiment_score += 0.1
                confidence += 0.1
            elif negative_count > positive_count:
                sentiment_score -= 0.1
                confidence += 0.1
            
            # 确保分数在合理范围内
            sentiment_score = max(0.0, min(1.0, sentiment_score))
            confidence = max(0.0, min(1.0, confidence))
            
            return {
                'sentiment_score': sentiment_score,
                'confidence': confidence,
                'stock_info': stock_info,
                'news_headlines': news_headlines,
                'sector_performance': sector_performance,
                'economic_indicators': economic_indicators,
                'analysis_summary': f"基于PE比率、行业表现和新闻情感的综合分析"
            }
            
        except Exception as e:
            self.logger.error(f"市场情感分析失败 {symbol}: {e}")
            return {
                'sentiment_score': 0.5,
                'confidence': 0.3,
                'error': str(e)
            }
    
    def collect_market_data_for_portfolio(self, symbols: List[str]) -> Dict:
        """
        为投资组合收集市场数据
        
        Args:
            symbols: 股票代码列表
            
        Returns:
            投资组合市场数据
        """
        portfolio_data = {}
        
        for symbol in symbols:
            self.logger.info(f"收集 {symbol} 的市场数据...")
            portfolio_data[symbol] = self.analyze_market_sentiment_from_data(symbol)
            time.sleep(0.1)  # 避免请求过于频繁
        
        # 计算投资组合整体情感
        sentiment_scores = [data.get('sentiment_score', 0.5) for data in portfolio_data.values() if 'error' not in data]
        
        if sentiment_scores:
            portfolio_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        else:
            portfolio_sentiment = 0.5
        
        return {
            'individual_stocks': portfolio_data,
            'portfolio_sentiment': portfolio_sentiment,
            'collection_timestamp': datetime.now().isoformat()
        }

# 测试功能
if __name__ == "__main__":
    collector = MarketDataCollector()
    
    # 测试单个股票
    print("测试股票信息获取:")
    aapl_info = collector.get_stock_info('AAPL')
    print(json.dumps(aapl_info, indent=2, default=str))
    
    print("\n测试新闻标题获取:")
    news = collector.get_market_news_headlines('AAPL')
    for i, headline in enumerate(news, 1):
        print(f"{i}. {headline}")
    
    print("\n测试市场情感分析:")
    sentiment = collector.analyze_market_sentiment_from_data('AAPL')
    print(json.dumps(sentiment, indent=2, default=str))
