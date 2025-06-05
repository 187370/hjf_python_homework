"""
期末大作业快速启动脚本
演示LLM增强的股票交易策略系统
"""

import os
import sys
import time
import json
from datetime import datetime

def print_banner():
    """打印项目横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    基于大语言模型的股票交易策略增强系统                        ║
║                               期末大作业                                     ║
║                          北京大学 Python程序设计                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """检查依赖包"""
    print("🔍 检查依赖包...")
    
    required_packages = [
        ('pandas', 'pandas'), ('numpy', 'numpy'), ('sklearn', 'scikit-learn'), ('torch', 'torch'), 
        ('openai', 'openai'), ('yfinance', 'yfinance'), ('networkx', 'networkx'), ('matplotlib', 'matplotlib'), 
        ('seaborn', 'seaborn'), ('ta', 'ta'), ('tqdm', 'tqdm'), ('bs4', 'beautifulsoup4')
    ]
    
    missing_packages = []
    
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print(f"  ✅ {package_name}")
        except ImportError:
            missing_packages.append(package_name)
            print(f"  ❌ {package_name} (未安装)")
    
    if missing_packages:
        print(f"\n⚠️  请安装缺失的包: pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ 所有依赖包检查完成")
    return True

def check_data_files():
    """检查数据文件"""
    print("\n📁 检查数据文件...")
    
    data_dir = "time-series-data"
    if not os.path.exists(data_dir):
        print(f"  ❌ 数据目录不存在: {data_dir}")
        return False
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if len(csv_files) == 0:
        print(f"  ❌ 数据目录中没有CSV文件")
        return False
    
    print(f"  ✅ 找到 {len(csv_files)} 个股票数据文件")
    return True

def test_llm_connection():
    """测试LLM连接"""
    print("\n🤖 测试LLM连接...")
    
    try:
        from llm_analyzer import LLMAnalyzer
        
        # 使用您提供的API密钥
        api_key = "sk-XnbHbzBOmPYGHgL_xCMCYcABqSNTuqAhKtmOXJEONih7BqRZSw-JuRM1RbI"
        llm = LLMAnalyzer(api_key)
        
        # 简单测试
        test_data = {
            'Close': [100, 101, 102, 103, 104],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        }
        
        import pandas as pd
        test_df = pd.DataFrame(test_data)
        
        # 进行一次简单的情感分析测试
        result = llm.analyze_market_sentiment("AAPL", test_df)
        
        if 'sentiment_score' in result:
            print(f"  ✅ LLM连接成功")
            print(f"     情感评分: {result['sentiment_score']:.3f}")
            print(f"     置信度: {result.get('confidence', 'N/A')}")
            return True
        else:
            print(f"  ❌ LLM返回异常结果")
            return False
            
    except Exception as e:
        print(f"  ❌ LLM连接失败: {e}")
        return False

def run_quick_demo():
    """运行快速演示"""
    print("\n🚀 运行快速演示...")
    
    try:
        # 演示单个组件
        print("\n1. 演示市场数据收集器...")
        from market_data_collector import MarketDataCollector
        collector = MarketDataCollector()
        
        # 获取AAPL股票信息
        aapl_info = collector.get_stock_info('AAPL')
        print(f"   📊 AAPL信息: {aapl_info.get('company_name', 'Apple Inc.')}")
        print(f"      行业: {aapl_info.get('sector', 'Technology')}")
        print(f"      市值: ${aapl_info.get('market_cap', 0):,}")
        
        # 获取新闻标题
        news = collector.get_market_news_headlines('AAPL', 3)
        print(f"   📰 相关新闻:")
        for i, headline in enumerate(news, 1):
            print(f"      {i}. {headline}")
        
        print("\n2. 演示LLM分析器...")
        from llm_analyzer import LLMAnalyzer
        import pandas as pd
        
        # 创建测试数据
        test_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=20),
            'Close': [100 + i + (i%3)*2 for i in range(20)],
            'Volume': [1000 + i*50 for i in range(20)]
        })
        
        api_key = "sk-XnbHbzBOmPYGHgL_xCMCYcABqSNTuqAhKtmOXJEONih7BqRZSw-JuRM1RbI"
        llm = LLMAnalyzer(api_key)
        
        # 情感分析
        sentiment = llm.analyze_market_sentiment("DEMO", test_data.tail(10))
        print(f"   🎭 情感分析结果:")
        print(f"      情感评分: {sentiment.get('sentiment_score', 0.5):.3f}")
        print(f"      置信度: {sentiment.get('confidence', 0.5):.3f}")
        print(f"      分析理由: {sentiment.get('reasoning', '无')[:100]}...")
        
        print("\n3. 演示股票关系网络...")
        import networkx as nx
        
        # 创建简单的股票关系网络
        G = nx.Graph()
        stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        G.add_nodes_from(stocks)
        
        # 添加一些关系边
        G.add_edge('AAPL', 'MSFT', weight=0.7)
        G.add_edge('GOOGL', 'AMZN', weight=0.6)
        G.add_edge('AAPL', 'GOOGL', weight=0.5)
        
        print(f"   🕸️  关系网络:")
        print(f"      节点数: {G.number_of_nodes()}")
        print(f"      边数: {G.number_of_edges()}")
        print(f"      密度: {nx.density(G):.3f}")
        
        print("\n✅ 快速演示完成!")
        return True
        
    except Exception as e:
        print(f"❌ 演示过程中出错: {e}")
        return False

def run_full_comparison():
    """运行完整的策略比较"""
    print("\n🏃 运行完整策略比较...")
    
    try:
        from enhanced_eval_main import StrategyComparison
        
        # 配置参数
        stock_pool = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'][:3]  # 使用前3只股票进行快速测试
        api_key = "sk-XnbHbzBOmPYGHgL_xCMCYcABqSNTuqAhKtmOXJEONih7BqRZSw-JuRM1RbI"
        
        # 创建比较器
        comparator = StrategyComparison(initial_cash=100000, max_shares_per_trade=1000)  # 减少初始资金以加快测试
        
        # 运行比较（使用较少的交易天数）
        results = comparator.run_comparison(
            stock_pool=stock_pool,
            llm_api_key=api_key,
            data_dir="time-series-data",
            start_date="2017-01-01",
            end_date="2017-06-01",
            num_trading_days=5  # 只使用5个交易日进行快速测试
        )
        
        # 保存结果
        with open("quick_demo_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print("✅ 完整比较完成，结果已保存到 quick_demo_results.json")
        return True
        
    except Exception as e:
        print(f"❌ 完整比较失败: {e}")
        return False

def main():
    """主函数"""
    print_banner()
    
    print(f"🕒 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查系统环境
    if not check_dependencies():
        print("❌ 环境检查失败，请安装缺失的依赖包")
        return
    
    if not check_data_files():
        print("❌ 数据文件检查失败，请确保time-series-data目录中有股票数据")
        return
    
    # 测试LLM连接（可选）
    print("\n" + "="*80)
    choice = input("是否测试LLM连接？这将消耗API调用次数 (y/n): ").lower().strip()
    
    if choice == 'y':
        if not test_llm_connection():
            print("⚠️  LLM连接测试失败，但可以继续演示其他功能")
        
        # 运行演示
        print("\n" + "="*80)
        choice = input("是否运行快速演示？这将消耗少量API调用 (y/n): ").lower().strip()
        
        if choice == 'y':
            run_quick_demo()
        
        # 运行完整比较
        print("\n" + "="*80)
        choice = input("是否运行完整策略比较？这将消耗较多API调用和时间 (y/n): ").lower().strip()
        
        if choice == 'y':
            run_full_comparison()
    
    print("\n" + "="*80)
    print("🎉 演示完成!")
    print("\n📖 详细使用说明请参考 README.md")
    print("📁 主要文件说明:")
    print("   - enhanced_trading_strategy.py: 增强版交易策略")
    print("   - llm_analyzer.py: LLM分析器")
    print("   - market_data_collector.py: 市场数据收集器")
    print("   - enhanced_eval_main.py: 完整的策略比较程序")
    
    print(f"\n🕒 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
