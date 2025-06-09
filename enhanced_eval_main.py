import pandas as pd
import numpy as np
import os
import glob
from collections import defaultdict
from datetime import datetime, timedelta
import random
from enhanced_trading_strategy import EnhancedTradingStrategy
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings

warnings.filterwarnings("ignore")
random.seed(42)

class EnhancedStrategyEvaluator:
    """
    增强策略专用评估器
    """
    def __init__(self, initial_cash=1000000, max_shares_per_trade=10000, enable_parallel=True):
        """
        初始化增强策略评估器
        Args:
            initial_cash: 初始资金
            max_shares_per_trade: 单次交易最大股数
            enable_parallel: 是否启用并行处理
        """
        self.initial_cash = initial_cash
        self.max_shares_per_trade = max_shares_per_trade
        self.enable_parallel = enable_parallel
        self.evaluation_results = {}
        self.daily_portfolio_details = []  # 每日详细投资组合状态
        
    def load_stock_data(self, data_dir):
        """加载股票数据"""
        all_data = {}
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        for file_path in csv_files:
            stock_code = os.path.basename(file_path).split("_")[0]
            df = pd.read_csv(file_path, parse_dates=["Date"])
            all_data[stock_code] = df
        return all_data

    def prepare_test_data(self, all_data, start_date, end_date):
        """准备测试数据"""
        test_data = {}
        for stock_code, df in all_data.items():
            mask = (df["Date"] >= start_date) & (df["Date"] < end_date)
            filtered_data = df.loc[mask]
            if not filtered_data.empty:
                data_list = []
                for _, row in filtered_data.iterrows():
                    data_list.append([
                        row["Date"], row["Open"], row["High"], 
                        row["Low"], row["Close"], row["Volume"], stock_code
                    ])
                test_data[stock_code] = data_list
        return test_data    
    def generate_trading_dates(self, start_date, num_days, min_gap=1, max_gap=100):
        """
        随机生成交易日序列
        :param start_date: 起始日期
        :param num_days: 交易日数量
        :param min_gap: 最小间隔天数
        :param max_gap: 最大间隔天数
        :return: 交易日列表
        """
        trading_dates = []
        current_date = start_date

        for _ in range(num_days):
            trading_dates.append(current_date)
            # 随机生成下一个交易日的间隔天数
            gap = random.randint(min_gap, max_gap)
            current_date = current_date + timedelta(days=gap)

        return trading_dates

    def run_enhanced_backtest(self, strategy, all_data, trading_dates):
        """
        运行增强策略回测
        
        Args:
            strategy: 增强交易策略实例
            all_data: 所有股票数据
            trading_dates: 交易日期列表
            
        Returns:
            回测结果
        """
        print("开始运行增强策略回测...")
        
        # 初始化资金和持仓
        cash = self.initial_cash
        positions = defaultdict(int)
        portfolio_values = []
        trade_log = []
        llm_analysis_log = []
        # 构建关系网络
        print("构建股票关系网络...")
        strategy.build_stock_relationship_network(all_data)
        for i, date in enumerate(tqdm(trading_dates, desc="增强策略回测进度")):
            current_data = self.prepare_test_data(all_data, trading_dates[0], date)

            # 计算当前投资组合价值供LLM参考
            portfolio_value_before = cash
            for sym, qty in positions.items():
                if sym in current_data and current_data[sym]:
                    portfolio_value_before += qty * current_data[sym][-1][4]

            portfolio_state = {"cash": cash, "positions": dict(positions)}
              
            # 每1个交易日进行一次LLM分析，这里最后换成了每一次执行一次（因为解决API并发限制）
            if i % 1 == 0:
                try:
                    print(f"分析 {date.strftime('%Y-%m-%d')} 的市场情感...")
                    
                    if self.enable_parallel:
                        # 使用并行处理
                        strategy.analyze_market_sentiment_for_all_stocks_parallel(all_data, date)
                        strategy.generate_llm_trading_signals_parallel(
                            all_data,
                            date,
                            portfolio_state=portfolio_state,
                            total_portfolio_value=portfolio_value_before,
                        )
                    else:
                        # 使用原有的串行处理
                        strategy.analyze_market_sentiment_for_all_stocks(all_data, date)
                        strategy.generate_llm_trading_signals(
                            all_data,
                            date,
                            portfolio_state=portfolio_state,
                            total_portfolio_value=portfolio_value_before,
                        )
                    
                    analysis_summary = strategy.get_analysis_summary()
                    llm_analysis_log.append({
                        'date': date,
                        'analysis': analysis_summary
                    })
                    
                except Exception as e:
                    print(f"LLM分析失败: {e}")
            
            # 计算当前投资组合价值供决策参考
            portfolio_value_before = cash
            for sym, qty in positions.items():
                if sym in current_data and current_data[sym]:
                    portfolio_value_before += qty * current_data[sym][-1][4]

            portfolio_state = {"cash": cash, "positions": dict(positions)}

            # 使用增强决策
            decisions = strategy.enhanced_make_decision(
                current_data,
                date,
                portfolio=portfolio_state,
                total_portfolio_value=portfolio_value_before,
            )
            
            # 执行交易决策
            day_trades = []  # 当日交易记录
            for decision in decisions:
                action = decision['action']
                symbol = action['symbol']
                shares = min(action['shares'], self.max_shares_per_trade)
                price = action['price']
                
                if action['type'] == 'buy' and cash >= shares * price:
                    # 买入
                    cost = shares * price
                    cash -= cost
                    positions[symbol] += shares
                    
                    trade_record = {
                        'date': date,
                        'action': 'buy',
                        'symbol': symbol,
                        'shares': shares,
                        'price': price,
                        'cost': cost,
                        'reason': decision.get('reason', ''),
                        'confidence': decision.get('confidence', 0.5),
                        'cash_after': cash
                    }
                    trade_log.append(trade_record)
                    day_trades.append(trade_record)
                elif action['type'] == 'sell' and positions[symbol] >= shares:
                    # 卖出
                    revenue = shares * price
                    cash += revenue
                    positions[symbol] -= shares
                    
                    trade_record = {
                        'date': date,
                        'action': 'sell',
                        'symbol': symbol,
                        'shares': shares,
                        'price': price,
                        'revenue': revenue,
                        'reason': decision.get('reason', ''),
                        'confidence': decision.get('confidence', 0.5),
                        'cash_after': cash
                    }
                    trade_log.append(trade_record)
                    day_trades.append(trade_record)
            
            # 计算当前投资组合价值
            portfolio_value = cash
            position_values = {}
            for symbol, shares in positions.items():
                if symbol in current_data and current_data[symbol]:
                    current_price = current_data[symbol][-1][4]  # 收盘价
                    position_value = shares * current_price
                    portfolio_value += position_value
                    position_values[symbol] = {
                        'shares': shares,
                        'price': current_price,
                        'value': position_value,
                        'weight': position_value / portfolio_value if portfolio_value > 0 else 0
                    }
            
            # 详细的每日投资组合状态
            daily_detail = {
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'cash_ratio': cash / portfolio_value if portfolio_value > 0 else 1,
                'positions': dict(positions),
                'position_values': position_values,
                'position_count': len([s for s, shares in positions.items() if shares > 0]),
                'day_trades': day_trades,
                'daily_return': 0,  # 将在后面计算
                'cumulative_return': (portfolio_value - self.initial_cash) / self.initial_cash * 100
            }
            
            # 计算日收益率
            if portfolio_values:
                prev_value = portfolio_values[-1]['portfolio_value']
                daily_detail['daily_return'] = (portfolio_value - prev_value) / prev_value * 100 if prev_value > 0 else 0
            
            portfolio_values.append(daily_detail)
            self.daily_portfolio_details.append(daily_detail)
            
            # 每日投资组合报告
            if i % 1 == 0 or i == len(trading_dates) - 1:
                self._print_daily_portfolio_report(date, daily_detail, day_trades)
        
        # 计算最终结果
        final_value = portfolio_values[-1]['portfolio_value'] if portfolio_values else self.initial_cash
        total_return = (final_value - self.initial_cash) / self.initial_cash * 100
        

        values = [p['portfolio_value'] for p in portfolio_values]
        returns = pd.Series(values).pct_change().dropna()
        
        # 风险指标
        volatility = returns.std() * np.sqrt(252) * 100  # 年化波动率
        sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
        max_drawdown = self.calculate_max_drawdown(values)
        
        # 交易统计
        buy_trades = [t for t in trade_log if t['action'] == 'buy']
        sell_trades = [t for t in trade_log if t['action'] == 'sell']
        
        result = {
            'strategy_name': 'LLM增强策略',
            'initial_cash': self.initial_cash,
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': ((final_value / self.initial_cash) ** (365.25 / len(trading_dates)) - 1) * 100,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'portfolio_history': portfolio_values,
            'trade_log': trade_log,
            'llm_analysis_log': llm_analysis_log,
            'total_trades': len(trade_log),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'win_rate': self.calculate_win_rate(trade_log),
            'llm_analysis': strategy.get_analysis_summary()
        }
        
        return result

    def calculate_max_drawdown(self, values):
        """计算最大回撤"""
        peak = values[0]
        max_dd = 0
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_dd:
                max_dd = drawdown
                
        return max_dd * 100

    def calculate_win_rate(self, trade_log):
        """计算胜率"""
        if not trade_log:
            return 0
        
        wins = 0
        total_rounds = 0
        
        positions = defaultdict(list)
        
        for trade in trade_log:
            if trade['action'] == 'buy':
                positions[trade['symbol']].append(trade)
            elif trade['action'] == 'sell':
                symbol = trade['symbol']
                if positions[symbol]:
                    buy_trade = positions[symbol].pop(0)
                    if trade['price'] > buy_trade['price']:
                        wins += 1
                    total_rounds += 1
        
        return (wins / total_rounds * 100) if total_rounds > 0 else 0

    def generate_detailed_report(self, results):
        """生成详细报告"""
        print("\n" + "="*80)
        print("LLM增强股票交易策略评估报告")
        print("="*80)
        
        # 基本性能指标
        print(f"\n 基本性能指标:")
        print(f"   初始资金: ${results['initial_cash']:,.2f}")
        print(f"   最终价值: ${results['final_value']:,.2f}")
        print(f"   总收益率: {results['total_return']:.2f}%")
        print(f"   年化收益率: {results['annualized_return']:.2f}%")
        print(f"   年化波动率: {results['volatility']:.2f}%")
        print(f"   夏普比率: {results['sharpe_ratio']:.3f}")
        print(f"   最大回撤: {results['max_drawdown']:.2f}%")
        
        # 交易统计
        print(f"\n 交易统计:")
        print(f"   总交易次数: {results['total_trades']}")
        print(f"   买入交易: {results['buy_trades']}")
        print(f"   卖出交易: {results['sell_trades']}")
        print(f"   胜率: {results['win_rate']:.1f}%")
        
        # LLM分析统计
        llm_info = results['llm_analysis']
        print(f"\n LLM分析统计:")
        print(f"   情感分析次数: {llm_info['sentiment_analysis_count']}")
        print(f"   交易信号生成次数: {llm_info['trading_signals_count']}")
        print(f"   股票关系网络: {llm_info['network_nodes']}个节点, {llm_info['network_edges']}条边")
        
        # 最近情感分析
        if llm_info['recent_sentiments']:
            print(f"\n 最近情感分析:")
            for stock, sentiment in llm_info['recent_sentiments'].items():
                print(f"   {stock}: 情感评分={sentiment['sentiment_score']:.3f}, 风险={sentiment['risk_level']}")
        
        # 投资组合演变
        portfolio_history = results['portfolio_history']
        if portfolio_history:
            print(f"\n 投资组合演变:")
            print(f"   最大持仓股票数: {max(p['position_count'] for p in portfolio_history)}")
            print(f"   平均持仓股票数: {np.mean([p['position_count'] for p in portfolio_history]):.1f}")
            print(f"   最终现金余额: ${portfolio_history[-1]['cash']:,.2f}")

    def plot_enhanced_analysis(self, results, save_path="enhanced_strategy_analysis.png"):
        """绘制增强策略分析图表"""
        from matplotlib import rcParams
        rcParams['axes.unicode_minus'] = False  
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('LLM-Enhanced Stock Trading Strategy Analysis', fontsize=16, fontweight='bold')
        
        portfolio_history = results['portfolio_history']
        dates = [p['date'] for p in portfolio_history]
        values = [p['portfolio_value'] for p in portfolio_history]
        
        #  投资组合价值变化
        ax1 = axes[0, 0]
        ax1.plot(dates, values, linewidth=2, color='blue')
        ax1.axhline(y=self.initial_cash, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        #  收益率分布
        ax2 = axes[0, 1]
        returns = pd.Series(values).pct_change().dropna() * 100
        ax2.hist(returns, bins=30, alpha=0.7, color='green')
        ax2.axvline(returns.mean(), color='red', linestyle='--', label=f'Avg. Return: {returns.mean():.2f}%')
        ax2.set_title('Daily Return Distribution')
        ax2.set_xlabel('Return (%)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        #  回撤分析
        ax3 = axes[0, 2]
        peak = np.maximum.accumulate(values)
        drawdown = (np.array(values) - peak) / peak * 100
        ax3.fill_between(dates, 0, drawdown, color='red', alpha=0.3)
        ax3.plot(dates, drawdown, color='red', linewidth=1)
        ax3.set_title('Drawdown Analysis')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Drawdown (%)')
        ax3.grid(True, alpha=0.3)
        
        #  交易活动
        ax4 = axes[1, 0]
        trade_log = results['trade_log']
        buy_dates = [t['date'] for t in trade_log if t['action'] == 'buy']
        sell_dates = [t['date'] for t in trade_log if t['action'] == 'sell']
        
        # 使用柱状图显示每月交易次数
        all_dates = pd.DatetimeIndex([t['date'] for t in trade_log])
        monthly_trades = all_dates.to_period('M').value_counts().sort_index()
        
        ax4.bar(range(len(monthly_trades)), monthly_trades.values, color='purple', alpha=0.7)
        ax4.set_title('Monthly Trade Count')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Number of Trades')
        
        #  持仓变化
        ax5 = axes[1, 1]
        position_counts = [p['position_count'] for p in portfolio_history]
        ax5.plot(dates, position_counts, marker='o', markersize=3, color='orange')
        ax5.set_title('Number of Held Stocks Over Time')
        ax5.set_xlabel('Date')
        ax5.set_ylabel('Stock Count')
        ax5.grid(True, alpha=0.3)
        
        #  LLM情感分析
        ax6 = axes[1, 2]
        llm_info = results['llm_analysis']
        recent_sentiments = llm_info.get('recent_sentiments', {})
        
        if recent_sentiments:
            stocks = list(recent_sentiments.keys())
            sentiment_scores = [recent_sentiments[stock]['sentiment_score'] for stock in stocks]
            
            bars = ax6.bar(stocks, sentiment_scores, color='lightblue')
            ax6.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Neutral Line')
            ax6.set_title('Recent Sentiment Scores')
            ax6.set_xlabel('Stock Symbol')
            ax6.set_ylabel('Sentiment Score')
            ax6.set_ylim(0, 1)
            ax6.legend()
            
            for bar, score in zip(bars, sentiment_scores):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"分析图表已保存到: {save_path}")

    def save_results(self, results, filepath="enhanced_strategy_results.json"):
        """保存评估结果"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"评估结果已保存到: {filepath}")

    def _print_daily_portfolio_report(self, date, daily_detail, day_trades):
        """
        打印每日投资组合详细报告
        
        Args:
            date: 交易日期
            daily_detail: 当日详细数据
            day_trades: 当日交易记录
        """
        print(f"\n {date.strftime('%Y-%m-%d')} 投资组合状态:")
        print(f"    总价值: ${daily_detail['portfolio_value']:,.2f}")
        print(f"    现金: ${daily_detail['cash']:,.2f} ({daily_detail['cash_ratio']:.1%})")
        print(f"    持仓数: {daily_detail['position_count']} 只股票")
        print(f"    日收益: {daily_detail['daily_return']:+.2f}%")
        print(f"    累计收益: {daily_detail['cumulative_return']:+.2f}%")
        
        # 显示前5大持仓
        if daily_detail['position_values']:
            top_positions = sorted(daily_detail['position_values'].items(), 
                                 key=lambda x: x[1]['value'], reverse=True)[:5]
            print(f"    前5大持仓:")
            for symbol, pos_data in top_positions:
                print(f"     {symbol}: {pos_data['shares']:,}股 ${pos_data['value']:,.0f} ({pos_data['weight']:.1%})")
        
        # 显示当日交易
        if day_trades:
            print(f"    当日交易 ({len(day_trades)}笔):")
            for trade in day_trades:
                action_emoji = "🛒" if trade['action'] == 'buy' else "🛍️"
                amount = trade.get('cost', trade.get('revenue', 0))
                print(f"     {action_emoji} {trade['symbol']}: {trade['shares']:,}股 ${amount:,.0f}")

    def generate_comprehensive_portfolio_report(self, results):
        """
        生成全面的投资组合报告
        
        Args:
            results: 回测结果
        """
        print("\n" + "="*100)
        print(" 详细投资组合分析报告")
        print("="*100)
        
        # 基本统计
        portfolio_history = results['portfolio_history']
        if not portfolio_history:
            print("  无投资组合历史数据")
            return
        
        # 收益率分析
        values = [p['portfolio_value'] for p in portfolio_history]
        daily_returns = [p['daily_return'] for p in portfolio_history if 'daily_return' in p]
        
        print(f"\n 收益率分析:")
        print(f"   最高投资组合价值: ${max(values):,.2f}")
        print(f"   最低投资组合价值: ${min(values):,.2f}")
        print(f"   平均日收益率: {np.mean(daily_returns):.3f}%" if daily_returns else "N/A")
        print(f"   收益率标准差: {np.std(daily_returns):.3f}%" if daily_returns else "N/A")
        print(f"   最大单日收益: {max(daily_returns):.2f}%" if daily_returns else "N/A")
        print(f"   最大单日亏损: {min(daily_returns):.2f}%" if daily_returns else "N/A")
        
        # 持仓分析
        print(f"\n 持仓分析:")
        position_counts = [p['position_count'] for p in portfolio_history]
        cash_ratios = [p['cash_ratio'] for p in portfolio_history if 'cash_ratio' in p]
        
        print(f"   平均持仓股票数: {np.mean(position_counts):.1f}")
        print(f"   最大持仓股票数: {max(position_counts)}")
        print(f"   平均现金比例: {np.mean(cash_ratios):.1%}" if cash_ratios else "N/A")
        
        # 交易频率分析
        trade_dates = set()
        total_trades = 0
        for detail in self.daily_portfolio_details:
            if detail.get('day_trades'):
                trade_dates.add(detail['date'])
                total_trades += len(detail['day_trades'])
        
        print(f"\n 交易活动分析:")
        print(f"   活跃交易日: {len(trade_dates)} 天")
        print(f"   总交易笔数: {total_trades}")
        print(f"   日均交易笔数: {total_trades / len(portfolio_history):.1f}")
        
        # 风险指标
        print(f"\n 风险指标:")
        if len(values) > 1:
            # 计算最大回撤的具体日期
            peak = values[0]
            max_dd = 0
            max_dd_start = portfolio_history[0]['date']
            max_dd_end = portfolio_history[0]['date']
            
            for i, value in enumerate(values):
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                if drawdown > max_dd:
                    max_dd = drawdown
                    max_dd_end = portfolio_history[i]['date']
                    # 找到开始日期
                    for j in range(i, -1, -1):
                        if values[j] == peak:
                            max_dd_start = portfolio_history[j]['date']
                            break
            
            print(f"   最大回撤: {max_dd:.2%}")
            print(f"   回撤期间: {max_dd_start.strftime('%Y-%m-%d')} 至 {max_dd_end.strftime('%Y-%m-%d')}")
        
        # 最终状态
        final_state = portfolio_history[-1]
        print(f"\n 最终投资组合状态:")
        print(f"   最终价值: ${final_state['portfolio_value']:,.2f}")
        print(f"   现金余额: ${final_state['cash']:,.2f}")
        print(f"   持仓股票: {final_state['position_count']} 只")
        
        if 'position_values' in final_state and final_state['position_values']:
            print(f"   最终持仓详情:")
            sorted_positions = sorted(final_state['position_values'].items(), 
                                   key=lambda x: x[1]['value'], reverse=True)
            for symbol, pos_data in sorted_positions:
                print(f"     {symbol}: {pos_data['shares']:,}股 @ ${pos_data['price']:.2f} = ${pos_data['value']:,.0f} ({pos_data['weight']:.1%})")

def main():
    print(" LLM增强股票交易策略评估系统")
    print("="*50)
    data_files = glob.glob("time-series-data/*.csv")
    STOCK_POOL = [os.path.basename(f).split("_")[0] for f in data_files]
    print(f"检测到 {len(STOCK_POOL)} 只股票: {STOCK_POOL}")
    
    LLM_API_KEY = ["sk-XnbHbzBOmPYGHgL_xCMCYcABqSNTuqAhKtmOXJEONih7BqRZSw-JuRM1RbI","sk-XnbHbzBOmPYGHgL_xCMCYcABqSNTuqAhKtmOXJEONih7BqRZSw-JuRM1RbI","sk-RMerNTL9uP4lmHh_1rwUHJvcaqKXXzb7IrAtjWTa5Ln_aFMFTNPkuzIi7vw","sk-RMerNTL9uP4lmHh_1rwUHJvcaqKXXzb7IrAtjWTa5Ln_aFMFTNPkuzIi7vw","sk-XnbHbzBOmPYGHgL_xCMCYcABqSNTuqAhKtmOXJEONih7BqRZSw-JuRM1RbI","sk-XnbHbzBOmPYGHgL_xCMCYcABqSNTuqAhKtmOXJEONih7BqRZSw-JuRM1RbI","sk-RMerNTL9uP4lmHh_1rwUHJvcaqKXXzb7IrAtjWTa5Ln_aFMFTNPkuzIi7vw","sk-RMerNTL9uP4lmHh_1rwUHJvcaqKXXzb7IrAtjWTa5Ln_aFMFTNPkuzIi7vw","sk-XnbHbzBOmPYGHgL_xCMCYcABqSNTuqAhKtmOXJEONih7BqRZSw-JuRM1RbI","sk-XnbHbzBOmPYGHgL_xCMCYcABqSNTuqAhKtmOXJEONih7BqRZSw-JuRM1RbI","sk-RMerNTL9uP4lmHh_1rwUHJvcaqKXXzb7IrAtjWTa5Ln_aFMFTNPkuzIi7vw","sk-RMerNTL9uP4lmHh_1rwUHJvcaqKXXzb7IrAtjWTa5Ln_aFMFTNPkuzIi7vw","sk-XnbHbzBOmPYGHgL_xCMCYcABqSNTuqAhKtmOXJEONih7BqRZSw-JuRM1RbI","sk-XnbHbzBOmPYGHgL_xCMCYcABqSNTuqAhKtmOXJEONih7BqRZSw-JuRM1RbI","sk-RMerNTL9uP4lmHh_1rwUHJvcaqKXXzb7IrAtjWTa5Ln_aFMFTNPkuzIi7vw","sk-RMerNTL9uP4lmHh_1rwUHJvcaqKXXzb7IrAtjWTa5Ln_aFMFTNPkuzIi7vw","sk-XnbHbzBOmPYGHgL_xCMCYcABqSNTuqAhKtmOXJEONih7BqRZSw-JuRM1RbI","sk-XnbHbzBOmPYGHgL_xCMCYcABqSNTuqAhKtmOXJEONih7BqRZSw-JuRM1RbI","sk-RMerNTL9uP4lmHh_1rwUHJvcaqKXXzb7IrAtjWTa5Ln_aFMFTNPkuzIi7vw","sk-RMerNTL9uP4lmHh_1rwUHJvcaqKXXzb7IrAtjWTa5Ln_aFMFTNPkuzIi7vw","sk-XnbHbzBOmPYGHgL_xCMCYcABqSNTuqAhKtmOXJEONih7BqRZSw-JuRM1RbI","sk-XnbHbzBOmPYGHgL_xCMCYcABqSNTuqAhKtmOXJEONih7BqRZSw-JuRM1RbI","sk-RMerNTL9uP4lmHh_1rwUHJvcaqKXXzb7IrAtjWTa5Ln_aFMFTNPkuzIi7vw","sk-RMerNTL9uP4lmHh_1rwUHJvcaqKXXzb7IrAtjWTa5Ln_aFMFTNPkuzIi7vw","sk-XnbHbzBOmPYGHgL_xCMCYcABqSNTuqAhKtmOXJEONih7BqRZSw-JuRM1RbI","sk-XnbHbzBOmPYGHgL_xCMCYcABqSNTuqAhKtmOXJEONih7BqRZSw-JuRM1RbI","sk-RMerNTL9uP4lmHh_1rwUHJvcaqKXXzb7IrAtjWTa5Ln_aFMFTNPkuzIi7vw","sk-RMerNTL9uP4lmHh_1rwUHJvcaqKXXzb7IrAtjWTa5Ln_aFMFTNPkuzIi7vw","sk-XnbHbzBOmPYGHgL_xCMCYcABqSNTuqAhKtmOXJEONih7BqRZSw-JuRM1RbI","sk-XnbHbzBOmPYGHgL_xCMCYcABqSNTuqAhKtmOXJEONih7BqRZSw-JuRM1RbI","sk-RMerNTL9uP4lmHh_1rwUHJvcaqKXXzb7IrAtjWTa5Ln_aFMFTNPkuzIi7vw","sk-RMerNTL9uP4lmHh_1rwUHJvcaqKXXzb7IrAtjWTa5Ln_aFMFTNPkuzIi7vw","sk-XnbHbzBOmPYGHgL_xCMCYcABqSNTuqAhKtmOXJEONih7BqRZSw-JuRM1RbI","sk-XnbHbzBOmPYGHgL_xCMCYcABqSNTuqAhKtmOXJEONih7BqRZSw-JuRM1RbI","sk-RMerNTL9uP4lmHh_1rwUHJvcaqKXXzb7IrAtjWTa5Ln_aFMFTNPkuzIi7vw","sk-RMerNTL9uP4lmHh_1rwUHJvcaqKXXzb7IrAtjWTa5Ln_aFMFTNPkuzIi7vw","sk-XnbHbzBOmPYGHgL_xCMCYcABqSNTuqAhKtmOXJEONih7BqRZSw-JuRM1RbI","sk-XnbHbzBOmPYGHgL_xCMCYcABqSNTuqAhKtmOXJEONih7BqRZSw-JuRM1RbI","sk-RMerNTL9uP4lmHh_1rwUHJvcaqKXXzb7IrAtjWTa5Ln_aFMFTNPkuzIi7vw","sk-RMerNTL9uP4lmHh_1rwUHJvcaqKXXzb7IrAtjWTa5Ln_aFMFTNPkuzIi7vw"] 
    
    # 创建评估器
    evaluator = EnhancedStrategyEvaluator(initial_cash=1000000, max_shares_per_trade=10000)
    print("加载股票数据...")
    all_data = evaluator.load_stock_data("time-series-data")
    print(f"成功加载 {len(all_data)} 只股票的数据")
    # 生成交易日期
    start_date = datetime(2011, 1, 1)  
    end_date = datetime(2018, 1, 1)
    trading_dates = evaluator.generate_trading_dates(start_date, 60, min_gap=10, max_gap=80)
    trading_dates = [d for d in trading_dates if d < end_date]
    
    print(f"生成 {len(trading_dates)} 个交易日期")

    print("初始化LLM增强策略...")
    try:
        enhanced_strategy = EnhancedTradingStrategy(STOCK_POOL, LLM_API_KEY)
        
        # 运行回测
        results = evaluator.run_enhanced_backtest(
            enhanced_strategy, all_data, trading_dates
        )
        
        # 生成详细报告
        evaluator.generate_detailed_report(results)
        
        # 绘制分析图表
        evaluator.plot_enhanced_analysis(results, "enhanced_strategy_detailed_analysis.png")
        
        # 保存结果
        evaluator.save_results(results, "enhanced_strategy_evaluation.json")
        
        # 保存策略分析结果
        enhanced_strategy.save_analysis_results("llm_analysis_details.json")
        
        print("\n LLM增强策略评估完成！")
        
    except Exception as e:
        print(f" 增强策略评估失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
