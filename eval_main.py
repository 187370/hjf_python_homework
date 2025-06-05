import pandas as pd
import numpy as np
import os
import glob
from collections import defaultdict
from datetime import datetime, timedelta
import random
from TradingStrategy import TradingStrategy
from tqdm import tqdm
import matplotlib.pyplot as plt

random.seed(42)


def load_stock_data(data_dir):
    all_data = {}
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    for file_path in csv_files:
        stock_code = os.path.basename(file_path).split("_")[0]
        df = pd.read_csv(file_path, parse_dates=["Date"])
        all_data[stock_code] = df
    return all_data


def prepare_test_data(all_data, start_date, end_date):
    test_data = {}
    for stock_code, df in all_data.items():
        mask = (df["Date"] >= start_date) & (df["Date"] < end_date)
        filtered_data = df.loc[mask]
        if not filtered_data.empty:
            data_list = []
            for _, row in filtered_data.iterrows():
                data_list.append(
                    [
                        row["Date"],
                        row["Open"],
                        row["High"],
                        row["Low"],
                        row["Close"],
                        row["Volume"],
                        stock_code,
                    ]
                )
            test_data[stock_code] = data_list
    return test_data


def generate_trading_dates(start_date, num_days, min_gap=1, max_gap=100):
    trading_dates = []
    current_date = start_date
    for _ in range(num_days):
        trading_dates.append(current_date)
        gap = random.randint(min_gap, max_gap)
        current_date = current_date + timedelta(days=gap)
    return trading_dates


def evaluate_strategy(
    strategy_class,
    all_stock_data,
    initial_cash=1000000,
    num_trading_days=60,
    enable_cluster=False,
):
    portfolio = {
        "cash": initial_cash,
        "holdings": defaultdict(int),
        "transaction_log": [],
    }
    asset_curve = []
    trading_days_record = []
    max_shares_per_trade = 10000
    max_position_value_ratio = 0.2
    max_short_ratio = 0.5
    start_date = datetime(2011, 1, 1)
    trading_dates = generate_trading_dates(start_date, num_trading_days)
    for i in tqdm(range(num_trading_days - 1)):
        current_date = trading_dates[i]
        next_trading_date = trading_dates[i + 1]
        test_data = prepare_test_data(all_stock_data, start_date, current_date)
        total_portfolio_value = portfolio["cash"]
        for stock, shares in portfolio["holdings"].items():
            if stock in test_data and test_data[stock]:
                stock_price = test_data[stock][-1][4]
                if shares > 0:
                    total_portfolio_value += stock_price * shares
                elif shares < 0:
                    total_portfolio_value -= stock_price * (-shares)
        asset_curve.append(total_portfolio_value)
        trading_days_record.append(current_date)
        print(
            f"[{current_date.strftime('%Y-%m-%d')}] 当前总资产: {total_portfolio_value:.2f}"
        )
        if current_date > start_date and i > 0:
            for stock, shares in list(portfolio["holdings"].items()):
                if shares == 0:
                    continue
                if stock in test_data and test_data[stock]:
                    closing_price = test_data[stock][-2][4]
                    if shares > 0:
                        portfolio["cash"] += closing_price * shares
                        portfolio["holdings"][stock] = 0
                    elif shares < 0:
                        portfolio["cash"] -= closing_price * (-shares)
                        portfolio["holdings"][stock] = 0
                    portfolio["holdings"] = {}
        strategy_list = strategy_class.generate_strategy(
            portfolio=portfolio,
            date=current_date,
            real_value=test_data,
            next_trading_date=next_trading_date,
            enable_cluster=enable_cluster,
        )
        strategy_dict = {}
        for item in strategy_list:
            for stock, order in item.items():
                strategy_dict[stock] = order
        valid_orders = []
        for strategy_item in strategy_list:
            for stock, order in strategy_item.items():
                if stock not in strategy_class.stock_pool:
                    continue
                if stock not in test_data or not test_data[stock]:
                    continue
                stock_data = test_data[stock][-1]
                current_open = stock_data[1]
                current_close = stock_data[4]
                if stock not in portfolio["holdings"]:
                    portfolio["holdings"][stock] = 0
                action1 = order.get("action1", "none")
                shares1 = min(order.get("shares1", 0), max_shares_per_trade)
                if action1 != "none" and shares1 > 0:
                    if action1 == "buy":
                        cost = current_open * shares1
                        if cost > portfolio["cash"]:
                            shares1 = int(portfolio["cash"] / current_open)
                        if shares1 <= 0:
                            continue
                        cost = current_open * shares1
                        new_position_value = (
                            portfolio["holdings"][stock] + shares1
                        ) * current_open
                        if (
                            new_position_value
                            > total_portfolio_value * max_position_value_ratio
                        ):
                            max_allowed_shares = int(
                                (
                                    total_portfolio_value * max_position_value_ratio
                                    - portfolio["holdings"][stock] * current_open
                                )
                                / current_open
                            )
                            shares1 = max(0, max_allowed_shares)
                            if shares1 <= 0:
                                continue
                            cost = current_open * shares1
                        portfolio["holdings"][stock] += shares1
                        portfolio["cash"] -= cost
                        valid_orders.append((stock, "buy", shares1))
                    elif action1 == "sell":
                        if (
                            stock in portfolio["holdings"]
                            and portfolio["holdings"][stock] > 0
                        ):
                            available_shares = portfolio["holdings"][stock]
                            sell_shares = min(shares1, available_shares)
                            portfolio["cash"] += current_close * sell_shares
                            portfolio["holdings"][stock] -= sell_shares
                            valid_orders.append((stock, "sell", sell_shares))
                        else:
                            short_value = shares1 * current_open
                            max_short_value = total_portfolio_value * max_short_ratio
                            if short_value > max_short_value:
                                shares1 = int(max_short_value / current_open)
                                if shares1 <= 0:
                                    continue
                            portfolio["holdings"][stock] -= shares1
                            portfolio["cash"] += current_close * shares1
                            valid_orders.append((stock, "short", shares1))
                action2 = order.get("action2", "none")
                shares2 = min(order.get("shares2", 0), max_shares_per_trade)
                if action2 != "none" and shares2 > 0:
                    if action2 == "buy":
                        cost = current_open * shares2
                        if cost > portfolio["cash"]:
                            shares2 = int(portfolio["cash"] / current_open)
                        if shares2 <= 0:
                            continue
                        cost = current_open * shares2
                        new_position_value = (
                            portfolio["holdings"][stock] + shares2
                        ) * current_open
                        if (
                            new_position_value
                            > total_portfolio_value * max_position_value_ratio
                        ):
                            max_allowed_shares = int(
                                (
                                    total_portfolio_value * max_position_value_ratio
                                    - portfolio["holdings"][stock] * current_open
                                )
                                / current_open
                            )
                            shares2 = max(0, max_allowed_shares)
                            if shares2 <= 0:
                                continue
                            cost = current_open * shares2
                        portfolio["holdings"][stock] += shares2
                        portfolio["cash"] -= cost
                        valid_orders.append((stock, "buy", shares2))
                    elif action2 == "sell":
                        if (
                            stock in portfolio["holdings"]
                            and portfolio["holdings"][stock] > 0
                        ):
                            available_shares = portfolio["holdings"][stock]
                            sell_shares = min(shares2, available_shares)
                            portfolio["cash"] += current_close * sell_shares
                            portfolio["holdings"][stock] -= sell_shares
                            valid_orders.append((stock, "sell", sell_shares))
                        else:
                            short_value = shares2 * current_open
                            max_short_value = total_portfolio_value * max_short_ratio
                            if short_value > max_short_value:
                                shares2 = int(max_short_value / current_open)
                                if shares2 <= 0:
                                    continue
                            portfolio["holdings"][stock] -= shares2
                            portfolio["cash"] += current_close * shares2
                            valid_orders.append((stock, "short", shares2))
        portfolio["transaction_log"].append(
            {
                "date": current_date,
                "valid_orders": valid_orders,
                "remaining_cash": portfolio["cash"],
                "holdings": dict(portfolio["holdings"]),
                "strategy": strategy_dict,
            }
        )
    final_date = trading_dates[-1]
    final_data = prepare_test_data(all_stock_data, start_date, final_date)
    final_value = portfolio["cash"]
    for stock, shares in list(portfolio["holdings"].items()):
        if shares == 0:
            continue
        if stock in final_data and final_data[stock]:
            final_price = final_data[stock][-1][4]
            if shares > 0:
                final_value += final_price * shares
            elif shares < 0:
                short_value = -shares * final_price
                final_value -= short_value
    asset_curve.append(final_value)
    trading_days_record.append(final_date)
    print(f"[{final_date.strftime('%Y-%m-%d')}] 空仓后总资产: {final_value:.2f}")
    portfolio["transaction_log"].append(
        {
            "date": final_date,
            "valid_orders": [],
            "remaining_cash": final_value,
            "holdings": {},
            "strategy": {},
        }
    )
    return final_value, portfolio["transaction_log"], asset_curve, trading_days_record


def main():
    data_dir = "./time-series-data"
    all_stock_data = load_stock_data(data_dir)
    stock_pool = list(all_stock_data.keys())
    user_input = input("是否使用聚类模型进行预测？(yes/no): ").strip().lower()
    if user_input == "yes":
        n_clusters = input("请输入聚类类别数（如5）: ").strip()
        try:
            n_clusters = int(n_clusters)
        except Exception:
            print("输入无效，使用默认类别数5")
            n_clusters = 5
        strategy = TradingStrategy(stock_pool)
        strategy.n_clusters = n_clusters
        enable_cluster = True
    else:
        strategy = TradingStrategy(stock_pool)
        enable_cluster = False
    final_value, transaction_log, asset_curve, trading_days_record = evaluate_strategy(
        strategy, all_stock_data, enable_cluster=enable_cluster
    )
    print(f"初始资金: 1,000,000")
    print(f"最终资产: {final_value:.2f}")
    print(f"收益率: {(final_value-1000000)/1000000*100:.2f}%")
    print("\n交易记录摘要:")
    for i, log in enumerate(transaction_log):
        print(f"日期: {log['date'].strftime('%Y-%m-%d')}")
        print(f"  交易: {len(log['valid_orders'])} 笔")
        print(f"  现金余额: {log['remaining_cash']:.2f}")
        print(f"  持仓股票数: {len(log['holdings'])}")
        print(f"  策略: {log['strategy']}")
        print("-" * 40)
    # 输出资产曲线图
    plt.figure(figsize=(10, 5))
    plt.plot(trading_days_record, asset_curve, marker="o")
    plt.title("资产变化曲线")
    plt.xlabel("日期")
    plt.ylabel("总资产")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
