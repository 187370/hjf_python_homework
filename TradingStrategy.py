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

warnings.filterwarnings("ignore")


class TradingStrategy:
    def __init__(self, stock_pool):
        """
        初始化参数：
        stock_pool: 允许交易的股票代码列表
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

    def _build_model(self, input_dim):
        """
        构建LSTM模型
        """

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
        """创建技术指标特征"""
        # 基础特征：价格、交易量
        df["returns"] = df["Close"].pct_change()
        df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
        df["log_volume"] = np.log(df["Volume"] + 1)

        # 趋势指标
        df["sma_10"] = ta.trend.sma_indicator(df["Close"], window=10)
        df["sma_20"] = ta.trend.sma_indicator(df["Close"], window=20)
        df["sma_50"] = ta.trend.sma_indicator(df["Close"], window=50)
        df["macd"] = ta.trend.macd_diff(df["Close"])
        df["adx"] = ta.trend.adx(df["High"], df["Low"], df["Close"])

        # 波动性指标
        df["atr"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"])
        df["bollinger_high"] = ta.volatility.bollinger_hband(df["Close"])
        df["bollinger_low"] = ta.volatility.bollinger_lband(df["Close"])

        # 动量指标
        df["rsi"] = ta.momentum.rsi(df["Close"])
        df["stoch"] = ta.momentum.stoch(df["High"], df["Low"], df["Close"])
        df["cci"] = ta.trend.cci(df["High"], df["Low"], df["Close"])

        # 额外特征 - 基于聚类结果添加
        if self.clusters is not None and df["Name"].iloc[0] in self.clusters:
            cluster_id = self.clusters[df["Name"].iloc[0]]
            df["cluster"] = cluster_id

        # 填充NaN值
        df = df.fillna(method="ffill").fillna(0)

        # 计算未来n天的收益率作为标签
        df["future_return"] = (
            df["Close"].pct_change(self.prediction_days).shift(-self.prediction_days)
        )

        return df

    def cluster_stocks(self, all_stock_data):
        """
        对股票进行聚类，分为n_clusters类
        :param all_stock_data: {stock: pd.DataFrame}
        """
        print(f"[聚类] 开始对{len(all_stock_data)}只股票进行特征提取和聚类分析...")
        feature_list = []
        stock_names = []

        if not all_stock_data:
            print("[聚类] 输入的股票数据为空，无法进行聚类。")
            self.stock2cluster = {}
            self.clusters = {}
            return

        for stock, df in all_stock_data.items():
            df_copy = df.copy()
            df_copy = self._create_features(df_copy)
            features = df_copy.drop(
                ["Date", "Name", "future_return"], axis=1, errors="ignore"
            )
            features = features.fillna(
                0
            )  # Ensure NaNs from feature creation are handled before tail
            tail = features.tail(60)

            if tail.empty:
                print(f"[聚类] 股票 {stock} 数据处理后为空，跳过。")
                continue

            mean_feat = tail.mean().values
            # 波动率（年化标准差）
            volatility = tail["returns"].std() * np.sqrt(252)
            # 夏普比率（假设无风险利率为0）
            sharpe = (
                tail["returns"].mean() / (tail["returns"].std() + 1e-8) * np.sqrt(252)
            )
            # 最大回撤
            cum_ret = (1 + tail["returns"]).cumprod()
            if cum_ret.isna().all():
                max_drawdown = 0.0
            else:
                drawdown = (cum_ret.cummax() - cum_ret) / cum_ret.cummax()
                max_drawdown = drawdown.max() if not drawdown.empty else 0.0

            # Hurst指数
            def hurst(ts):
                lags = range(2, 20)
                tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return poly[0] * 2.0

            try:
                hurst_exp = hurst(tail["Close"].values)
            except Exception:
                hurst_exp = 0.0  # Default to 0.0 if calculation fails

            macd_mean = tail["macd"].mean()

            # Convert all potential NaNs or Infs from calculations to numbers
            mean_feat = np.nan_to_num(mean_feat, nan=0.0, posinf=0.0, neginf=0.0)
            volatility = np.nan_to_num(volatility, nan=0.0, posinf=0.0, neginf=0.0)
            sharpe = np.nan_to_num(sharpe, nan=0.0, posinf=0.0, neginf=0.0)
            max_drawdown = np.nan_to_num(max_drawdown, nan=0.0, posinf=0.0, neginf=0.0)
            hurst_exp = np.nan_to_num(hurst_exp, nan=0.0, posinf=0.0, neginf=0.0)
            macd_mean = np.nan_to_num(macd_mean, nan=0.0, posinf=0.0, neginf=0.0)

            cluster_feat = np.concatenate(
                [mean_feat, [volatility, sharpe, max_drawdown, hurst_exp, macd_mean]]
            )
            feature_list.append(cluster_feat)
            stock_names.append(stock)

        if not feature_list:
            print("[聚类] 没有有效的股票特征数据用于聚类分析。")
            self.stock2cluster = {}
            self.clusters = {}
            return

        feature_arr = np.array(feature_list)

        if feature_arr.shape[0] < self.n_clusters:
            print(
                f"[聚类] 样本数量 ({feature_arr.shape[0]}) 少于聚类数量 ({self.n_clusters})。无法进行有效聚类。"
            )
            self.stock2cluster = {}
            self.clusters = {}
            return

        print(
            f"[聚类] 特征提取完成，使用 {feature_arr.shape[0]} 个样本进行KMeans聚类..."
        )
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(feature_arr)
        self.stock2cluster = {stock: int(c) for stock, c in zip(stock_names, clusters)}
        self.clusters = self.stock2cluster  # 兼容原有接口
        for cid in range(self.n_clusters):
            count = sum(1 for v in self.stock2cluster.values() if v == cid)
            print(f"[聚类] 类别{cid} 包含股票数: {count}")
        print("[聚类] 聚类分析完成。")

    def train_cluster_models(self, all_stock_data):
        """
        为每个类别训练一个模型
        :param all_stock_data: {stock: pd.DataFrame}
        """
        print("[聚类模型] 开始为每个类别训练LSTM模型...")
        cluster_data = {i: [] for i in range(self.n_clusters)}
        for stock, df in all_stock_data.items():
            if stock in self.stock2cluster:
                df = df.copy()
                df = self._create_features(df)
                cluster_id = self.stock2cluster[stock]
                cluster_data[cluster_id].append(df)
        for cid, dfs in cluster_data.items():
            if not dfs:
                print(f"[聚类模型] 类别{cid} 无数据，跳过。")
                continue
            print(
                f"[聚类模型] 正在训练类别{cid}的模型，样本数: {sum(len(df) for df in dfs)}"
            )
            all_df = pd.concat(dfs, ignore_index=True)
            features = all_df.drop(
                ["Date", "Name", "future_return"], axis=1, errors="ignore"
            ).fillna(0)
            target = all_df["future_return"].fillna(0)
            scaler = MinMaxScaler()
            features_scaled = scaler.fit_transform(features)
            X, y = [], []
            window_size = self.window_size
            pred_days = self.prediction_days
            for i in range(len(features_scaled) - window_size - pred_days + 1):
                X.append(features_scaled[i : i + window_size])
                y.append(target.iloc[i + window_size + pred_days - 1])
            if not X:
                print(f"[聚类模型] 类别{cid} 样本过少，跳过。")
                continue
            X = np.array(X)
            y = np.array(y).reshape(-1, 1)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            X_train = torch.FloatTensor(X_train).to(self.device)
            y_train = torch.FloatTensor(y_train).to(self.device)
            X_test = torch.FloatTensor(X_test).to(self.device)
            y_test = torch.FloatTensor(y_test).to(self.device)
            model = self._build_model(X_train.shape[2])
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            patience = 10
            best_loss = float("inf")
            counter = 0
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            epochs = 50
            for epoch in range(epochs):
                model.train()
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    test_outputs = model(X_test)
                    test_loss = criterion(test_outputs, y_test).item()
                if test_loss < best_loss:
                    best_loss = test_loss
                    counter = 0
                    torch.save(model.state_dict(), f"cluster_model_{cid}.pth")
                else:
                    counter += 1
                    if counter >= patience:
                        break
                if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                    print(
                        f"[聚类模型] 类别{cid} Epoch {epoch+1}/{epochs}, Test Loss: {test_loss:.6f}"
                    )
            model.load_state_dict(torch.load(f"cluster_model_{cid}.pth"))
            self.cluster_models[cid] = model
            self.cluster_scalers[cid] = scaler
            print(f"[聚类模型] 类别{cid} 模型训练完成，最佳Test Loss: {best_loss:.6f}")
        print("[聚类模型] 所有类别模型训练完成。")

    def _prepare_data(self, df, stock_name):
        """准备训练数据"""
        features = df.drop(
            ["Date", "Name", "future_return"], axis=1, errors="ignore"
        ).copy()
        target = df["future_return"].copy()

        # 处理缺失值
        features = features.fillna(0)
        target = target.fillna(0)

        # 将数据标准化
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)
        self.scalers[stock_name] = scaler

        # 创建滑动窗口数据
        X, y = [], []
        for i in range(
            len(features_scaled) - self.window_size - self.prediction_days + 1
        ):
            X.append(features_scaled[i : i + self.window_size])
            y.append(target.iloc[i + self.window_size + self.prediction_days - 1])

        # 转换为numpy数组
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # 转换为PyTorch张量
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_test = torch.FloatTensor(y_test).to(self.device)

        return X_train, y_train, X_test, y_test, features.columns.tolist()

    def _train_model(self, X_train, y_train, X_test, y_test, input_dim, stock_name):
        """训练模型"""
        model = self._build_model(input_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 早停设置
        patience = 10
        best_loss = float("inf")
        counter = 0
        early_stop = False

        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # 训练循环
        epochs = 100
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # 验证
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs, y_test).item()

            # 早停
            if test_loss < best_loss:
                best_loss = test_loss
                counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), f"model_{stock_name}.pth")
            else:
                counter += 1
                if counter >= patience:
                    early_stop = True
                    break

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Test Loss: {test_loss:.4f}"
                )

            if early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # 加载最佳模型
        model.load_state_dict(torch.load(f"model_{stock_name}.pth"))
        self.models[stock_name] = model

        return model

    def preprocess_data(self, real_value, enable_cluster=False):
        """
        数据预处理
        """
        stock_data = {}
        for stock, data in real_value.items():
            if stock in self.stock_pool:
                df = pd.DataFrame(
                    data,
                    columns=["Date", "Open", "High", "Low", "Close", "Volume", "Name"],
                )
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.sort_values("Date")
                df = self._create_features(df)
                stock_data[stock] = df

                # 只有不启用聚类时才训练单股票模型
                if not enable_cluster and stock not in self.models:
                    X_train, y_train, X_test, y_test, columns = self._prepare_data(
                        df, stock
                    )
                    if len(X_train) > 0:
                        self._train_model(
                            X_train, y_train, X_test, y_test, X_train.shape[2], stock
                        )

        return stock_data

    def _predict(self, stock, df, enable_cluster=False):
        """预测股票未来收益率，支持聚类模型"""
        if enable_cluster and self.stock2cluster and stock in self.stock2cluster:
            cid = self.stock2cluster[stock]
            if cid not in self.cluster_models or cid not in self.cluster_scalers:
                return 0
            model = self.cluster_models[cid]
            scaler = self.cluster_scalers[cid]
        else:
            if stock not in self.models or stock not in self.scalers:
                return 0
            model = self.models[stock]
            scaler = self.scalers[stock]
        features = (
            df.drop(["Date", "Name", "future_return"], axis=1, errors="ignore")
            .copy()
            .fillna(0)
        )
        features_scaled = scaler.transform(features)
        X = features_scaled[-self.window_size :].reshape(1, self.window_size, -1)
        X = torch.FloatTensor(X).to(self.device)
        model.eval()
        with torch.no_grad():
            predicted_return = model(X).cpu().numpy()[0][0]
        return predicted_return

    def _calculate_signal_strength(self, stock, df, predicted_return):
        """计算交易信号强度"""
        # 基于技术指标和预测收益率计算信号强度
        signal = 0

        # 预测收益率信号
        if predicted_return > 0.02:  # 预测收益率大于2%
            signal += 2
        elif predicted_return > 0.01:  # 预测收益率大于1%
            signal += 1
        elif predicted_return < -0.02:  # 预测收益率小于-2%
            signal -= 2
        elif predicted_return < -0.01:  # 预测收益率小于-1%
            signal -= 1

        # MACD信号
        recent_macd = df["macd"].iloc[-1]
        if recent_macd > 0:
            signal += 1
        elif recent_macd < 0:
            signal -= 1

        # RSI信号
        recent_rsi = df["rsi"].iloc[-1]
        if recent_rsi > 70:  # 超买
            signal -= 1
        elif recent_rsi < 30:  # 超卖
            signal += 1

        # 趋势信号
        sma_short = df["sma_10"].iloc[-1]
        sma_long = df["sma_50"].iloc[-1]
        if sma_short > sma_long:  # 短期均线在长期均线上方
            signal += 1
        elif sma_short < sma_long:  # 短期均线在长期均线下方
            signal -= 1

        # 布林带信号
        close = df["Close"].iloc[-1]
        bollinger_high = df["bollinger_high"].iloc[-1]
        bollinger_low = df["bollinger_low"].iloc[-1]
        if close > bollinger_high:  # 价格高于布林带上轨
            signal -= 1
        elif close < bollinger_low:  # 价格低于布林带下轨
            signal += 1

        # 根据聚类信息调整信号
        if "cluster" in df.columns:
            cluster_id = df["cluster"].iloc[0]
            # 不同聚类组使用不同的策略权重
            # 这里根据前面的聚类分析结果进行调整
            if cluster_id == 0:  # 假设聚类0是高风险高收益组
                signal = signal * 1.5
            elif cluster_id == 1:  # 假设聚类1是低风险低收益组
                signal = signal * 0.8

        return signal

    def _determine_action(self, signal, stock, portfolio, current_price):
        """根据信号确定交易行为"""
        current_holdings = portfolio["holdings"].get(stock, 0)
        cash = portfolio["cash"]
        total_value = cash

        # 计算当前持仓总价值
        for s, shares in portfolio["holdings"].items():
            # 使用最后一个收盘价作为当前价格
            stock_price = current_price.get(s, 0)
            total_value += shares * stock_price

        # 计算单个股票最大持仓价值
        max_position_value = total_value * 0.2  # 单个股票不超过总资产的20%
        max_shares = min(
            10000, int(max_position_value / current_price.get(stock, 1))
        )  # 最大10000股

        # 计算当前持仓价值
        current_position_value = current_holdings * current_price.get(stock, 0)

        # 根据信号确定操作
        if signal > 2:  # 强买信号
            if current_position_value < max_position_value:
                # 计算可买入的股数
                affordable_shares = min(
                    max_shares - current_holdings,
                    int(cash / current_price.get(stock, 1)),
                )
                if affordable_shares > 0:
                    return "buy", affordable_shares
        elif signal < -2:  # 强卖信号
            if current_holdings > 0:
                return "sell", current_holdings
        elif signal > 0:  # 弱买信号
            if current_position_value < max_position_value / 2:
                # 计算可买入的股数
                affordable_shares = min(
                    max_shares - current_holdings,
                    int(cash / current_price.get(stock, 1) / 2),
                )
                if affordable_shares > 0:
                    return "buy", affordable_shares
        elif signal < 0:  # 弱卖信号
            if current_holdings > 0:
                return "sell", current_holdings // 2

        return "none", 0

    def _determine_hedge_action(
        self, signal, action1, shares1, stock, portfolio, current_price
    ):
        """确定对冲操作（第二个action）"""
        # 如果第一个动作是买入，第二个动作考虑对冲或增强
        if action1 == "buy":
            # 如果是强烈买入信号但资金不足以全部买入，第二个操作为none
            if signal > 3:
                return "none", 0
            # 如果是中等买入信号，可以考虑同时卖出其他股票（资金再平衡）
            elif signal > 1:
                return "none", 0
            # 如果是弱买入信号，可以考虑设置止损
            else:
                return "sell", shares1  # 止损单，如果价格下跌则卖出

        # 如果第一个动作是卖出，第二个动作考虑对冲或再入场
        elif action1 == "sell":
            # 如果是强卖信号，第二个操作为none
            if signal < -3:
                return "none", 0
            # 如果是中等卖出信号，可以考虑部分再买入（止盈回补）
            elif signal < -1:
                return "buy", shares1 // 2
            # 如果是弱卖出信号，可以考虑完全再买入
            else:
                return "buy", shares1

        # 如果第一个动作是none，第二个动作考虑观望或小仓位试探
        else:
            # 如果信号接近临界值，可以小仓位试探
            if abs(signal) > 1.5 and abs(signal) < 2:
                if signal > 0:
                    # 小仓位买入试探
                    cash = portfolio["cash"]
                    if cash > 0:
                        affordable_shares = min(
                            1000, int(cash / current_price.get(stock, 1) / 10)
                        )
                        if affordable_shares > 0:
                            return "buy", affordable_shares
                else:
                    # 做空试探
                    current_holdings = portfolio["holdings"].get(stock, 0)
                    if current_holdings > 0:
                        return "sell", min(current_holdings, 1000)

            return "none", 0

    def add_cluster_info(self, clusters):
        """添加聚类信息"""
        self.clusters = clusters

    def generate_strategy(
        self, portfolio, date, real_value, next_trading_date=None, enable_cluster=False
    ):
        """
        生成每日交易策略，根据当前日期和下一个交易日的日期间隔调整策略
        :param
        portfolio: 当前投资组合字典，包含以下字段：
        {
            'cash': 当前现金余额 float,
            'holdings': 目前持股信息 {stock: shares},
            'transaction_log': 历史交易记录 []
        }
        date: 需要决策的日期k, str 或 datetime 对象
        real_value: 包含前面k-1天股票的真实开盘收盘价，最高价和最低价->Dict[str, List[List[Any]]]
        {
            'AAPL':[[Date,Open,High,Low,Close,Volume,Name], [Date,Open,High,Low,Close,Volume,Name], ......]
        }
        next_trading_date: 下一个交易日的日期，决定是短期还是长期策略
        enable_cluster: 是否启用聚类模型
        :return: 交易策略列表->List[Dict[str, Dict[str, Any]]]
        [
            {'AAPL': {'action1': 'buy', 'shares1': 100, 'action2':'sell', 'shares2':50}},
            {'MSFT': {'action1': 'none', 'shares1': 0, 'action2': 'none', 'shares2': 50}}
        ]
        """
        # ----------- 每个测试点都重新建模 -----------
        if enable_cluster:
            # 用当前real_value重新聚类和训练聚类模型
            all_stock_data_for_training = {}
            for stock, data_list_for_stock in real_value.items():
                if not data_list_for_stock:
                    continue
                if stock in self.stock_pool:
                    df = pd.DataFrame(
                        data_list_for_stock,  # Use data_list_for_stock
                        columns=[
                            "Date",
                            "Open",
                            "High",
                            "Low",
                            "Close",
                            "Volume",
                            "Name",
                        ],
                    )
                    df["Date"] = pd.to_datetime(df["Date"])
                    df = df.sort_values("Date")
                    all_stock_data_for_training[stock] = df

            if not all_stock_data_for_training:
                print("[策略生成] 没有可用于训练聚类模型的数据。")  # Log message
                self.clusters = {}
                self.stock2cluster = {}
                self.cluster_models = {}
                self.cluster_scalers = {}
                # self.models and self.scalers are cleared below, so this is fine
            else:
                self.cluster_stocks(all_stock_data_for_training)
                self.train_cluster_models(all_stock_data_for_training)

            self.models = {}
            self.scalers = {}
        else:
            # 每次都重新训练所有单股票模型
            self.models = {}
            self.scalers = {}
            for stock, data in real_value.items():
                if stock in self.stock_pool:
                    df = pd.DataFrame(
                        data,
                        columns=[
                            "Date",
                            "Open",
                            "High",
                            "Low",
                            "Close",
                            "Volume",
                            "Name",
                        ],
                    )
                    df["Date"] = pd.to_datetime(df["Date"])
                    df = df.sort_values("Date")
                    df = self._create_features(df)
                    X_train, y_train, X_test, y_test, columns = self._prepare_data(
                        df, stock
                    )
                    if len(X_train) > 0:
                        self._train_model(
                            X_train, y_train, X_test, y_test, X_train.shape[2], stock
                        )

        # 预处理数据
        stock_data = self.preprocess_data(real_value, enable_cluster=enable_cluster)

        # 生成交易策略
        strategy_list = []
        current_price = {}

        # 获取每支股票的当前价格
        for stock, data in real_value.items():
            if stock in self.stock_pool and len(data) > 0:
                current_price[stock] = data[-1][4]  # 最后一天的收盘价

        # 设置可交易的股票数量上限
        max_stocks_to_trade = min(6, len(self.stock_pool))
        stocks_to_trade = []

        # 计算每支股票的信号强度
        stock_signals = {}
        for stock, df in stock_data.items():
            if stock in self.stock_pool:
                # 预测未来收益率
                predicted_return = self._predict(
                    stock, df, enable_cluster=enable_cluster
                )

                # 计算信号强度
                signal = self._calculate_signal_strength(stock, df, predicted_return)
                stock_signals[stock] = signal

        # 按信号强度绝对值排序，选择信号最强的股票进行交易
        sorted_stocks = sorted(
            stock_signals.items(), key=lambda x: abs(x[1]), reverse=True
        )
        stocks_to_trade = [stock for stock, _ in sorted_stocks[:max_stocks_to_trade]]

        # 短期与长期交易模式判断
        is_short_term = True
        if next_trading_date:
            # 确保 date 是 datetime 对象
            date_obj = (
                date
                if isinstance(date, datetime)
                else datetime.strptime(date, "%Y-%m-%d")
            )
            # 确保 next_trading_date 是 datetime 对象
            next_date_obj = (
                next_trading_date
                if isinstance(next_trading_date, datetime)
                else datetime.strptime(next_trading_date, "%Y-%m-%d")
            )
            days_between = (next_date_obj - date_obj).days
            if days_between > 3:  # 如果下一个交易日超过3天，考虑长期策略
                is_short_term = False

        # 生成交易策略
        for stock in stocks_to_trade:
            signal = stock_signals[stock]

            # 根据是否短期交易调整信号
            if not is_short_term:
                # 长期交易更关注趋势
                if abs(signal) < 2:  # 弱信号可能不足以支撑长期持仓
                    signal = signal * 0.7  # 减弱信号

            # 确定第一个操作
            action1, shares1 = self._determine_action(
                signal, stock, portfolio, current_price
            )

            # 根据第一个操作确定第二个操作（对冲或补充策略）
            action2, shares2 = self._determine_hedge_action(
                signal, action1, shares1, stock, portfolio, current_price
            )

            # 将操作添加到策略列表
            strategy_list.append(
                {
                    stock: {
                        "action1": action1,
                        "shares1": shares1,
                        "action2": action2,
                        "shares2": shares2,
                    }
                }
            )

        return strategy_list
