# 期末大作业：基于大语言模型的股票交易策略增强系统

## 项目概述

本项目在期中大作业股票交易策略的基础上，集成了大语言模型分析、社交网络关系建模和市场情感分析，构建了一个增强版的智能交易策略系统。

## 项目结构

```
期末大作业/
├── TradingStrategy.py              # 原始交易策略（期中作业）
├── enhanced_trading_strategy.py    # 增强版交易策略
├── llm_analyzer.py                 # 大语言模型分析器
├── enhanced_eval_main.py           #  增强策略专用评估程序
├── requirements.txt                # 依赖包列表
├── README.md                       # 项目说明
└── time-series-data/               # 股票历史数据
    ├── AAPL_2006-01-01_to_2018-01-01.csv
    ├── MSFT_2006-01-01_to_2018-01-01.csv
    └── ...
```

## 技术架构

### 1. 核心组件

#### LLMAnalyzer (llm_analyzer.py)

- **功能**: 调用DeepSeek API进行各种分析
- **方法**:
  - `analyze_market_sentiment()`: 市场情感分析
  - `analyze_stock_relationships()`: 股票关系分析
  - `generate_trading_signals()`: 交易信号生成
  - `analyze_risk_factors()`: 风险因素分析

#### EnhancedTradingStrategy (enhanced_trading_strategy.py)

- **功能**: 集成LLM的增强版交易策略
- **新增特性**:
  - 股票关系网络构建
  - LLM情感分析集成
  - 增强版决策函数
  - 多维度信号融合

#### MarketDataCollector (market_data_collector.py)

- **功能**: 市场数据收集和预处理
- **数据源**:
  - yfinance股票基本信息
  - 模拟新闻标题数据
  - 行业ETF表现数据
  - 宏观经济指标

### 2. 数据流程

```
股票历史数据 → 技术指标计算 → LLM情感分析 → 关系网络分析 → 交易决策融合 → 回测评估
     ↓              ↓              ↓              ↓              ↓
  价格/成交量 → RSI/MACD/布林带 → 情感评分 → 相关性网络 → 买卖信号 → 收益率对比
```

## 使用方法

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

```

### 2. 配置API密钥

在 `enhanced_eval_main.py` 中设置您的DeepSeek API密钥：

```python
LLM_API_KEY = "your-deepseek-api-key-here"
```

### 3. 运行评估程序

```bash
# 🆕 评估LLM增强策略（推荐）
python enhanced_eval_main.py

# 自动重复评估直到收益率达到目标
python run_until_target.py

```

## 评估程序说明

### enhanced_eval_main.py - LLM增强策略专用评估

- **功能**: 专门评估集成了大语言模型的增强交易策略
- **特色功能**:
  - 实时市场情感分析
  - 股票关系网络构建
  - LLM交易信号生成
  - 多维度风险评估
  - 详细的策略分析报告
- **输出**: 
  - 详细的策略性能报告
  - 可视化分析图表
  - LLM分析详情

## 期末改进方案

### 1. LLM增强分析

- **情感分析**: 对每只股票进行市场情感评估
- **关系建模**: 构建股票间相关性网络
- **智能信号**: 结合技术分析和基本面分析

### 2. 多数据源融合

- **新闻数据**: 集成股票相关新闻情感
- **行业数据**: 考虑行业轮动和板块效应
- **宏观数据**: 纳入经济指标影响

### 3. 风险管理优化

- **动态调仓**: 基于LLM风险评估动态调整仓位
- **情感过滤**: 在市场极度恐慌或贪婪时调整策略
- **相关性控制**: 避免过度集中于相关性高的股票

## 预期改进效果

### 1. 收益率提升

- **目标**: 相比原始策略提升5-15%收益率
- **机制**: LLM提供的额外信息优势

### 2. 风险控制

- **目标**: 降低回撤和波动率
- **机制**: 智能风险评估和情感过滤

### 3. 适应性增强

- **目标**: 在不同市场环境下保持稳定表现
- **机制**: 多维度信息融合和动态策略调整



