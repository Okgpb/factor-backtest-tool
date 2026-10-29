# Crypto Factor Backtest Tool

面向 USDT 永续合约的多周期横截面因子回测框架。支持 Rank IC、分组收益、美元中性多空组合、
手续费、滑点、资金费率、BTC 基准和自包含 HTML 报告。

## 特性

- 字符串币对、UTC 时间和 7×24 小时年化
- 每周期独立因子文件，避免隐式重采样和未来函数
- 多个周期一键批量扫描
- Top/Bottom 分位数等权多空
- 手续费、滑点、资金费率收益拆解
- 动态流动性过滤与可选 point-in-time universe
- Binance USD-M 公共行情下载器
- 无网络依赖的确定性测试数据

## 安装

```bash
python -m pip install -e .
```

在不能联网下载构建依赖的隔离环境中，可使用：

```bash
python -m pip install -e . --no-build-isolation --no-deps
```

Python 3.10 及以上版本。

## 快速自测

仓库包含小型合成数据，只用于检查程序是否正常：

```bash
python scripts/run_crypto_factor_backtest.py --test-data
```

默认同时运行 `1h` 和 `4h`，结果保存到 `output/run_<UTC时间>/`。

## 真实回测

```bash
python scripts/run_crypto_factor_backtest.py \
  --config configs/default.yaml \
  --data-root D:/crypto/market \
  --factor-root D:/crypto/factors \
  --factor-column my_factor \
  --frequencies 1h 4h 1d
```

因子数值越高默认越优；低值更优时增加 `--reverse`。

行情目录和因子目录必须按周期分别存储。详细格式见
[`references/data-contract.md`](references/data-contract.md)。

## 下载 Binance 数据

```bash
python scripts/download_binance_usdm.py \
  --output-root D:/crypto/market \
  --symbols BTCUSDT ETHUSDT SOLUSDT \
  --frequencies 1h 4h \
  --start 2024-01-01 \
  --end 2025-01-01
```

不传 `--symbols` 时会读取当前活跃 USDT 永续合约，这可能产生幸存者偏差。正式研究建议提供
经过整理的历史币种列表以及 `universe.parquet`。

## 主要输出

```text
output/run_.../
├── run_summary.json
├── frequency_comparison.csv
├── report.html
├── 1h/
│   ├── summary.json
│   ├── stats.csv
│   ├── ic_series.csv
│   ├── ic_summary.csv
│   ├── group_returns.csv
│   ├── weights.parquet
│   ├── trades.parquet
│   ├── trades.csv
│   ├── funding_pnl.csv
│   ├── cost_breakdown.csv
│   └── report.html
└── 4h/
```

## 研究限制

这是向量化因子研究工具，不模拟订单簿、部分成交、保证金档位、强平和 ADL。任何实盘策略都应
在事件驱动撮合与风险系统中再次验证。本项目不构成投资建议。

## 开发与验证

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest
```

配置中的布尔值、整数和有限数值会严格校验。行情必须按指定周期连续排列；持仓或目标仓位缺少
开盘/收盘价格时会明确报错。`--start` 之后重新预热历史与流动性窗口，但会保留起始日之前的
最后一条交易池状态。详细计算口径见 [`references/backtest-methodology.md`](references/backtest-methodology.md)，
本次审查记录见 [`references/code-review.md`](references/code-review.md)。

## License

MIT
