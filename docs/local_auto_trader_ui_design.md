# BTC 本机自动交易 UI 测试版设计

更新时间：2026-05-29

## 目标

做一个本机运行的 Web UI，用来测试 `BTC-USDT-SWAP` 4H 唐奇安趋势突破策略的自动交易流程。第一版只做 dry-run，不真实下单。

## 范围

第一版包含：

```text
1. 本机浏览器 UI
2. 实时拉取 OKX 行情
3. 展示实时预警和收盘确认
4. 展示入场价、止损价、3R止盈、仓位数量
5. 支持记录 dry-run 拟下单计划
6. 保存运行日志和 dry-run 记录
7. 一键启动脚本
```

第一版不包含：

```text
1. 真实 OKX 下单
2. OKX 模拟盘下单
3. API Key 输入和保存
4. 自动平仓和跟踪止损实盘执行
```

## 技术方案

使用纯 Python 标准库实现本地 HTTP 服务：

```text
服务脚本：local_auto_trader_ui.py
启动脚本：start_local_auto_trader_ui.bat
访问地址：http://127.0.0.1:8765
状态文件：data/local_auto_trader_state.json
```

选择标准库而不是 FastAPI/Flask，是因为当前本机没有安装这些依赖，第一版可以做到开箱即用。

## 数据流

```text
浏览器 UI
  -> GET /api/status
  -> Python 服务拉 OKX 4H/15m
  -> 复用 trend_breakout_opportunity_scanner 的策略判断
  -> 返回 JSON 给 UI
```

dry-run 记录：

```text
UI 点击记录 dry-run
  -> POST /api/dry-run
  -> 若收盘确认有信号，保存拟下单计划
  -> 若无信号，只写日志，不生成交易
```

## 风控原则

```text
1. 默认模式永远是 dry-run
2. 页面明确显示真实下单关闭
3. 无收盘确认信号时不能记录拟下单
4. 同一根4H信号不重复记录
5. 所有拟下单计划写入本地 JSON
```

## UI 布局

```text
顶部：品种、运行模式、更新时间、连接状态
左侧：实时预警、收盘确认
中间：执行计划、仓位、止损止盈
右侧：风控状态、系统状态
底部：dry-run 记录和运行日志
```

## 后续升级路径

```text
第二版：接 OKX 模拟盘 API
第三版：加入真实下单开关，但默认关闭
第四版：部署到 NAS / VPS，加入通知和守护进程
```
