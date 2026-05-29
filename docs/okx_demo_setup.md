# OKX 模拟盘接入说明

更新时间：2026-05-29

## 先处理已暴露的 key

你刚才贴到聊天里的 key 已经视为暴露。建议在欧易模拟盘 API 管理里删除它们，然后重新创建一组新 key。新的 `API Key`、`Secret Key`、`Passphrase` 只填到本机 `.env`，不要再发到聊天里。

权限只开：

```text
读取
交易
```

不要开提现权限。

## IP 白名单

本机当前查询到的公网 IP：

```text
50.7.252.69
```

我分别用直连和 `127.0.0.1:7897` 代理查到的结果都是这个 IP。后面如果你换网络、换代理、重启路由器，IP 可能变化，需要重新查。

## 本机配置

项目目录下已经有 `.env`，只在本机编辑它：

```text
OKX_TRADING_MODE=demo
OKX_API_KEY=你的新模拟盘API_KEY
OKX_SECRET_KEY=你的新模拟盘SECRET_KEY
OKX_PASSPHRASE=你的passphrase
OKX_POS_SIDE=net
OKX_ENABLE_DEMO_ORDER=false
```

第一次先保持：

```text
OKX_ENABLE_DEMO_ORDER=false
```

这样 UI 只能测试连接和看信号，不能提交模拟盘订单。确认连接通过后，再改成：

```text
OKX_ENABLE_DEMO_ORDER=true
```

## 启动

双击：

```text
start_local_auto_trader_ui.bat
```

打开：

```text
http://127.0.0.1:8765
```

页面里有两个 OKX 模拟盘按钮：

```text
测试连接
提交模拟盘订单
```

提交订单仍然有三道限制：

```text
1. 必须是 OKX_TRADING_MODE=demo
2. 必须 OKX_ENABLE_DEMO_ORDER=true
3. 必须收盘确认存在做多或做空信号
```

真实盘下单在当前版本仍然被代码禁用。
