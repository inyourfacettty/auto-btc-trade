# GitHub Actions 每 4 小时邮件提醒配置

更新时间：2026-05-29

## 功能

GitHub Actions 会每 4 小时运行一次：

```text
python scheduled_status_email.py --send
```

它只会拉 OKX 公开行情、计算 BTC-USDT-SWAP 4H 唐奇安趋势突破策略状态，并把中文报告发送到邮箱。这个工作流不会读取 OKX API Key，也不会自动下单。

## GitHub Secrets

进入仓库：

```text
Settings -> Secrets and variables -> Actions -> New repository secret
```

添加这些 Secrets：

```text
SMTP_HOST=smtp.qq.com
SMTP_PORT=465
SMTP_USERNAME=你的QQ邮箱
SMTP_PASSWORD=你的QQ邮箱授权码
MAIL_FROM=你的QQ邮箱
MAIL_TO=接收提醒的邮箱
```

注意：`SMTP_PASSWORD` 填 QQ 邮箱授权码，不是 QQ 登录密码，也不要提交到代码里。

## 运行时间

当前工作流配置：

```text
7 */4 * * *
```

意思是 UTC 时间每 4 小时的第 7 分钟执行一次。换成北京时间大约是：

```text
00:07
04:07
08:07
12:07
16:07
20:07
```

GitHub 定时任务可能延迟几分钟，这是正常现象。

## 手动测试

在 GitHub 仓库页面打开：

```text
Actions -> BTC strategy status email -> Run workflow
```

第一次建议手动点一次，确认邮箱能收到。
