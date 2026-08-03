# Codex 当前会话迁移

当前会话已经打包到数据盘：

`/root/autodl-tmp/codex_migration/thread_019fa64b-8fdc-79f0-ab96-6fa36accbeda/`

包内包含完整会话 JSONL、Codex 会话状态数据库副本、会话索引和 `MANIFEST.json` 校验清单，不包含 `auth.json` 或其他认证秘密。

在目标机器上把数据盘挂载到相同路径、登录同一个 ChatGPT 账户，并关闭 Codex 后执行：

```bash
python /root/autodl-tmp/DiffusioNeRF-main/scripts/restore_codex_session.py \
  /root/autodl-tmp/codex_migration/thread_019fa64b-8fdc-79f0-ab96-6fa36accbeda \
  --replace-state
```

然后重启 Codex Desktop。恢复脚本会先备份目标机已有的 `state_5.sqlite`，再注册该会话。目标机的项目目录也应保持为 `/root/autodl-tmp/DiffusioNeRF-main`，这样会话中的工作区路径可以继续访问。
