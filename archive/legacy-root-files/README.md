# 旧根目录文件

这些文件已从项目根目录移入此处，避免与当前入口和运行状态混在一起。

- `start.bat`：过期启动器，目标为不存在的 `app:app`；请改用根目录 `start_simple_agent.bat` 或 `python app_prod.py`。
- `_api_inventory.txt`：2026 年 7 月生成的静态 API 清单，不会随源码更新。
- `runtime.sessions.json`：未被当前代码引用的历史会话状态快照。

`enhanced_state.json` **不是旧文件**：`src/my_agent/agent.py` 仍会读取和写入它，因此已经恢复到项目根目录。
