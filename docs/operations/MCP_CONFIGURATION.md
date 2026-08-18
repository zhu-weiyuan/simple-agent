# MCP 服务器配置

SimpleAgent 使用显式的 JSON 配置加载外部 MCP（Model Context Protocol）工具服务器。配置路径由环境变量 `MCP_CONFIG_PATH` 指定；没有该变量时不会自动启动任何外部 MCP 进程。

这样不会因为另一款桌面应用保存了 MCP 配置就自动执行其中的命令。若要复用 AstrCode 的配置，显式设置：

```env
MCP_CONFIG_PATH=C:/Users/Administrator/.astrcode/mcp.json
```

也可以使用 SimpleAgent 自己的文件，例如 `C:/Users/Administrator/.simpleagent/mcp.json`。

可直接复制仓库内的 xamples/mcp.json.example 到你的配置位置，再把允许目录、服务地址和环境变量改为自己的值。

## 配置格式

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:/workspace/allowed"],
      "env": {}
    },
    "web-reader": {
      "type": "http",
      "url": "https://mcp.example.com/mcp",
      "headers": {
        "Authorization": "Bearer ${WEB_READER_TOKEN}"
      },
      "timeoutSeconds": 30
    }
  }
}
```

- 缺省 `type` 表示 `stdio`；stdio 以子进程方式启动，`command` 与 `args` 分开传递，**不会**通过 shell 执行。
- `type: "http"` 使用 JSON-RPC 的 MCP Streamable HTTP 请求，并兼容单条 SSE JSON-RPC 响应。
- `env` 和 `headers` 支持 `${VARIABLE}` 环境变量展开。若变量不存在，服务器不会加载，避免意外使用空令牌。
- `timeoutSeconds` 范围是 1–120 秒，默认 30 秒。
- 每个远程工具会注册为 `mcp_<服务器名>_<工具名>`，例如 `mcp_filesystem_list_directory`，不会覆盖 SimpleAgent 内置工具。
- 所有外部 MCP 工具默认处于 `ask` 权限级别，需要经过现有权限决策流程。

## 兼容旧配置

旧的 `MCP_WEATHER_COMMAND` 仍可作为单一 stdio 服务器的兼容回退；一旦设置 `MCP_CONFIG_PATH`，JSON 配置优先。

## 运行建议

仅配置你信任的本地命令和 HTTPS 服务器。文件系统 MCP 的允许目录应尽量收窄，不要把包含密钥的用户根目录直接交给外部服务器。

