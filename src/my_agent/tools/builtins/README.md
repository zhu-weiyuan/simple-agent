# builtins

内置工具由生产入口注册后，LLM 才能通过 tool call 使用。file.py 提供 read_file 和 list_files；calculator.py 做计算；time.py 查询时间；shell.py 执行命令，安全风险最高。调试文件工具时按“注册 → /api/tools → engine tool loop”排查。
