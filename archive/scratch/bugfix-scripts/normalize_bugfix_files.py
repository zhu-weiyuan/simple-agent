from pathlib import Path
root=Path(r"C:\Users\Administrator\.openclaw\workspace1\simple-agent")
p=root/'src/my_agent/tools/builtins/shell.py'
text=p.read_text(encoding='utf-8')
text=text.replace('import subprocess\nfrom typing import Any, Dict\n', 'import re\nimport subprocess\nfrom typing import Any, Dict, Optional\n')
p.write_text(text,encoding='utf-8',newline='\n')
for rel in [
 'src/my_agent/graph/graph.py',
 'src/my_agent/memory/store.py',
 'src/my_agent/mcp_client.py',
 'src/my_agent/core/engine.py',
 'src/my_agent/agent.py',
 'src/my_agent/core/context_assembler.py',
]:
 p=root/rel
 text=p.read_text(encoding='utf-8')
 p.write_text(text,encoding='utf-8',newline='\n')
