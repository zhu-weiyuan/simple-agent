from pathlib import Path
import textwrap

root = Path(r"C:\Users\Administrator\.openclaw\workspace1\simple-agent")

def read(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return f.read()

def write(path, text):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)

def replace(path, old, new):
    text = read(path)
    if old not in text:
        raise RuntimeError(f"Target not found: {path}\n{old[:180]!r}")
    write(path, text.replace(old, new, 1))

# 1) Checkpoint loading must use the caller's run_id, not the first graph node.
p = root / "src/my_agent/graph/graph.py"
replace(p,
'''        if self._checkpointer_enabled:\r
            checkpoint = self._load_checkpoint(current)\r
            if checkpoint:\r
''',
'''        if self._checkpointer_enabled:\r
            checkpoint = self._load_checkpoint(current, state.run_id)\r
            if checkpoint:\r
''')
replace(p,
'''    def _load_checkpoint(self, node_name: str) -> Optional[Dict[str, Any]]:\r
        """加载检查点"""\r
        try:\r
            import os\r
            path = self._checkpoint_path or "checkpoints"\r
            checkpoint_file = os.path.join(path, f"{node_name}_{self._nodes_list()[0]}.json")  # 简化\r
            if os.path.exists(checkpoint_file):\r
                with open(checkpoint_file, "r", encoding="utf-8") as f:\r
                    return json.load(f)\r
        except Exception as e:\r
            print(f"[Checkpoint] Load failed: {e}")\r
        return None\r
\r
    def _nodes_list(self) -> List[str]:\r
        return list(self.nodes.keys())\r
''',
'''    def _load_checkpoint(self, node_name: str, run_id: str) -> Optional[Dict[str, Any]]:\r
        """加载某一次运行在指定节点保存的检查点。\r
\r
        检查点文件名与保存逻辑严格对应：``<node_name>_<run_id>.json``。\r
        ``run_id`` 来自调用方的 ``GraphState``，不能从图节点列表推断，\r
        否则不同运行会互相错配且无法恢复。\r
        """\r
        try:\r
            import os\r
            import re\r
\r
            # 节点和运行 ID 都来自内部状态；仍限制为文件名安全字符，避免\r
            # 自定义 GraphState 把路径片段带进 checkpoint 目录。\r
            safe_part = re.compile(r"^[A-Za-z0-9_-]{1,128}$")\r
            if not safe_part.fullmatch(node_name) or not safe_part.fullmatch(run_id):\r
                print("[Checkpoint] Load skipped: invalid node name or run_id")\r
                return None\r
\r
            path = self._checkpoint_path or "checkpoints"\r
            checkpoint_file = os.path.join(path, f"{node_name}_{run_id}.json")\r
            if os.path.exists(checkpoint_file):\r
                with open(checkpoint_file, "r", encoding="utf-8") as f:\r
                    return json.load(f)\r
        except Exception as e:\r
            print(f"[Checkpoint] Load failed: {e}")\r
        return None\r
''')

# 2) Make lesson de-duplication + append a single critical section and atomically publish.
p = root / "src/my_agent/memory/store.py"
replace(p, 'import json\r\n', 'import json\r\nimport os\r\nimport threading\r\nimport uuid\r\n')
replace(p,
'''        self._ensure_seed_files()\r
        self._injected_memory: List[str] = []\r
''',
'''        self._ensure_seed_files()\r
        self._injected_memory: List[str] = []\r
        # Protect read -> de-duplicate -> write as one operation.  The lock is\r
        # intentionally per MemoryStore instance; the atomic replacement below\r
        # also guarantees readers never observe a partially-written markdown file.\r
        self._lessons_lock = threading.RLock()\r
''')
replace(p,
'''    def append_lesson(self, lesson: str) -> bool:\r
        lesson = lesson.strip()\r
        if not lesson:\r
            return False\r
        lessons = self.load_lessons()\r
        if lesson in lessons:\r
            return False\r
        with self.lessons_path.open("a", encoding="utf-8") as f:\r
            if not self.lessons_path.read_text(encoding="utf-8").endswith("\\n"):\r
                f.write("\\n")\r
            f.write(f"- {lesson}\\n")\r
        return True\r
''',
'''    def append_lesson(self, lesson: str) -> bool:\r
        """Append a unique lesson without exposing a partial file to readers.\r
\r
        A previous implementation checked for duplicates and then appended in\r
        separate operations, so concurrent threads could write the same lesson\r
        or interleave a newline.  Keep the whole read/modify/write sequence in\r
        one critical section and publish the completed file with ``os.replace``.\r
        """\r
        lesson = lesson.strip()\r
        if not lesson:\r
            return False\r
\r
        with self._lessons_lock:\r
            text = self.lessons_path.read_text(encoding="utf-8")\r
            existing = {\r
                line.strip()[2:].strip()\r
                for line in text.splitlines()\r
                if line.strip().startswith("- ")\r
            }\r
            if lesson in existing:\r
                return False\r
\r
            updated = text if text.endswith("\\n") else text + "\\n"\r
            updated += f"- {lesson}\\n"\r
            tmp_path = self.lessons_path.with_name(\r
                f".{self.lessons_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"\r
            )\r
            try:\r
                tmp_path.write_text(updated, encoding="utf-8")\r
                os.replace(tmp_path, self.lessons_path)\r
            finally:\r
                # ``os.replace`` removes the source path.  This covers failures\r
                # before replace without deleting the last known-good file.\r
                if tmp_path.exists():\r
                    tmp_path.unlink(missing_ok=True)\r
        return True\r
''')

# 3) Synchronize MCP request state, writes and shutdown notification.
p = root / "src/my_agent/mcp_client.py"
text = read(p)
start = text.index('class MCPClient:')
# preserve preamble and replace the entire simple client class to avoid partial locking gaps.
preamble = text[:start]
new_class = '''class MCPClient:\r
    """Small JSON-RPC-over-stdio MCP client.\r
\r
    Requests may be sent by several tool-execution threads.  Request IDs, the\r
    pending-response registry and stdin writes therefore have separate locks.\r
    """\r
\r
    _STOPPED = object()\r
\r
    def __init__(self, command: List[str]):\r
        self.command = command\r
        self.process: Optional[subprocess.Popen] = None\r
        self._request_id = 0\r
        self._pending_requests: Dict[int, queue.Queue] = {}\r
        self._state_lock = threading.RLock()\r
        self._write_lock = threading.Lock()\r
        self._reader_thread: Optional[threading.Thread] = None\r
        self._running = False\r
        self.server_info: Dict[str, Any] = {}\r
        self.protocol_version = "2024-11-05"\r
        self._tools_cache: Optional[List[Dict[str, Any]]] = None\r
\r
    def start(self) -> None:\r
        with self._state_lock:\r
            if self.process is not None:\r
                raise MCPError("MCP 客户端已启动")\r
            self.process = subprocess.Popen(\r
                self.command,\r
                stdin=subprocess.PIPE,\r
                stdout=subprocess.PIPE,\r
                stderr=subprocess.PIPE,\r
                text=True,\r
                bufsize=1,\r
            )\r
            self._running = True\r
            self._tools_cache = None\r
            self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)\r
            self._reader_thread.start()\r
        try:\r
            self._initialize()\r
        except Exception:\r
            self.stop()\r
            raise\r
\r
    def stop(self) -> None:\r
        """Stop the subprocess and fail waiters immediately instead of timing out."""\r
        with self._state_lock:\r
            self._running = False\r
            process = self.process\r
            self.process = None\r
            pending = list(self._pending_requests.values())\r
            self._pending_requests.clear()\r
            self._tools_cache = None\r
            reader = self._reader_thread\r
\r
        for response_queue in pending:\r
            response_queue.put(self._STOPPED)\r
\r
        if process:\r
            try:\r
                process.terminate()\r
                process.wait(timeout=5)\r
            except subprocess.TimeoutExpired:\r
                process.kill()\r
            except Exception as exc:\r
                logger.debug("停止 MCP 进程时出错: %s", exc)\r
\r
        if reader and reader is not threading.current_thread():\r
            reader.join(timeout=1)\r
\r
    def _initialize(self) -> None:\r
        response = self._send_request(\r
            "initialize",\r
            {\r
                "protocolVersion": self.protocol_version,\r
                "capabilities": {"tools": {}},\r
                "clientInfo": {"name": "my-agent-python", "version": "0.1.0"},\r
            },\r
        )\r
        self.server_info = response.get("serverInfo", {})\r
        capabilities = response.get("capabilities", {})\r
        if "tools" not in capabilities:\r
            raise MCPError("服务器不支持工具功能")\r
\r
    def list_tools(self) -> List[Dict[str, Any]]:\r
        with self._state_lock:\r
            if self._tools_cache is not None:\r
                return list(self._tools_cache)\r
        response = self._send_request("tools/list", {})\r
        tools = response.get("tools", [])\r
        with self._state_lock:\r
            self._tools_cache = list(tools)\r
            return list(self._tools_cache)\r
\r
    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:\r
        response = self._send_request("tools/call", {"name": name, "arguments": arguments})\r
        content = response.get("content", [])\r
        result = ""\r
        for item in content:\r
            if item.get("type") == "text":\r
                result += item.get("text", "")\r
            elif item.get("type") == "error":\r
                raise MCPError(f"工具错误:{item.get('text', '未知错误')}")\r
        return result or "（空结果）"\r
\r
    def _send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:\r
        with self._state_lock:\r
            process = self.process\r
            if not self._running or not process or not process.stdin:\r
                raise MCPError("MCP 客户端未启动")\r
            self._request_id += 1\r
            request_id = self._request_id\r
            response_queue: queue.Queue = queue.Queue(maxsize=1)\r
            self._pending_requests[request_id] = response_queue\r
\r
        request = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}\r
        try:\r
            # TextIOWrapper is not safe for simultaneous write/flush calls.\r
            with self._write_lock:\r
                process.stdin.write(json.dumps(request) + "\\n")\r
                process.stdin.flush()\r
        except Exception as exc:\r
            with self._state_lock:\r
                self._pending_requests.pop(request_id, None)\r
            raise MCPError(f"发送请求失败:{exc}") from exc\r
\r
        try:\r
            response = response_queue.get(timeout=30)\r
        except queue.Empty as exc:\r
            with self._state_lock:\r
                self._pending_requests.pop(request_id, None)\r
            raise MCPError(f"请求超时:{method}") from exc\r
\r
        with self._state_lock:\r
            self._pending_requests.pop(request_id, None)\r
\r
        if response is self._STOPPED:\r
            raise MCPError("MCP 客户端已停止")\r
        if not isinstance(response, dict):\r
            raise MCPError("MCP 响应格式无效")\r
        if "error" in response:\r
            error = response["error"]\r
            raise MCPError(f"RPC 错误 [{method}]: {error.get('message', '未知错误')}")\r
\r
        return response.get("result", {})\r
\r
    def _read_loop(self) -> None:\r
        while True:\r
            with self._state_lock:\r
                running = self._running\r
                process = self.process\r
                stdout = process.stdout if process else None\r
            if not running or not stdout:\r
                break\r
            try:\r
                line = stdout.readline()\r
                if not line:\r
                    break\r
                line = line.strip()\r
                if not line:\r
                    continue\r
                try:\r
                    message = json.loads(line)\r
                except json.JSONDecodeError:\r
                    logger.warning("无法解析 MCP 消息")\r
                    continue\r
\r
                request_id = message.get("id")\r
                if request_id is not None:\r
                    with self._state_lock:\r
                        response_queue = self._pending_requests.get(request_id)\r
                    if response_queue is not None:\r
                        response_queue.put(message)\r
            except Exception as exc:\r
                with self._state_lock:\r
                    should_log = self._running\r
                if should_log:\r
                    logger.error("读取 MCP 响应时出错:%s", exc)\r
                break\r
\r
    def __enter__(self):\r
        self.start()\r
        return self\r
\r
    def __exit__(self, exc_type, exc_val, exc_tb):\r
        self.stop()\r
'''
write(p, preamble + new_class)

# 5) Filter literal path candidates after regex matching.
p = root / "src/my_agent/core/engine.py"
replace(p, 'from fnmatch import fnmatchcase\r\n', 'from fnmatch import fnmatchcase\r\nfrom pathlib import PurePosixPath\r\n')
replace(p,
'''def _explicit_workspace_file(query: str) -> Optional[str]:\r
    """Return a workspace-relative file path that appears literally in text."""\r
    match = _EXPLICIT_FILE_PATH_RE.search(query or "")\r
    return match.group(1).replace("\\\\", "/") if match else None\r
''',
'''def _explicit_workspace_file(query: str) -> Optional[str]:\r
    """Return a safe workspace-relative source path stated literally by the user.\r
\r
    The regular expression deliberately recognizes familiar source-file syntax,\r
    but matching alone is not a path-security decision.  Reject URL fragments,\r
    query/anchor syntax and traversal before a candidate can influence tool\r
    arguments.\r
    """\r
    text = query or ""\r
    for match in _EXPLICIT_FILE_PATH_RE.finditer(text):\r
        candidate = match.group(1).replace("\\\\", "/")\r
        start = match.start(1)\r
        prefix = text[max(0, start - 12):start].casefold()\r
        if prefix.endswith(("http://", "https://", "www.")):\r
            continue\r
        if any(marker in candidate for marker in ("://", "?", "#")):\r
            continue\r
        path = PurePosixPath(candidate)\r
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):\r
            continue\r
        return str(path)\r
    return None\r
''')

# 6) Construct enhanced modules locally; publish only on full success.
p = root / "src/my_agent/agent.py"
replace(p,
'''    def _init_enhanced_modules(self) -> None:\r
        self._router = DynamicRouter()\r
        self._persona_memory = PersonaMemory()\r
        self._persona_extractor = PersonaExtractor()\r
        self._category_rag = CategoryRAG(self._persona_memory)\r
        self._hallucination_detector = HallucinationDetector()\r
        self._citation_system = DeterministicCitation()\r
        self._multi_index = MultiIndexRetrieval()\r
        self._load_enhanced_state()\r
\r
    def _load_enhanced_state(self) -> None:\r
''',
'''    def _init_enhanced_modules(self) -> None:\r
        """Initialize enhanced services transactionally.\r
\r
        An optional module must be either fully usable or entirely disabled;\r
        publishing objects one by one left a half-initialized agent after a\r
        constructor/state-load failure.\r
        """\r
        try:\r
            router = DynamicRouter()\r
            persona_memory = PersonaMemory()\r
            persona_extractor = PersonaExtractor()\r
            category_rag = CategoryRAG(persona_memory)\r
            hallucination_detector = HallucinationDetector()\r
            citation_system = DeterministicCitation()\r
            multi_index = MultiIndexRetrieval()\r
            self._load_enhanced_state(\r
                persona_memory=persona_memory,\r
                multi_index=multi_index,\r
                citation_system=citation_system,\r
                strict=True,\r
            )\r
        except Exception as exc:\r
            self.enable_enhanced = False\r
            self._router = None\r
            self._persona_memory = None\r
            self._persona_extractor = None\r
            self._category_rag = None\r
            self._hallucination_detector = None\r
            self._citation_system = None\r
            self._multi_index = None\r
            self._debug(f"增强模块初始化失败，已回退到基础模式: {exc}")\r
            return\r
\r
        self._router = router\r
        self._persona_memory = persona_memory\r
        self._persona_extractor = persona_extractor\r
        self._category_rag = category_rag\r
        self._hallucination_detector = hallucination_detector\r
        self._citation_system = citation_system\r
        self._multi_index = multi_index\r
\r
    def _load_enhanced_state(\r
        self,\r
        *,\r
        persona_memory: Optional[Any] = None,\r
        multi_index: Optional[Any] = None,\r
        citation_system: Optional[Any] = None,\r
        strict: bool = False,\r
    ) -> None:\r
''')
replace(p,
'''                    self._persona_memory.add_fact(fact)\r
''',
'''                    target_persona_memory = persona_memory or self._persona_memory\r
                    if target_persona_memory is None:\r
                        raise RuntimeError("PersonaMemory 未初始化")\r
                    target_persona_memory.add_fact(fact)\r
''')
replace(p,
'''                    self._multi_index.add_document(doc)\r
''',
'''                    target_multi_index = multi_index or self._multi_index\r
                    if target_multi_index is None:\r
                        raise RuntimeError("MultiIndexRetrieval 未初始化")\r
                    target_multi_index.add_document(doc)\r
''')
replace(p,
'''                    self._citation_system.add_citation(cit)\r
        except Exception as e:\r
            self._debug(f"加载增强状态失败: {e}")\r
''',
'''                    target_citation_system = citation_system or self._citation_system\r
                    if target_citation_system is None:\r
                        raise RuntimeError("DeterministicCitation 未初始化")\r
                    target_citation_system.add_citation(cit)\r
        except Exception as e:\r
            if strict:\r
                raise\r
            self._debug(f"加载增强状态失败: {e}")\r
''')

# 7) Replace fragile blacklist with a deliberately tiny, non-composable diagnostics allowlist.
p = root / "src/my_agent/tools/builtins/shell.py"
text = read(p)
old_start = text.index('    # 危险命令模式（按类别分组）')
old_end = text.index('    def execute(self, params: Dict[str, Any]) -> str:', old_start)
new_policy = '''    # Arbitrary PowerShell is intentionally unavailable.  Even though this tool\r
    # is permission_level="deny", execute() can be called directly by code or a\r
    # future permission override; a substring blacklist is not a security boundary.\r
    # Keep only non-mutating, no-composition diagnostics.\r
    _SAFE_COMMANDS = frozenset({"get-date", "get-location", "get-process", "get-service"})\r
    _SAFE_COMMAND_RE = re.compile(\r
        r"^\\s*(get-date|get-location|get-process|get-service)\\s*$", re.IGNORECASE\r
    )\r
\r
    @classmethod\r
    def _validate_command(cls, command: str) -> Optional[str]:\r
        """Return a reason when *command* is outside the safe diagnostic subset."""\r
        if not command:\r
            return "未提供 PowerShell 命令"\r
        # Reject multiline input and every PowerShell composition/interpolation\r
        # primitive.  No aliases or command arguments are accepted, so an\r
        # allowlisted command cannot be redirected to a sensitive path.\r
        if any(token in command for token in ("\\r", "\\n", ";", "|", "&", "`", "$", "(", ")", "<", ">")):\r
            return "命令包含组合、重定向或动态执行语法"\r
        match = cls._SAFE_COMMAND_RE.fullmatch(command)\r
        if not match or match.group(1).casefold() not in cls._SAFE_COMMANDS:\r
            return "仅允许只读诊断命令: Get-Date、Get-Location、Get-Process、Get-Service"\r
        return None\r
\r
'''
text = text[:old_start] + new_policy + text[old_end:]
old_execute = '''    def execute(self, params: Dict[str, Any]) -> str:\r
        command = str(params.get("command", "")).strip()\r
        if not command:\r
            return "错误:未提供 PowerShell 命令"\r
\r
        cmd_lower = command.lower()\r
        for pattern, reason in self._BLOCK_PATTERNS.items():\r
            if pattern.lower() in cmd_lower:\r
                return (\r
                    f"错误:拒绝执行危险命令 '{pattern}'（{reason}）。"\r
                    "如需此操作，请明确说明并获得确认。"\r
                )\r
\r
        # 检查命令链接 + 可疑操作（避免 $ 字符误杀，PowerShell 变量极常见）\r
        if any(ch in command for ch in [';', '&', '`']) and any(\r
            p in cmd_lower for p in ['http', 'download', 'invoke', 'comobj']\r
        ):\r
            return "错误:拒绝执行包含可疑模式的复合命令。"\r
\r
        try:\r
'''
new_execute = '''    def execute(self, params: Dict[str, Any]) -> str:\r
        command = str(params.get("command", "")).strip()\r
        rejection = self._validate_command(command)\r
        if rejection:\r
            return f"错误:拒绝执行 PowerShell 命令（{rejection}）。"\r
\r
        try:\r
'''
if old_execute not in text:
    raise RuntimeError('shell execute prefix not found')
write(p, text.replace(old_execute, new_execute, 1))

# 10) Drop auxiliary system context (especially summary boundaries) before history.
p = root / "src/my_agent/core/context_assembler.py"
replace(p,
'''    规则 (engine._call_llm/_acall_llm 组装路径):\r
    - system 消息(含摘要边界)始终保留、置于最前\r
    - 历史消息按 旧→新 顺序保留,超预算时从最旧的非 system 消息开始成组丢弃\r
      (assistant 的 tool_calls 与其后的 tool 结果作为一组,避免孤儿 tool 消息)\r
    - 最新一条(当前问题)始终保留在最后\r
    - 目标利用率 40–60%: 以 target_ratio (默认 0.6) 为硬上限\r
''',
'''    规则 (engine._call_llm/_acall_llm 组装路径):\r
    - 首条规范 system prompt 始终保留、置于最前；摘要边界等辅助 system\r
      内容仅在预算允许时保留\r
    - 历史消息按 旧→新 顺序保留,超预算时从最旧的非 system 消息开始成组丢弃\r
      (assistant 的 tool_calls 与其后的 tool 结果作为一组,避免孤儿 tool 消息)\r
    - 最新一条(当前问题)始终保留在最后\r
    - ``target_ratio`` 是尽力达到的上限。若首条 system prompt 或当前问题\r
      单独已超限，函数保留它们并记录告警，而不会静默截断指令或用户输入。\r
''')
replace(p,
'''    system_msgs = [m for m in messages if m.role.value == "system"]\r
    history = [m for m in messages if m.role.value != "system"]\r
\r
    used = sum(_message_tokens(m) for m in system_msgs)\r
\r
    # 把历史按 "组" 切分: assistant(tool_calls) + 其 tool 结果 是一组\r
''',
'''    system_msgs = [m for m in messages if m.role.value == "system"]\r
    history = [m for m in messages if m.role.value != "system"]\r
\r
    # Keep the canonical prompt, but do not let a large compaction summary take\r
    # the whole budget before the current turn is considered.  SessionState puts\r
    # the canonical prompt first and marks summaries explicitly.\r
    primary_system: List["Message"] = []\r
    auxiliary_system: List["Message"] = []\r
    for message in system_msgs:\r
        if not primary_system and not getattr(message, "metadata", {}).get("summary_boundary"):\r
            primary_system.append(message)\r
        else:\r
            auxiliary_system.append(message)\r
\r
    used = sum(_message_tokens(m) for m in primary_system)\r
    kept_system = list(primary_system)\r
    for message in auxiliary_system:\r
        msg_tokens = _message_tokens(message)\r
        if used + msg_tokens <= budget:\r
            kept_system.append(message)\r
            used += msg_tokens\r
        else:\r
            logger.info("Dropping auxiliary system context to honor message budget")\r
\r
    # 把历史按 "组" 切分: assistant(tool_calls) + 其 tool 结果 是一组\r
''')
replace(p,
'''    result = system_msgs + kept\r
''',
'''    result = kept_system + kept\r
''')

print('Applied targeted fixes.')
