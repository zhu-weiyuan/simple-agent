"""Versioned prompts with SQLite persistence and variable validation."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable

from .memory.sqlite_store import PromptStore


@dataclass(frozen=True)
class PromptVersion:
    name: str
    version_no: str
    content: str

    @property
    def identifier(self) -> str:
        return f"{self.name}@{self.version_no}"


class PromptRegistry:
    _variable_pattern = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")

    def __init__(self) -> None:
        self._prompts: Dict[str, PromptVersion] = {}
        self._store: PromptStore | None = None

    def _get_store(self) -> PromptStore:
        if self._store is None:
            self._store = PromptStore(os.getenv("PROMPT_DB_PATH", "runtime/prompts.db"))
        return self._store

    def register(self, name: str, version_no: str, content: str) -> PromptVersion:
        prompt = PromptVersion(name, str(version_no), content)
        self._prompts[name] = prompt
        self._prompts[prompt.identifier] = prompt
        variables = {"required": sorted(set(self._variable_pattern.findall(content)))}
        self._get_store().add_prompt(name, prompt.version_no, content, variables, is_default=True)
        return prompt

    def get(self, name: str) -> PromptVersion:
        local = self._prompts.get(name)
        if local is not None:
            return local
        prompt_name, separator, version = name.partition("@")
        stored = self._get_store().get_prompt(prompt_name, version if separator else None)
        if stored:
            prompt = PromptVersion(stored["name"], stored["version_no"], stored["content"])
            self._prompts[prompt.identifier] = prompt
            if stored.get("is_default"):
                self._prompts[prompt.name] = prompt
            return prompt
        raise KeyError(f"Unknown prompt version: {name}")

    def render(self, prompt_name: str, **kwargs: object) -> str:
        prompt = self.get(prompt_name)
        required = set(self._variable_pattern.findall(prompt.content))
        missing = sorted(required - set(kwargs))
        if missing:
            raise ValueError(f"Prompt '{prompt.identifier}' missing required variables: {', '.join(missing)}")
        content = self._variable_pattern.sub(lambda m: "{" + m.group(1) + "}", prompt.content)
        try:
            return content.format(**kwargs)
        except KeyError as exc:
            raise ValueError(f"Prompt '{prompt.identifier}' missing required variable: {exc.args[0]}") from exc

    def versions(self) -> Iterable[PromptVersion]:
        stored = self._get_store().list_prompts()
        for item in stored:
            prompt = PromptVersion(item["name"], item["version_no"], item["content"])
            self._prompts.setdefault(prompt.identifier, prompt)
            if item.get("is_default"):
                self._prompts.setdefault(prompt.name, prompt)
        return {id(value): value for value in self._prompts.values()}.values()
