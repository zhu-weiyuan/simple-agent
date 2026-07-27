#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.prod.py — 兼容 shim (P6)。

规范入口已迁移至 ``app_prod.py`` (文件名含点不能作为 python 模块名,
无法用于 ``uvicorn app.prod:app`` import string / 多 worker 启动)。
本文件仅保留向后兼容: ``python app.prod.py`` 仍可启动。
"""
from app_prod import *          # noqa: F401,F403
from app_prod import app        # noqa: F401  (explicit re-export)

if __name__ == "__main__":
    import os
    import uvicorn
    workers = int(os.environ.get("WORKERS", "1"))
    uvicorn.run("app_prod:app", host="0.0.0.0", port=8000, workers=workers)
