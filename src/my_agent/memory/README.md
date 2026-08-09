# memory

记忆抽象和持久化实现。store.py 是存储接口，sqlite_store.py 是 SQLite 实现，retrieval.py 是检索接口。先读接口，再读 SQLite，最后对照 test_sqlite_store.py 和 tests/test_persistence.py。
