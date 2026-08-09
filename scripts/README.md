# scripts

本目录是本地诊断、接口验证和端到端复现工具，不是核心库。diagnose.py 最完整；verify_endpoints.py 做接口验证；check_db.py/check_data.py/check_obs.py/check_userid.py 检查数据和可观测性；feed_* 灌入请求；verify_fix* 和 quick_verify.py 做历史回归。脚本可能访问 .env、数据库和 8000 端口，执行前不要输出 API key。
