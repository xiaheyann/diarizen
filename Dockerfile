# 基于已经包含 Poetry 的基础镜像
FROM registry.cn-hangzhou.aliyuncs.com/migo-dl/python:3.10.18-poetry-0-4-1-arm

# 设置工作目录
WORKDIR /app

# 拷贝 pyproject.toml 和 poetry.lock 文件
COPY . .

# 安装依赖
RUN apt-get update && apt-get install -y libsndfile1 && rm -rf /var/lib/apt/lists/* && poetry install

# 默认入口
CMD ["python", "-m", "diarizen.commands.grpc_app"]
