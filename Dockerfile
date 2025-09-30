# 基于已经包含 Poetry 的基础镜像
FROM registry.cn-hangzhou.aliyuncs.com/migo-dl/python:3.10-poetry-0-3-1

# 设置工作目录
WORKDIR /app

# 拷贝 pyproject.toml 和 poetry.lock 文件
COPY . .

# 安装依赖
RUN poetry install

# 默认入口
CMD ["python", "-m", "diarizen.commands.grpc_app"]
