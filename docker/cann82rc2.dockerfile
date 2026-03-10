# 基于已经包含 Poetry 的基础镜像
FROM registry.cn-hangzhou.aliyuncs.com/migo-dl/python:3.10.18-poetry-0-5-0-cann82rc2

# 设置工作目录
WORKDIR /app

# 拷贝必要的文件以安装依赖
COPY pyproject.toml poetry.lock README.md ./

# 安装依赖
RUN mkdir -p src/diarizen && \
    touch src/diarizen/__init__.py && \
    poetry install --no-root && \
    apt install libsndfile1 -y && \
    apt clean && rm -rf /var/lib/apt/lists/*

# 拷贝 pyproject.toml 和 poetry.lock 文件
COPY . .

# 安装依赖
RUN poetry install

# 暴露 gRPC 服务端口
EXPOSE 50051

# 默认入口
CMD ["/bin/bash", "-c", "\
    source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH} && \
    poetry run python -m diarizen.commands.grpc_app "]