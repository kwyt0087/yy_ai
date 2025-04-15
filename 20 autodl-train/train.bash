使用 huggingface 官方提供的 huggingface-cli 命令行工具。

1. 安装相关依赖
pip install -U huggingface_hub

2. 基本命令示例
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download --local-dir-use-symlinks False facebook/musicgen-small --local-dir musicgen-small