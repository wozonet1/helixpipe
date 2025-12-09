#!/bin/bash
set -e  # 遇到错误立即退出

# --- 配置 ---
ENV_NAME="helixpipe-gnn"
PYTHON_VERSION="3.9"

echo "🚀 Starting Environment Setup for: $ENV_NAME"

# 1. 检查并初始化 Conda (防止 Shell 没加载)
# 注意：这里假设你已经修复了 .zshrc
source ~/.zshrc || true
eval "$(conda shell.bash hook)" 

# 2. 重建 Conda 基础环境
echo "📦 [Step 1/4] Creating Conda Base Environment..."
# 为了彻底干净，如果存在则删除
conda env remove -n $ENV_NAME -y > /dev/null 2>&1 || true
# 创建新环境
conda env create -f conf/env_core.yaml

# 3. 激活环境
echo "🔌 Activating Environment..."
conda activate $ENV_NAME

# 4. 安装 PyG (最难搞的部分，单独处理)
# 针对 PyTorch 2.2.1 + CPU
echo "🧠 [Step 2/4] Installing PyTorch Geometric (CPU Version)..."
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch-geometric

# 5. 安装普通 Pip 依赖
echo "📚 [Step 3/4] Installing Common Pip Requirements..."
pip install -r requirements.txt

# 6. 挂载本地开发包 (Editable Mode)
echo "🔗 [Step 4/4] Linking Local Projects..."

# 6.2 HelixPipe 本身
echo "   -> Linking HelixPipe..."
pip install -e .

echo ""
echo "✅ Environment Setup Complete!"
echo "   To use: conda activate $ENV_NAME"