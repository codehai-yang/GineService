#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GINE 模型环境依赖检查脚本
用于验证部署环境是否满足模型运行要求
"""

import sys
import platform
from datetime import datetime


def print_section(title):
    """打印分隔标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_python_environment():
    """检查Python基础环境"""
    print_section("1. Python 基础环境")

    print(f"Python 版本:        {sys.version.split()[0]}")
    print(f"完整版本信息:       {sys.version}")
    print(f"平台:               {platform.platform()}")
    print(f"系统:               {platform.system()} {platform.release()}")
    print(f"架构:               {platform.architecture()[0]}")
    print(f"处理器:             {platform.processor()}")

    # 检查Python版本是否满足要求（建议3.8+）
    major, minor = sys.version_info[:2]
    if major == 3 and minor >= 8:
        print("✓ Python 版本符合要求 (≥3.8)")
    else:
        print("✗ 警告: Python 版本过低，建议使用 Python 3.8+")


def check_pytorch_environment():
    """检查PyTorch环境（最关键）"""
    print_section("2. PyTorch 核心环境")

    try:
        import torch

        print(f"✓ PyTorch 版本:     {torch.__version__}")
        print(f"  安装路径:         {torch.__file__}")

        # CUDA 支持
        cuda_available = torch.cuda.is_available()
        print(f"\n  CUDA 可用:        {cuda_available}")

        if cuda_available:
            print(f"  ✓ CUDA 版本:      {torch.version.cuda}")
            print(f"  ✓ cuDNN 版本:     {torch.backends.cudnn.version()}")
            print(f"  ✓ GPU 设备数:     {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\n  GPU {i} 详细信息:")
                print(f"    名称:           {props.name}")
                print(f"    显存:           {props.total_memory / 1024**3:.2f} GB")
                print(f"    计算能力:       {props.major}.{props.minor}")

            # CUDA 设备当前状态
            print(f"\n  当前CUDA设备:     {torch.cuda.current_device()}")
            print(f"  已分配显存:       {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"  缓存显存:         {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        else:
            print("  ⚠ 警告: CUDA 不可用，将使用CPU推理（速度较慢）")
            print("  提示: 如需GPU加速，请安装CUDA版本的PyTorch")

        # CPU线程设置
        print(f"\n  CPU线程数:        {torch.get_num_threads()}")
        print(f"  互操作线程数:     {torch.get_num_interop_threads()}")

    except ImportError as e:
        print(f"✗ PyTorch 未安装: {e}")
        return False

    return True


def check_torch_geometric():
    """检查PyTorch Geometric（GINE模型必需）"""
    print_section("3. PyTorch Geometric (图神经网络库)")

    try:
        import torch_geometric
        print(f"✓ PyG 版本:         {torch_geometric.__version__}")
        print(f"  安装路径:         {torch_geometric.__file__}")

        # 检查关键组件
        from torch_geometric.nn import GINEConv, global_add_pool
        print("✓ GINEConv 可用")
        print("✓ global_add_pool 可用")

        # 检查相关依赖
        try:
            import torch_scatter
            print(f"✓ torch-scatter:    {torch_scatter.__version__}")
        except ImportError:
            print("✗ torch-scatter 未安装（PyG依赖）")

        try:
            import torch_sparse
            print(f"✓ torch-sparse:     {torch_sparse.__version__}")
        except ImportError:
            print("✗ torch-sparse 未安装（PyG依赖）")

    except ImportError as e:
        print(f"✗ PyTorch Geometric 未安装: {e}")
        print("  安装命令: pip install torch-geometric")
        return False

    return True


def check_numpy():
    """检查NumPy"""
    print_section("4. NumPy (数值计算)")

    try:
        import numpy
        print(f"✓ NumPy 版本:       {numpy.__version__}")
        print(f"  安装路径:         {numpy.__file__}")

        # 测试基本功能
        test_array = numpy.array([1, 2, 3], dtype=numpy.float32)
        print(f"✓ float32 支持:     正常")
        print(f"✓ int32 支持:       正常")

    except ImportError as e:
        print(f"✗ NumPy 未安装: {e}")
        return False

    return True


def check_web_framework():
    """检查Web框架依赖"""
    print_section("5. Web 服务框架")

    packages = [
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('pydantic', 'Pydantic'),
    ]

    all_installed = True

    for import_name, display_name in packages:
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {display_name:20s} {version}")
        except ImportError:
            print(f"✗ {display_name:20s} 未安装")
            all_installed = False

    return all_installed


def check_model_config():
    """检查模型配置参数"""
    print_section("6. GINE 模型配置参数")

    try:
        import app.config.GlobalConfig as config

        print(f"节点特征维度:     {config.NODE_FEAT_DIM} (期望: 176)")
        print(f"边特征维度:       {config.EDGE_FEAT_DIM} (期望: 4)")
        print(f"节点数量:         {config.NUM_NODES} (期望: 175)")
        print(f"边数量:           {config.NUM_BRANCHES} (期望: 211)")
        print(f"隐藏层维度:       {config.HIDDEN_DIM} (期望: 64)")
        print(f"GINE层数:         {config.NUM_LAYERS} (期望: 3)")

        # 验证配置是否正确
        checks = [
            (config.NODE_FEAT_DIM == 176, "NODE_FEAT_DIM"),
            (config.EDGE_FEAT_DIM == 4, "EDGE_FEAT_DIM"),
            (config.NUM_NODES == 175, "NUM_NODES"),
            (config.NUM_BRANCHES == 211, "NUM_BRANCHES"),
            (config.HIDDEN_DIM == 64, "HIDDEN_DIM"),
            (config.NUM_LAYERS == 3, "NUM_LAYERS"),
        ]

        print("\n配置验证:")
        for check, name in checks:
            status = "✓" if check else "✗"
            print(f"  {status} {name}")

    except Exception as e:
        print(f"✗ 无法加载配置: {e}")
        return False

    return True


def check_model_file():
    """检查模型文件"""
    print_section("7. 模型文件检查")

    import os

    model_paths = [
        'best_model.pt',
        'F:\\office\\pythonProjects\\GINEModel\\Pt\\best_model.pt'
    ]

    model_found = False
    for path in model_paths:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"✓ 模型文件存在:     {path}")
            print(f"  文件大小:         {size / 1024:.2f} KB")
            model_found = True

            # 尝试加载模型
            try:
                import torch
                from app.ml.GINEClassifier import CostModelV2

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = CostModelV2()
                model.load_state_dict(torch.load(path, map_location=device))
                model.eval()
                print(f"✓ 模型加载成功")
                print(f"  设备:             {device}")

                # 统计模型参数
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"  总参数量:         {total_params:,}")
                print(f"  可训练参数:       {trainable_params:,}")

            except Exception as e:
                print(f"✗ 模型加载失败:     {e}")
            break

    if not model_found:
        print("✗ 模型文件未找到")
        print("  请将 best_model.pt 放在项目根目录或配置的路径中")

    return model_found


def generate_requirements():
    """生成requirements.txt"""
    print_section("8. Requirements 格式输出")
    print("# GINE 模型部署依赖")
    print(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Python 版本: {sys.version.split()[0]}")
    print()

    packages = [
        ('torch', 'torch'),
        ('torch_geometric', 'torch-geometric'),
        ('numpy', 'numpy'),
        ('fastapi', 'fastapi'),
        ('uvicorn', 'uvicorn'),
        ('pydantic', 'pydantic'),
    ]

    print("# 核心依赖")
    for import_name, pip_name in packages:
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'latest')
            print(f"{pip_name}=={version}")
        except ImportError:
            print(f"# {pip_name}  # 未安装")

    print("\n# PyG 相关依赖（版本需与PyTorch匹配）")
    print("# 请参考: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html")
    print("# 示例: pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-{torch_version}+{cuda_version}.html")


def check_cuda_compatibility():
    """检查CUDA兼容性"""
    print_section("9. CUDA 兼容性检查")

    try:
        import torch

        if not torch.cuda.is_available():
            print("⚠ CUDA 不可用")
            print("\n如需启用GPU加速，请:")
            print("1. 安装 NVIDIA 显卡驱动")
            print("2. 安装 CUDA Toolkit (推荐 11.8 或 12.1)")
            print("3. 安装对应版本的 PyTorch:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            return

        cuda_version = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()

        print(f"CUDA 版本:          {cuda_version}")
        print(f"cuDNN 版本:         {cudnn_version}")

        # 版本兼容性建议
        print("\n推荐的 PyTorch + CUDA 组合:")
        print("  - PyTorch 2.0.x + CUDA 11.8")
        print("  - PyTorch 2.1.x + CUDA 12.1")
        print("  - PyTorch 2.2.x + CUDA 12.1")

        print("\n环境变量检查:")
        import os
        cuda_home = os.environ.get('CUDA_HOME', '未设置')
        cuda_path = os.environ.get('CUDA_PATH', '未设置')
        print(f"  CUDA_HOME:          {cuda_home}")
        print(f"  CUDA_PATH:          {cuda_path}")

    except Exception as e:
        print(f"检查失败: {e}")


def main():
    """主函数"""
    print("\n" + "🔍 GINE 模型部署环境检查")
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = []

    # 执行各项检查
    check_python_environment()
    pytorch_ok = check_pytorch_environment()

    if pytorch_ok:
        pyg_ok = check_torch_geometric()
        numpy_ok = check_numpy()

    web_ok = check_web_framework()
    config_ok = check_model_config()
    model_ok = check_model_file()
    check_cuda_compatibility()
    generate_requirements()

    # 总结
    print_section("检查总结")

    if pytorch_ok:
        print("✓ PyTorch 环境正常")
    else:
        print("✗ PyTorch 环境异常")

    if pyg_ok:
        print("✓ PyTorch Geometric 正常")
    else:
        print("✗ PyTorch Geometric 缺失")

    if numpy_ok:
        print("✓ NumPy 正常")
    else:
        print("✗ NumPy 缺失")

    if web_ok:
        print("✓ Web 框架依赖完整")
    else:
        print("⚠ Web 框架依赖不完整")

    if config_ok:
        print("✓ 模型配置正确")
    else:
        print("✗ 模型配置异常")

    if model_ok:
        print("✓ 模型文件可用")
    else:
        print("⚠ 模型文件缺失或不可用")

    print("\n" + "=" * 70)
    print("✅ 环境检查完毕")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
