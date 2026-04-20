import pandas as pd
import numpy as np
import sys

# ==================== 配置区 ====================
CSV_FILE = r"F:\office\idearProjects\project20251009\src\main\resources\branch_data.csv"  # Java 生成的 CSV 文件路径
MIN_PRICE = 0.01                      # 避免除零，单价总和小于此值的行将被过滤
SHOW_PLOTS = True                     # 是否显示图形（无 GUI 环境请设为 False）
# ===============================================

def main():
    # 1. 读取数据
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"错误：文件 '{CSV_FILE}' 未找到。请确认 Java 已成功导出数据。")
        sys.exit(1)

    print(f"成功读取 {len(df)} 行数据。")
    print(f"样本数量: {df['sample_id'].nunique()}")
    print(f"分支点总数: {len(df)}")

    # 2. 数据清洗：转换数据类型并过滤异常值
    df['sum_price'] = pd.to_numeric(df['sum_price'], errors='coerce')
    df['wet_cost'] = pd.to_numeric(df['wet_cost'], errors='coerce')
    df = df.dropna(subset=['sum_price', 'wet_cost'])

    # 3. 【核心改动】只保留湿区成本大于 0 的分支点
    df_wet = df[(df['sum_price'] > MIN_PRICE) & (df['wet_cost'] > 0)].copy()
    dry_count = len(df[(df['sum_price'] > MIN_PRICE) & (df['wet_cost'] == 0)])
    removed_by_price = len(df[df['sum_price'] <= MIN_PRICE])
    removed_by_wet_zero = len(df[(df['sum_price'] > MIN_PRICE) & (df['wet_cost'] == 0)])

    print(f"\n数据过滤统计：")
    print(f"  - 因单价总和 ≤ {MIN_PRICE} 过滤: {removed_by_price} 行")
    print(f"  - 因湿区成本 = 0 过滤（干区）: {dry_count} 行")
    print(f"  - 剩余湿区有效数据: {len(df_wet)} 行")

    if len(df_wet) == 0:
        print("无湿区有效数据，程序退出。")
        return

    # 4. 计算湿区专属比值 r = wet_cost / sum_price
    df_wet['ratio'] = df_wet['wet_cost'] / df_wet['sum_price']
    r = df_wet['ratio'].values

    # 5. 核心统计量
    N = len(r)
    mean_r = np.mean(r)
    std_r = np.std(r)
    cv = (std_r / mean_r) * 100 if mean_r != 0 else np.inf

    q1 = np.percentile(r, 25)
    median_r = np.median(r)
    q3 = np.percentile(r, 75)
    min_r = np.min(r)
    max_r = np.max(r)

    # 6. 输出统计结果
    print("\n" + "=" * 60)
    print("          湿区分支点比值统计 (wet_cost / sum_price)")
    print("=" * 60)
    print(f"有效样本数 (N)        : {N}")
    print(f"均值 (Mean)           : {mean_r:.6f}")
    print(f"标准差 (Std)          : {std_r:.6f}")
    print(f"变异系数 (CV)         : {cv:.2f}%")
    print(f"最小值 (Min)          : {min_r:.6f}")
    print(f"25% 分位数 (Q1)       : {q1:.6f}")
    print(f"中位数 (Median)       : {median_r:.6f}")
    print(f"75% 分位数 (Q3)       : {q3:.6f}")
    print(f"最大值 (Max)          : {max_r:.6f}")
    print("=" * 60)

    # 7. 给出湿区专属系数建议
    k_wet = 1 + mean_r
    print(f"\n建议湿区专属系数 k_wet = 1 + 均值 = {k_wet:.6f}")
    print("使用方式：")
    print("  - 若干区分支点（wet_cost == 0）: 总成本 = 单价总和")
    print("  - 湿区分支点（wet_cost > 0）  : 总成本 ≈ 单价总和 × k_wet")

    # 8. 根据变异系数评估湿区内部稳定性
    print("\n【湿区内部可靠性评估】")
    if cv < 10:
        print("✅ 变异系数 < 10%，湿区内部比值高度集中，单一系数非常可靠。")
    elif cv < 20:
        print("⚠️ 变异系数在 10%~20% 之间，湿区数据有一定波动，建议检查最大误差是否可接受。")
    elif cv < 30:
        print("⚠️ 变异系数在 20%~30% 之间，湿区内部差异较大，建议按湿区等级进一步分组。")
    else:
        print("❌ 变异系数 > 30%，湿区内部差异极大，单一系数误差大，强烈建议细分湿区等级。")

    # 9. 可选：绘制箱线图和直方图（仅展示湿区数据）
    if SHOW_PLOTS:
        try:
            import matplotlib.pyplot as plt
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # 箱线图
            ax1.boxplot(r, vert=True, patch_artist=True)
            ax1.set_title(f'湿区比值箱线图 (CV={cv:.1f}%)')
            ax1.set_ylabel('wet_cost / sum_price')
            ax1.grid(axis='y', linestyle='--', alpha=0.7)

            # 直方图
            ax2.hist(r, bins=30, edgecolor='black', alpha=0.7)
            ax2.axvline(mean_r, color='red', linestyle='dashed', linewidth=2, label=f'均值 = {mean_r:.4f}')
            ax2.axvline(median_r, color='green', linestyle='dashed', linewidth=2, label=f'中位数 = {median_r:.4f}')
            ax2.set_title('湿区比值直方图')
            ax2.set_xlabel('wet_cost / sum_price')
            ax2.set_ylabel('频数')
            ax2.legend()
            ax2.grid(axis='y', linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.show()
        except ImportError:
            print("\n提示：未安装 matplotlib，无法绘图。可执行 `pip install matplotlib` 后重试。")
        except Exception as e:
            print(f"\n绘图失败：{e}")

    # 10. 按样本分组检查湿区比值稳定性（仅输出前5个样本）
    print("\n【按样本分组的湿区比值均值预览（前5个样本）】")
    sample_stats = df_wet.groupby('sample_id')['ratio'].agg(['mean', 'std', 'count']).head()
    print(sample_stats)

if __name__ == "__main__":
    main()