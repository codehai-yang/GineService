import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
import sys
import numpy as np

# ==================== 配置区 ====================
CSV_FILE = r"F:\office\idearProjects\project20251009\src\main\resources\branch_data.csv"  # 请修改为实际路径
HEADER_STR = "sample_id,point_id,sum_price,wet_cost"

# 是否仅显示湿区分支点（False 表示显示所有点，包括干区）
ONLY_WET = False

# 最多显示方案数（若超过此数量则随机抽样，设为 None 表示全部显示）
MAX_SAMPLES = 1200  # 你可以根据实际情况调整

# 是否对每个方案内部分支点按单价排序（连线更有意义）
SORT_BY_PRICE = True

# 输出 HTML 文件名
OUTPUT_HTML = "all_branch_points_interactive.html"

# 图表标题
CHART_TITLE = "多方案全部分支点成本对比（单价总和 vs 湿区成本）"
# =============================================

def load_clean_data(file_path):
    """清洗可能包含重复表头的 CSV，返回干净的 DataFrame"""
    clean_lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line == HEADER_STR:
                continue
            clean_lines.append(line)
    if not clean_lines:
        raise ValueError("CSV 文件中无有效数据！")
    clean_csv = HEADER_STR + "\n" + "\n".join(clean_lines)
    df = pd.read_csv(StringIO(clean_csv))
    # 转换数据类型
    df['sum_price'] = pd.to_numeric(df['sum_price'], errors='coerce')
    df['wet_cost'] = pd.to_numeric(df['wet_cost'], errors='coerce')
    df = df.dropna(subset=['sum_price', 'wet_cost'])
    # 去重（以防万一）
    df = df.drop_duplicates(subset=['sample_id', 'point_id'])
    return df

def main():
    print("正在加载并清洗数据...")
    try:
        df = load_clean_data(CSV_FILE)
    except Exception as e:
        print(f"数据加载失败: {e}")
        sys.exit(1)

    print(f"原始有效数据行数: {len(df)}")
    print(f"样本总数: {df['sample_id'].nunique()}")

    # 根据配置筛选湿区数据
    if ONLY_WET:
        df = df[df['wet_cost'] > 0].copy()
        print(f"筛选后湿区分支点数: {len(df)}")
        if len(df) == 0:
            print("没有湿区数据，程序退出。")
            return
    else:
        print("显示所有分支点（包含干区，湿区成本可为0）")

    # 确定要绘制的方案列表
    all_samples = sorted(df['sample_id'].unique())
    if MAX_SAMPLES is not None and len(all_samples) > MAX_SAMPLES:
        np.random.seed(42)  # 固定随机种子，保证结果可复现
        plot_samples = np.random.choice(all_samples, MAX_SAMPLES, replace=False)
        plot_samples = sorted(plot_samples)
        print(f"方案数过多（{len(all_samples)}），随机抽取 {MAX_SAMPLES} 个进行绘制。")
    else:
        plot_samples = all_samples

    print(f"即将绘制 {len(plot_samples)} 个方案。")

    # 生成足够多的颜色（结合多个 Plotly 定性色板，最多可区分约 60 种颜色）
    color_palette = (px.colors.qualitative.Plotly +
                     px.colors.qualitative.Light24 +
                     px.colors.qualitative.Dark24)
    line_styles = ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']

    # 创建图形对象
    fig = go.Figure()

    for i, sid in enumerate(plot_samples):
        sub_df = df[df['sample_id'] == sid].copy()
        if SORT_BY_PRICE:
            sub_df.sort_values('sum_price', inplace=True)

        # 分配颜色和线型，确保区分度
        color = color_palette[i % len(color_palette)]
        dash_style = line_styles[(i // len(color_palette)) % len(line_styles)]

        fig.add_trace(go.Scatter(
            x=sub_df['sum_price'],
            y=sub_df['wet_cost'],
            mode='lines+markers',
            name=f'方案 {sid}',
            line=dict(color=color, dash=dash_style, width=1.8),
            marker=dict(color=color, size=5),
            hovertemplate=(
                '<b>方案 %{fullData.name}</b><br>'
                '单价总和: %{x:.4f}<br>'
                '湿区成本: %{y:.4f}<br>'
                '<extra></extra>'
            )
        ))

    # 设置图表布局
    fig.update_layout(
        title=CHART_TITLE,
        xaxis_title='分支点回路单价总和 (sum_price)',
        yaxis_title='分支点湿区成本 (wet_cost)',
        hovermode='closest',
        template='plotly_white',
        width=1400,
        height=800,
        legend=dict(
            title='方案编号',
            itemsizing='constant',
            itemclick='toggleothers',   # 点击图例可单独显示/隐藏
            itemdoubleclick='toggle'
        )
    )

    # 显示图表（在 Jupyter 中会自动嵌入）
    fig.show()

    # 保存为 HTML 文件
    fig.write_html(OUTPUT_HTML)
    print(f"\n交互式图表已保存为: {OUTPUT_HTML}")

if __name__ == "__main__":
    main()