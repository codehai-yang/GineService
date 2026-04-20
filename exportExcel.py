import pandas as pd
import sys

# ==================== 配置区 ====================
CSV_FILE = r"F:\office\idearProjects\project20251009\src\main\resources\branch_data.csv"  # 原始 CSV
OUTPUT_EXCEL = "湿区成本去inline.xlsx"  # 输出 Excel 文件
# ===============================================

HEADER_STR = "sample_id,point_id,sum_price,wet_cost"

def load_and_clean_csv(file_path):
    """
    读取包含重复表头的 CSV 文件，返回干净的 DataFrame。
    策略：逐行读取，丢弃与表头字符串完全相同的行。
    """
    clean_lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            # 跳过任何与表头完全一致的行（无论出现在文件何处）
            if line == HEADER_STR:
                continue
            clean_lines.append(line)

    if not clean_lines:
        raise ValueError("CSV 文件中没有有效数据行！")

    # 将清洗后的内容转换为 DataFrame
    from io import StringIO
    # 第一行需要手动加上表头，因为原始文件中的表头已被全部删除
    clean_csv = HEADER_STR + "\n" + "\n".join(clean_lines)
    df = pd.read_csv(StringIO(clean_csv))
    return df

def main():
    print("正在读取并清洗 CSV...")
    try:
        df = load_and_clean_csv(CSV_FILE)
    except Exception as e:
        print(f"读取失败: {e}")
        sys.exit(1)

    print(f"清洗后总行数: {len(df)}")
    print(f"样本数量: {df['sample_id'].nunique()}")

    # 数据类型转换（避免字符串干扰）
    df['sum_price'] = pd.to_numeric(df['sum_price'], errors='coerce')
    df['wet_cost'] = pd.to_numeric(df['wet_cost'], errors='coerce')
    df = df.dropna(subset=['sum_price', 'wet_cost'])

    # 去重（基于样本ID和分支点ID，保留第一次出现）
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['sample_id', 'point_id'], keep='first')
    after_dedup = len(df)
    print(f"去重移除行数: {before_dedup - after_dedup}")

    # 筛选湿区分支点（湿区成本 > 0）
    df_wet = df[df['wet_cost'] > 0].copy()
    print(f"湿区分支点数量: {len(df_wet)}")
    print(f"干区分支点数量: {len(df) - len(df_wet)}")

    if len(df_wet) == 0:
        print("没有湿区分支点数据，程序退出。")
        return

    # 排序，便于查看
    df_wet.sort_values(by=['sample_id', 'point_id'], inplace=True)

    # 导出到 Excel
    df_wet.to_excel(OUTPUT_EXCEL, index=False, engine='openpyxl')
    print(f"\n✅ 湿区分支点数据已成功导出到: {OUTPUT_EXCEL}")

    # 预览前 10 行
    print("\n预览导出的前 10 行：")
    print(df_wet.head(10))

if __name__ == "__main__":
    main()