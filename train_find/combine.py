import pandas as pd
from pypinyin import lazy_pinyin
import re
import os

def clean_pinyin(text):
    """只保留字母并转为大写，用于强力匹配"""
    if pd.isna(text):
        return ""
    # 移除所有非字母字符（空格、^、_、数字等）
    return re.sub(r'[^a-zA-Z]', '', str(text)).upper()

def merge_results_v2(csv_path, xlsx_path, output_path):
    # 1. 读取数据
    # 指定 dtype={'Patient Name': str} 防止某些 ID 被误读为数字
    df_model = pd.read_csv(csv_path, dtype=str)
    df_human = pd.read_excel(xlsx_path)

    # 2. 数据清洗
    # 移除“检查结论”列（如果存在）
    if '检查结论' in df_human.columns:
        df_human = df_human.drop(columns=['检查结论'])
        print("已移除 '检查结论' 列")

    if 'File Path' in df_human.columns:
        df_human = df_human.drop(columns=['File Path'])
        print("已移除 'File Path' 列")

    # 3. 准备匹配主键
    # 处理人工表的姓名列：转拼音 -> 提取纯字母 -> 大写
    df_human['match_key'] = df_human['姓名'].apply(
        lambda x: "".join(lazy_pinyin(str(x)))
    ).apply(clean_pinyin)

    # 处理模型表的 Patient Name 列：提取纯字母 -> 大写
    # 这样可以解决 "LIU ZHANQUAN" 和 "LIUZHANQUAN" 不匹配的问题
    df_model['match_key'] = df_model['Patient Name'].apply(clean_pinyin)

    # 4. 执行合并 (Left Join)
    # 以人工表为主，把模型数据填进去
    merged_df = pd.merge(
        df_human, 
        df_model.drop(columns=['Patient Name']), # 避开重复的原始名称列
        on='match_key', 
        how='left'
    )

    # 5. 最后清理
    # 移除辅助匹配列
    merged_df = merged_df.drop(columns=['match_key'])
    
    # 确保没有重复行（如果一个人有多条记录，根据实际情况调整）
    # merged_df = merged_df.drop_duplicates()

    # 6. 保存结果
    merged_df.to_excel(output_path, index=False)
    print(f"合并成功！数据已保存至: {output_path}")
    
    # 检查匹配情况并打印
    matched_count = merged_df['Status'].notna().sum()
    print(f"匹配统计: 人工表共 {len(df_human)} 行，成功匹配模型数据 {matched_count} 行。")
# --- 使用示例 ---
if __name__ == "__main__":
    # 请替换为你实际的文件名
    csv_file = "/home/kevin/Code/ROI/validation_results.csv" 
    xlsx_file = "/media/kevin/3167F095163AC0C3/GFR/200例像素值-放射性计数拟合数据/医生测量的放射性计数/肾脏放射性计数.xlsx"
    output_file = "最终对比分析表.xlsx"

    if os.path.exists(csv_file) and os.path.exists(xlsx_file):
        merge_results_v2(csv_file, xlsx_file, output_file)
    else:
        print("请检查输入文件路径是否存在。")