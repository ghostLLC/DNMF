import numpy as np
import pandas as pd  # 导入 pandas 库
from sklearn.decomposition import NMF
import datetime # 导入 datetime 库用于生成带时间戳的文件名

# --- 1. 从 Excel 文件加载数据 ---
# !!! 用户需要修改这里的文件路径 !!!
excel_file_path = 'data/AD_P_1.xlsx'  # <--- 修改为你实际的 Excel 文件名或完整路径

# 使用 pandas 读取 Excel 文件
# header=0 表示第一行是列名（特征名），可以根据你的 Excel 文件调整
# index_col=None 表示不使用 Excel 中的任何列作为 DataFrame 的索引
# 如果你的 Excel 文件第一列是样本标签而非数据，可设置 index_col=0
df = pd.read_excel(excel_file_path, header=None, index_col=None)

# 将 DataFrame 转换为 NumPy 数组
V = df.to_numpy()

print(f"\n从 Excel 文件 '{excel_file_path}' 成功加载数据。")
print("原始矩阵 V (shape: {}):\n{}".format(V.shape, V))

# --- NMF 前置检查：确保数据非负 ---
if np.any(V < 0):
    print("\n警告：输入矩阵 V 包含负值。NMF 要求所有元素非负。")
    print("将负值替换为 0。")
    V = np.maximum(V, 0)  # 将所有小于0的值设置为0
    # 或者你可以选择抛出错误：
    # raise ValueError("输入矩阵 V 包含负值，无法进行 NMF。请清理数据。")

# --- 检查数据是否都是数值类型 ---
if not np.issubdtype(V.dtype, np.number):
    print(f"\n警告：加载的数据类型为 {V.dtype}，可能包含非数值数据。")
    print("尝试将其转换为浮点数。如果失败，表示 Excel 中有文本或其他非数值内容。")
    try:
        V = V.astype(float)
    except ValueError as e:
        raise TypeError(f"无法将 Excel 中的数据转换为数值矩阵。请检查文件内容。错误: {e}")


# --- 2. 初始化并配置 NMF 模型 ---
# 设置分解后的低秩 k (即 n_components)
# k 是 W 的列数和 H 的行数
# k 的选择通常取决于应用场景和数据，需要小于 min(m, n)
n_components = 2 # 例如，我们想将数据降到 2 个隐含特征

# 创建 NMF 模型实例
# - n_components: 分解的成分数量 (k)
# - init: 初始化 W 和 H 的方法 ('random', 'nndsvd', 'nndsvda', 'nndsvdar')
#         'nndsvd' 系列通常效果更好，但 'random' 是最简单的。
# - solver: 优化求解器 ('cd' - 坐标下降, 'mu' - 乘法更新)
# - max_iter: 最大迭代次数
# - random_state: 用于随机初始化的种子，确保结果可复现
model = NMF(n_components=n_components,
            init='random',
            solver='mu', # 或者 'cd'
            max_iter=500,
            random_state=42)

# --- 3. 拟合并转换数据 ---
# 使用 fit_transform() 方法来拟合模型到数据 V 并返回 W 矩阵
# V ≈ W * H
# fit_transform(V) 返回 W (m x k)
W = model.fit_transform(V)

# H 矩阵 (k x n) 存储在模型的 components_ 属性中
H = model.components_

# --- 4. 打印结果 ---
print("\n分解得到的 W 矩阵 (shape: {}):\n{}".format(W.shape, W))
print("\n分解得到的 H 矩阵 (shape: {}):\n{}".format(H.shape, H))

# --- 5. (可选) 重构矩阵并计算误差 ---
V_reconstructed = np.dot(W, H)
print("\n重构的矩阵 V' = W * H (shape: {}):\n{}".format(V_reconstructed.shape, V_reconstructed))

# scikit-learn NMF 模型还提供了重构误差
reconstruction_err = model.reconstruction_err_
print("\nNMF 重构误差 (Frobenius 范数): {:.4f}".format(reconstruction_err))

# 你也可以手动计算 Frobenius 范数误差
# manual_reconstruction_err = np.linalg.norm(V - V_reconstructed, 'fro')
# print("手动计算的重构误差 (Frobenius 范数): {:.4f}".format(manual_reconstruction_err))

# --- 6. 将 W 和 H 矩阵保存到 Excel 文件 ---

# !!! 用户可以修改这里的文件名 !!!
# 为避免覆盖，可以添加时间戳
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_excel_path = f'nmf_results_{timestamp}.xlsx' # <--- 输出文件名

print(f"\n准备将纯数字结果保存到: '{output_excel_path}'")

# 先创建 DataFrame，因为 to_excel 是 DataFrame 的方法
W_df = pd.DataFrame(W)
H_df = pd.DataFrame(H)


# 使用 ExcelWriter 将多个 DataFrame 写入同一个文件的不同 Sheet
with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
    # 将 W 写入名为 'W_Matrix' 的 Sheet
    # index=False: 不写入行索引
    # header=False: 不写入列标题
    W_df.to_excel(writer, sheet_name='W_Matrix', index=False, header=False)
    print(f" - W 矩阵的纯数字已写入 Sheet 'W_Matrix'")

    # 将 H 写入名为 'H_Matrix' 的 Sheet
    # index=False: 不写入行索引
    # header=False: 不写入列标题
    H_df.to_excel(writer, sheet_name='H_Matrix', index=False, header=False)
    print(f" - H 矩阵的纯数字已写入 Sheet 'H_Matrix'")

print(f"\n纯数字结果成功保存到 '{output_excel_path}'")

