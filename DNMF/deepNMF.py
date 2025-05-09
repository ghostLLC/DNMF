import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
import datetime
import os # Import os for path joining if needed

# --- Configuration for Deep NMF ---
# Define the structure of the deep factorization
num_layers = 3  # Number of layers (L)
# List containing the number of components (k) for each layer [k1, k2, ..., kL]
# The number of components typically decreases or stays the same in deeper layers
# Example: V (m*n) -> W1 (m*k1), H1 (k1*n) -> W2 (k1*k2), H2 (k2*n) -> W3 (k2*k3), H3 (k3*n)
# Ensure k(l) <= k(l-1) or based on your specific model needs.
# Also k(l) must be <= features of the input to that layer's NMF.
# Example structure: 10 -> 5 -> 2 components
# Adjust these based on your V's dimensions and goals. Let's assume V has n features.
# layer_components = [10, 5, 2] # Example: Adjust k1, k2, k3
# Let's try to make it dependent on input V's dimensions later if needed
# For now, let's use placeholder values, ensure kL > 0 and kl <= features of H_{l-1}
layer_components = [60, 30, 10] # Example: User MUST adjust this based on data!

# --- 1. 从 Excel 文件加载数据 ---
# !!! 用户需要修改这里的文件路径 !!!
excel_file_path = 'data/AD_P_1.xlsx'  # <--- 修改为你实际的 Excel 文件名或完整路径

# 使用 pandas 读取 Excel 文件
try:
    # header=None: 输入文件没有表头行
    # index_col=None: 输入文件没有用作索引的列
    df = pd.read_excel(excel_file_path, header=None, index_col=None)

    # 将 DataFrame 转换为 NumPy 数组
    V = df.to_numpy()

    print(f"\n从 Excel 文件 '{excel_file_path}' 成功加载数据。")
    print("原始矩阵 V (shape: {}):\n{}".format(V.shape, V))

    # --- NMF/Deep NMF 前置检查：确保数据非负 ---
    if np.any(V < 0):
        print("\n警告：输入矩阵 V 包含负值。NMF 要求所有元素非负。")
        print("将负值替换为 0。")
        V = np.maximum(V, 0)

    # --- 检查数据是否都是数值类型 ---
    if not np.issubdtype(V.dtype, np.number):
        print(f"\n警告：加载的数据类型为 {V.dtype}。尝试转换为浮点数。")
        try:
            V = V.astype(float)
        except ValueError as e:
            raise TypeError(f"无法将 Excel 数据转换为数值矩阵: {e}")

except FileNotFoundError:
    print(f"错误：Excel 文件 '{excel_file_path}' 未找到。请检查路径和文件名。")
    exit()
except Exception as e:
    print(f"读取 Excel 文件时发生错误: {e}")
    exit()

# --- Dynamic Adjustment/Validation for layer_components ---
# Ensure components are valid based on V's shape
n_features = V.shape[1]
if layer_components[-1] <= 0:
     raise ValueError("The last layer must have at least 1 component.")
if layer_components[0] > min(V.shape):
    print(f"\n警告: 第一层的组件数 ({layer_components[0]}) 大于 V 的最小维度 ({min(V.shape)}). 可能导致问题或效率低下。建议减少组件数。")
# You might add more checks here, e.g., kl <= features of input to layer l

print(f"\n配置 Deep NMF 层级结构:")
print(f" - 层数 (L): {num_layers}")
print(f" - 每层组件数 (k1, k2, ..., kL): {layer_components}")
if len(layer_components) != num_layers:
    raise ValueError("`layer_components` 列表的长度必须等于 `num_layers`")


# --- 2. & 3. 执行 Deep NMF 分解 (逐层应用标准 NMF) ---
W_list = []  # 存储每一层的 W 矩阵
H_list = []  # 存储每一层的 H 矩阵 (主要用于传递给下一层)
current_input = V # 第一层的输入是 V
base_random_state = 42 # Base random state for reproducibility

print("\n开始执行 Deep NMF 分解...")
for l in range(num_layers):
    print(f"--- Layer {l+1}/{num_layers} ---")
    n_components_layer = layer_components[l]
    print(f" - 输入矩阵 shape: {current_input.shape}")
    print(f" - 本层组件数 (k{l+1}): {n_components_layer}")

    # 检查组件数是否有效
    if n_components_layer > current_input.shape[1]:
         raise ValueError(f"Layer {l+1} 的组件数 ({n_components_layer}) 不能大于其输入矩阵的特征数 ({current_input.shape[1]})。")
    if n_components_layer <= 0:
        raise ValueError(f"Layer {l+1} 的组件数必须大于 0。")

    # 创建并配置当前层的 NMF 模型
    # 使用不同的 random_state 种子（可选，或保持一致）
    model_layer = NMF(n_components=n_components_layer,
                      init='random', # 或者 'nndsvd' 等
                      solver='mu',    # 或者 'cd'
                      max_iter=500,   # 可以为不同层设置不同迭代次数
                      random_state=base_random_state + l) # Increment seed per layer

    # 对当前输入进行 NMF 分解
    # H_{l-1} ≈ W_l * H_l (其中 H_0 = V)
    W_l = model_layer.fit_transform(current_input)
    H_l = model_layer.components_

    # 存储结果
    W_list.append(W_l)
    H_list.append(H_l) # 存储 H_l 供下一层使用

    print(f" - 输出 W{l+1} shape: {W_l.shape}")
    print(f" - 输出 H{l+1} shape: {H_l.shape}")
    print(f" - 本层重构误差: {model_layer.reconstruction_err_:.4f}")

    # 更新下一层的输入
    current_input = H_l

print("\nDeep NMF 分解完成。")

# 最终的 H 矩阵是最后一个 H
H_final = H_list[-1]

# --- 4. 打印最终结果概述 ---
print("\n--- Deep NMF 最终结果 ---")
for i, W_l in enumerate(W_list):
    print(f"W{i+1} 矩阵 shape: {W_l.shape}")
print(f"H{num_layers} (Final H) 矩阵 shape: {H_final.shape}")


# --- 5. (可选) 重构原始矩阵并计算整体误差 ---
print("\n计算整体重构...")
V_reconstructed = W_list[0]
print(f" - Step 0: Start with W1 ({V_reconstructed.shape})")
# 逐层乘上 W 矩阵
for i in range(1, num_layers):
    print(f" - Step {i}: Multiply by W{i+1} ({W_list[i].shape})")
    V_reconstructed = np.dot(V_reconstructed, W_list[i])
    print(f"   - Intermediate shape: {V_reconstructed.shape}")

# 最后乘上 H_final
print(f" - Step {num_layers}: Multiply by H{num_layers} ({H_final.shape})")
V_reconstructed = np.dot(V_reconstructed, H_final)
print(f" - Final reconstructed V' shape: {V_reconstructed.shape}")

# 计算整体 Frobenius 范数误差
overall_reconstruction_err = np.linalg.norm(V - V_reconstructed, 'fro')
print("\nDeep NMF 整体重构误差 (Frobenius 范数 V vs W1*W2*...*WL*HL): {:.4f}".format(overall_reconstruction_err))


# --- 6. 将所有 W 和最终 H 矩阵保存到 Excel 文件 (纯数字) ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_excel_path = f'deep_nmf_results_{timestamp}.xlsx' # <--- 输出文件名

print(f"\n准备将纯数字结果保存到: '{output_excel_path}'")

try:
    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        # 保存每一层的 W 矩阵
        for i, W_l in enumerate(W_list):
            W_l_df = pd.DataFrame(W_l)
            sheet_name = f'W{i+1}_Matrix'
            W_l_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
            print(f" - W{i+1} 矩阵的纯数字已写入 Sheet '{sheet_name}'")

        # 保存最终的 H 矩阵 (H_L)
        H_final_df = pd.DataFrame(H_final)
        sheet_name_h = f'H{num_layers}_Matrix'
        H_final_df.to_excel(writer, sheet_name=sheet_name_h, index=False, header=False)
        print(f" - H{num_layers} (Final H) 矩阵的纯数字已写入 Sheet '{sheet_name_h}'")

    print(f"\n纯数字结果成功保存到 '{output_excel_path}'")

except Exception as e:
    print(f"\n保存结果到 Excel 文件时发生错误: {e}")