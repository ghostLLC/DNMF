import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
import datetime
import os

# --- Configuration for Deep NMF ---
num_layers = 3  # Number of layers (L)
# Example structure: Adjust based on your data!
layer_components = [60, 30, 10] # [k1, k2, ..., kL]

# --- 1. 加载时间序列数据 ---
# !!! 用户需要修改这里的文件路径列表 !!!
# 假设你有三个文件，分别代表 t=1, t=2, t=3 的 90x90 矩阵
excel_file_paths = [
    'data/AD_P_1.xlsx',  # <--- 修改为时间点 1 的文件名
    'data/AD_P_2.xlsx',  # <--- 修改为时间点 2 的文件名
    'data/AD_P_3.xlsx'   # <--- 修改为时间点 3 的文件名
]
num_time_steps = len(excel_file_paths)

V_list = [] # 存储每个时间点的原始矩阵
print(f"加载 {num_time_steps} 个时间点的矩阵...")
for i, file_path in enumerate(excel_file_paths):
    try:
        df = pd.read_excel(file_path, header=None, index_col=None)
        V = df.to_numpy()

        print(f"\n - 从 '{file_path}' (时间点 t={i+1}) 加载数据。")
        print(f"   原始矩阵 V{i+1} shape: {V.shape}")

        # --- NMF 前置检查 ---
        if V.shape != (90, 90):
             raise ValueError(f"矩阵 V{i+1} 的 shape ({V.shape}) 不是预期的 (90, 90)。")
        if np.any(V < 0):
            print(f"   警告：V{i+1} 包含负值。替换为 0。")
            V = np.maximum(V, 0)
        if not np.issubdtype(V.dtype, np.number):
            print(f"   警告：V{i+1} 数据类型为 {V.dtype}。尝试转换为浮点数。")
            try:
                V = V.astype(float)
            except ValueError as e:
                raise TypeError(f"无法将 V{i+1} 的 Excel 数据转换为数值矩阵: {e}")

        V_list.append(V)

    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。")
        exit()
    except Exception as e:
        print(f"读取文件 '{file_path}' 时发生错误: {e}")
        exit()

print(f"\n成功加载 {len(V_list)} 个时间点的 90x90 矩阵。")

# --- 2. 对每个时间点的矩阵执行 Deep NMF 分解 ---

# 存储每个时间点的分解结果
W_factors_list = [] # List of lists: [[W1_t1, W2_t1,...], [W1_t2, W2_t2,...], ...]
H_final_list = []   # List of final H matrices: [H_L_t1, H_L_t2, H_L_t3]
base_random_state = 42 # Base random state for reproducibility

print("\n开始对每个时间点执行 Deep NMF 分解...")
for t in range(num_time_steps):
    print(f"\n===== 时间点 t={t+1} =====")
    V_t = V_list[t]
    W_list_t = []
    H_list_t = []
    current_input = V_t

    print(f"配置 Deep NMF 层级结构:")
    print(f" - 层数 (L): {num_layers}")
    print(f" - 每层组件数: {layer_components}")
    if len(layer_components) != num_layers:
        raise ValueError("`layer_components` 列表的长度必须等于 `num_layers`")
    # Optional: Add validation for layer_components based on V_t.shape here

    print(f"\n执行 Deep NMF 分解 for V{t+1}...")
    for l in range(num_layers):
        print(f"--- Layer {l+1}/{num_layers} (t={t+1}) ---")
        n_components_layer = layer_components[l]
        print(f" - 输入矩阵 shape: {current_input.shape}")
        print(f" - 本层组件数 (k{l+1}): {n_components_layer}")

        # 检查组件数
        if n_components_layer > current_input.shape[1]:
             raise ValueError(f"Layer {l+1} (t={t+1}) 的组件数 ({n_components_layer}) 不能大于输入特征数 ({current_input.shape[1]})。")
        if n_components_layer <= 0:
            raise ValueError(f"Layer {l+1} (t={t+1}) 的组件数必须 > 0。")

        model_layer = NMF(n_components=n_components_layer,
                          init='random',
                          solver='mu',
                          max_iter=700, # 可调整
                          random_state=base_random_state + l) # 保持种子一致性可能有助于比较 H

        W_l = model_layer.fit_transform(current_input)
        H_l = model_layer.components_

        W_list_t.append(W_l)
        H_list_t.append(H_l)

        print(f" - 输出 W{l+1} shape: {W_l.shape}")
        print(f" - 输出 H{l+1} shape: {H_l.shape}")
        print(f" - 本层重构误差: {model_layer.reconstruction_err_:.4f}")

        current_input = H_l

    W_factors_list.append(W_list_t) # Store Ws for this time step
    H_final_list.append(H_list_t[-1]) # Store final H for this time step
    print(f"\n时间点 t={t+1} Deep NMF 分解完成。")
    print(f"最终 H{num_layers} (t={t+1}) 矩阵 shape: {H_final_list[-1].shape}")


# --- 3. 预测下一个时间点的 H_L 矩阵 ---
print("\n===== 预测下一个时间点 (t=4) =====")
if num_time_steps < 2:
    print("错误：需要至少两个时间点的数据才能进行预测。")
    exit()

# 使用最后两个时间点的 H_L 进行简单的线性外插
# H_pred = H_t + (H_t - H_{t-1})
H_L_t_minus_1 = H_final_list[-2] # H_L at t=2
H_L_t = H_final_list[-1]       # H_L at t=3

# 可选：可以使用更复杂的预测模型，例如 ARIMA, LSTM 等，如果时间序列更长。
# 例如:
# from statsmodels.tsa.arima.model import ARIMA
# H_sequence = np.array([h.flatten() for h in H_final_list]) # Flatten H matrices
# # Fit ARIMA/LSTM model to H_sequence and predict the next step
# # This requires reshaping H and potentially fitting models element-wise or using vector ARIMA.

print(f"使用 H{num_layers}(t=3) 和 H{num_layers}(t=2) 进行线性外插预测 H{num_layers}(t=4)...")
delta_H = H_L_t - H_L_t_minus_1
H_L_pred = H_L_t + delta_H

# 重要：确保预测的 H 矩阵仍然是非负的
print("确保预测的 H 矩阵非负...")
H_L_pred = np.maximum(H_L_pred, 0)
print(f"预测得到的 H{num_layers}(t=4)_pred 矩阵 shape: {H_L_pred.shape}")


# --- 4. 使用 t=3 的 W 矩阵和预测的 H_L_pred 重构 V_pred ---
print("\n使用最后一个时间点 (t=3) 的 W 矩阵和预测的 H 进行重构...")

# 获取最后一个时间点 (t=3) 的 W 矩阵列表
W_factors_last = W_factors_list[-1] # [W1_t3, W2_t3, ..., WL_t3]

# 执行重构 V_pred ≈ W1_t3 * W2_t3 * ... * WL_t3 * H_L_pred
V_pred_reconstructed = W_factors_last[0]
print(f" - Step 0: Start with W1(t=3) ({V_pred_reconstructed.shape})")
for i in range(1, num_layers):
    print(f" - Step {i}: Multiply by W{i+1}(t=3) ({W_factors_last[i].shape})")
    V_pred_reconstructed = np.dot(V_pred_reconstructed, W_factors_last[i])
    print(f"   - Intermediate shape: {V_pred_reconstructed.shape}")

print(f" - Step {num_layers}: Multiply by H{num_layers}_pred ({H_L_pred.shape})")
V_pred_reconstructed = np.dot(V_pred_reconstructed, H_L_pred)
print(f" - 最终预测的 V_pred 矩阵 shape: {V_pred_reconstructed.shape}")

# 检查预测结果的 shape 是否为 (90, 90)
if V_pred_reconstructed.shape != (90, 90):
     print(f"警告：预测矩阵的最终 shape ({V_pred_reconstructed.shape}) 不是 (90, 90)。请检查 layer_components 设置。")


# --- 5. (可选) 保存预测结果 ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_pred_excel_path = f'predicted_matrix_t4_{timestamp}.xlsx'

print(f"\n准备将预测的 V_pred 矩阵保存到: '{output_pred_excel_path}'")
try:
    V_pred_df = pd.DataFrame(V_pred_reconstructed)
    V_pred_df.to_excel(output_pred_excel_path, index=False, header=False)
    print(f"预测矩阵成功保存到 '{output_pred_excel_path}'")
except Exception as e:
    print(f"\n保存预测矩阵到 Excel 文件时发生错误: {e}")


# --- 6. (可选) 同时保存分解结果 (W 和 H 矩阵) ---
output_decomp_excel_path = f'deep_nmf_temporal_results_{timestamp}.xlsx'
print(f"\n准备将所有时间点的分解结果保存到: '{output_decomp_excel_path}'")
try:
    with pd.ExcelWriter(output_decomp_excel_path, engine='openpyxl') as writer:
        for t in range(num_time_steps):
            # 保存 W 矩阵
            for l in range(num_layers):
                W_l_df = pd.DataFrame(W_factors_list[t][l])
                sheet_name = f'W{l+1}_t{t+1}'
                W_l_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
                print(f" - W{l+1}(t={t+1}) 写入 Sheet '{sheet_name}'")
            # 保存 H_L 矩阵
            H_L_df = pd.DataFrame(H_final_list[t])
            sheet_name_h = f'H{num_layers}_t{t+1}'
            H_L_df.to_excel(writer, sheet_name=sheet_name_h, index=False, header=False)
            print(f" - H{num_layers}(t={t+1}) 写入 Sheet '{sheet_name_h}'")
        # 保存预测的 H_L
        H_L_pred_df = pd.DataFrame(H_L_pred)
        sheet_name_h_pred = f'H{num_layers}_pred_t4'
        H_L_pred_df.to_excel(writer, sheet_name=sheet_name_h_pred, index=False, header=False)
        print(f" - 预测的 H{num_layers}(t=4)_pred 写入 Sheet '{sheet_name_h_pred}'")

    print(f"\n所有分解结果和预测的 H 矩阵成功保存到 '{output_decomp_excel_path}'")

except Exception as e:
    print(f"\n保存分解结果到 Excel 文件时发生错误: {e}")

print("\n链接预测流程完成。")