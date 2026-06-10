import json
import os
import numpy as np

# ============================================================
# 预计算统计量加载
# ============================================================
_PARAMS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                            'normalization_params.json')


def _load_params():
    """加载预计算的标准化参数。"""
    with open(_PARAMS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_branch_feature(branch_feature, mean=None, std=None):
    """
    对边特征做归一化（分支长度标准化）。

    branch_feature [211, 4]：
        前3列：通断状态 one-hot（0或1），不归一化
        第4列：分支长度（连续值），标准化

    参数：
        branch_feature: numpy数组 [211, 4]
        mean:           预计算的分支长度均值，为None则动态计算
        std:            预计算的分支长度标准差，为None则动态计算

    返回：
        归一化后的 branch_feature [211, 4]
    """
    result = branch_feature.copy()

    length_col = result[:, 3]

    if mean is None:
        mean = length_col.mean()
    if std is None:
        std = length_col.std()

    if std > 0:
        result[:, 3] = (length_col - mean) / std
    else:
        result[:, 3] = 0.0

    return result


def normalize_price_matrix(circuit_cost, mean=None, std=None):
    """
    对节点特征矩阵的前175列（回路单价矩阵）做归一化。
    只对非零值做标准化，零（无回路）保持零。

    参数：
        circuit_cost: numpy数组 [175, 176]
        mean:          预计算的单价均值，为None则动态计算
        std:           预计算的单价标准差，为None则动态计算

    返回：
        归一化后的 circuit_cost [175, 176]
    """
    result = circuit_cost.copy()

    price_matrix = result[:, :175].copy()
    nonzero_mask = price_matrix != 0

    if nonzero_mask.sum() > 0:
        nonzero_vals = price_matrix[nonzero_mask]

        if mean is None:
            mean = nonzero_vals.mean()
        if std is None:
            std = nonzero_vals.std()

        if std > 0:
            price_matrix[nonzero_mask] = (nonzero_vals - mean) / std
        else:
            price_matrix[nonzero_mask] = 1.0

    result[:, :175] = price_matrix
    return result


def normalize_wet_cost(circuit_cost, mean=None, std=None):
    """
    对节点特征矩阵的第176列（湿区成本）做归一化。
    只对非零值做标准化，零（无湿区）保持零。

    参数：
        circuit_cost: numpy数组 [175, 176]
        mean:          预计算的湿区成本均值，为None则动态计算
        std:           预计算的湿区成本标准差，为None则动态计算

    返回：
        归一化后的 circuit_cost [175, 176]
    """
    result = circuit_cost.copy()

    wet_col = result[:, 175].copy()
    nonzero_mask = wet_col != 0

    if nonzero_mask.sum() > 0:
        nonzero_vals = wet_col[nonzero_mask]

        if mean is None:
            mean = nonzero_vals.mean()
        if std is None:
            std = nonzero_vals.std()

        if std > 0:
            wet_col[nonzero_mask] = (nonzero_vals - mean) / std
        else:
            wet_col[nonzero_mask] = 1.0

    result[:, 175] = wet_col
    return result


def normalize_input(branch_feature, circuit_cost):
    """
    使用预计算统计量对输入特征做标准化。

    标准化字段：
        - branch_feature 第4列（分支长度）
        - circuit_cost 前175列（回路单价矩阵）
        - circuit_cost 第176列（湿区成本）

    参数：
        branch_feature: numpy数组 [211, 4]
        circuit_cost:   numpy数组 [175, 176]

    返回：
        branch_feature_norm: 标准化后的边特征   [211, 4]
        circuit_cost_norm:   标准化后的节点特征  [175, 176]
    """
    params = _load_params()

    branch_feature_norm = normalize_branch_feature(
        branch_feature,
        mean=params['branch_length_mean'],
        std=params['branch_length_std']
    )

    circuit_cost_norm = normalize_price_matrix(
        circuit_cost,
        mean=params['price_mean'],
        std=params['price_std']
    )

    circuit_cost_norm = normalize_wet_cost(
        circuit_cost_norm,
        mean=params['wet_cost_mean'],
        std=params['wet_cost_std']
    )

    return branch_feature_norm, circuit_cost_norm



def denormalize_output(total_cost_norm):
    """
    将标准化后的模型输出反标准化回原始尺度。

    参数：
        total_cost_norm:   标准化后的总成本
        total_length_norm: 标准化后的总长度
        total_weight_norm: 标准化后的总重量

    返回：
        total_cost:   原始尺度的总成本
        total_length: 原始尺度的总长度
        total_weight: 原始尺度的总重量
    """
    params = _load_params()

    cost = total_cost_norm * params['total_cost_std'] + params['total_cost_mean']

    return cost