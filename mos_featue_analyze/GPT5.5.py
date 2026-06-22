import pandas as pd
import numpy as np


EPS = 1e-6


def get_context_stats(g, feature):
    """
    g: 9行DataFrame，对应3x3 block
    feature: 特征名
    第5行 index=4 是中心block，其余8个是邻域
    """
    center = float(g.iloc[4][feature])
    neigh = g.drop(g.index[4])[feature].astype(float).values

    return {
        f"c_{feature}": center,
        f"n_med_{feature}": float(np.median(neigh)),
        f"n_mean_{feature}": float(np.mean(neigh)),
        f"n_std_{feature}": float(np.std(neigh)),
        f"n_max_{feature}": float(np.max(neigh)),
        f"d_med_{feature}": center - float(np.median(neigh)),
        f"r_med_{feature}": center / (float(np.median(neigh)) + EPS),
        f"rank_{feature}": float(pd.Series(g[feature].astype(float).values).rank().iloc[4] / 9.0),
    }


def extract_one_sample_features(g):
    """
    输入9行，即一个3x3 block样本。
    输出中心块 + 上下文统计特征。
    """
    assert len(g) == 9, "每个样本必须是9行，对应3x3个8x8 block"

    feature_cols = [c for c in g.columns if c not in ["row", "col"]]

    out = {
        "center_row": int(g.iloc[4]["row"]),
        "center_col": int(g.iloc[4]["col"]),
    }

    for f in feature_cols:
        out.update(get_context_stats(g, f))

    return out


def rule_predict_one_sample(g, return_debug=True):
    """
    对一个3x3样本进行规则判断。
    g: 9行DataFrame
    return:
        label: 0/1
        level: 0/1/2/3
        score: 规则分数
        debug: 命中的条件
    """
    x = extract_one_sample_features(g)
    score = 0
    hit = []

    def add(cond, points, name):
        nonlocal score
        if cond:
            score += points
            hit.append((name, points))

    # ================================
    # 1. 中心块方差 / 复杂度
    # ================================
    add(x["c_max_var"] > 2500, 3, "center max_var very high")
    add(1500 < x["c_max_var"] <= 2500, 2, "center max_var high")
    add(800 < x["c_max_var"] <= 1500, 1, "center max_var medium")

    add(x["c_top5_var"] > 2200, 3, "center top5_var very high")
    add(1200 < x["c_top5_var"] <= 2200, 2, "center top5_var high")
    add(700 < x["c_top5_var"] <= 1200, 1, "center top5_var medium")

    add(x["c_high_var_count"] >= 30, 2, "center high_var_count high")
    add(15 <= x["c_high_var_count"] < 30, 1, "center high_var_count medium")

    add(x["c_very_high_var_count"] >= 10, 2, "center very_high_var_count high")
    add(2 <= x["c_very_high_var_count"] < 10, 1, "center very_high_var_count medium")

    # ================================
    # 2. 中心块高频 / 残差
    # ================================
    add(x["c_residual_max"] > 45, 3, "center residual_max very high")
    add(30 < x["c_residual_max"] <= 45, 2, "center residual_max high")
    add(20 < x["c_residual_max"] <= 30, 1, "center residual_max medium")

    add(x["c_lap_max"] > 180, 3, "center lap_max very high")
    add(120 < x["c_lap_max"] <= 180, 2, "center lap_max high")
    add(80 < x["c_lap_max"] <= 120, 1, "center lap_max medium")

    add(x["c_grad_max"] > 220, 3, "center grad_max very high")
    add(150 < x["c_grad_max"] <= 220, 2, "center grad_max high")
    add(100 < x["c_grad_max"] <= 150, 1, "center grad_max medium")

    add(x["c_edge_strength"] > 25, 2, "center edge_strength high")
    add(15 < x["c_edge_strength"] <= 25, 1, "center edge_strength medium")

    # ================================
    # 3. 中心块 second diff / ringing
    # ================================
    add(x["c_col_second_diff"] > 30, 3, "center col_second_diff very high")
    add(15 < x["c_col_second_diff"] <= 30, 2, "center col_second_diff high")
    add(8 < x["c_col_second_diff"] <= 15, 1, "center col_second_diff medium")

    add(x["c_row_second_diff"] > 35, 2, "center row_second_diff high")
    add(15 < x["c_row_second_diff"] <= 35, 1, "center row_second_diff medium")

    add(x["c_profile_ringing_max"] > 0.80, 3, "center profile_ringing_max very high")
    add(0.60 < x["c_profile_ringing_max"] <= 0.80, 2, "center profile_ringing_max high")
    add(0.40 < x["c_profile_ringing_max"] <= 0.60, 1, "center profile_ringing_max medium")

    add(x["c_col_ringing_max"] > 0.70, 3, "center col_ringing_max very high")
    add(0.50 < x["c_col_ringing_max"] <= 0.70, 2, "center col_ringing_max high")
    add(0.30 < x["c_col_ringing_max"] <= 0.50, 1, "center col_ringing_max medium")

    add(x["c_col_ringing_mean"] > 0.35, 2, "center col_ringing_mean high")
    add(0.20 < x["c_col_ringing_mean"] <= 0.35, 1, "center col_ringing_mean medium")

    add(x["c_col_ringing_d2_score"] > 0.25, 2, "center col_ringing_d2_score high")
    add(0.12 < x["c_col_ringing_d2_score"] <= 0.25, 1, "center col_ringing_d2_score medium")

    add(x["c_col_ringing_sign_score"] > 0.40, 2, "center col_ringing_sign_score high")
    add(0.20 < x["c_col_ringing_sign_score"] <= 0.40, 1, "center col_ringing_sign_score medium")

    # ================================
    # 4. 3x3上下文离散度
    # ================================
    add(x["n_std_col_second_diff"] > 12, 3, "neighbor std col_second_diff high")
    add(6 < x["n_std_col_second_diff"] <= 12, 1, "neighbor std col_second_diff medium")

    add(x["n_std_row_second_diff"] > 12, 2, "neighbor std row_second_diff high")
    add(6 < x["n_std_row_second_diff"] <= 12, 1, "neighbor std row_second_diff medium")

    add(x["n_std_residual_mean"] > 5, 3, "neighbor std residual_mean high")
    add(2.5 < x["n_std_residual_mean"] <= 5, 1, "neighbor std residual_mean medium")

    add(x["n_std_lap_mean"] > 15, 3, "neighbor std lap_mean high")
    add(8 < x["n_std_lap_mean"] <= 15, 1, "neighbor std lap_mean medium")

    add(x["n_std_col_ringing_d2_score"] > 0.15, 3, "neighbor std col_ringing_d2_score high")
    add(0.08 < x["n_std_col_ringing_d2_score"] <= 0.15, 1, "neighbor std col_ringing_d2_score medium")

    add(x["n_std_col_ringing_mean"] > 0.16, 2, "neighbor std col_ringing_mean high")
    add(0.08 < x["n_std_col_ringing_mean"] <= 0.16, 1, "neighbor std col_ringing_mean medium")

    add(x["n_std_median_var"] > 250, 2, "neighbor std median_var high")
    add(80 < x["n_std_median_var"] <= 250, 1, "neighbor std median_var medium")

    # ================================
    # 5. 强dm直接触发条件
    # ================================
    hard_dm = (
        (
            x["c_max_var"] > 2500
            and x["c_top5_var"] > 2000
            and x["c_grad_max"] > 180
        )
        or
        (
            x["c_profile_ringing_max"] > 0.75
            and x["c_col_ringing_max"] > 0.65
            and x["c_residual_max"] > 35
        )
        or
        (
            x["n_std_col_second_diff"] > 12
            and x["n_std_residual_mean"] > 5
            and x["n_std_col_ringing_d2_score"] > 0.15
        )
    )

    # ================================
    # 6. 强not_dm保护条件
    # ================================
    strong_not_dm = (
        x["c_max_var"] < 800
        and x["c_top5_var"] < 700
        and x["c_grad_max"] < 100
        and x["c_lap_max"] < 90
        and x["c_residual_max"] < 25
        and x["c_profile_ringing_max"] < 0.45
        and x["c_col_ringing_max"] < 0.40
    )

    # ================================
    # 7. score -> level
    # ================================
    if strong_not_dm:
        level = 0
        label = 0
        hit.append(("strong not_dm guard", -999))
    elif hard_dm:
        if score >= 24:
            level = 3
        else:
            level = 2
        label = 1
        hit.append(("hard dm trigger", 999))
    else:
        if score < 10:
            level = 0
        elif score < 16:
            level = 1
        elif score < 24:
            level = 2
        else:
            level = 3

        label = 1 if level >= 2 else 0

    if return_debug:
        return {
            "center_row": x["center_row"],
            "center_col": x["center_col"],
            "score": score,
            "level": level,
            "label": label,
            "hit_conditions": hit,
        }
    else:
        return label, level, score


def predict_csv_by_9rows(csv_path):
    """
    对整个CSV进行预测。
    每9行为一个样本。
    """
    df = pd.read_csv(csv_path)
    assert len(df) % 9 == 0, "CSV行数必须是9的整数倍"

    results = []

    for i in range(0, len(df), 9):
        g = df.iloc[i:i + 9].reset_index(drop=True)
        pred = rule_predict_one_sample(g, return_debug=True)
        pred["sample_id"] = i // 9
        results.append(pred)

    return pd.DataFrame(results)


if __name__ == "__main__":
    # 示例：预测一个文件
    result = predict_csv_by_9rows("input0012dm.csv")
    result = predict_csv_by_9rows("input0012_not_dm.csv")
    result = predict_csv_by_9rows("input0002_not_dm.csv")
    result = predict_csv_by_9rows("input0002_not_dm.csv")

    # 展示预测结果
    print(result[["sample_id", "center_row", "center_col", "score", "level", "label"]])

    # 如果想看某个中心点的命中条件，例如 row=59, col=46
    target = result[(result["center_row"] == 59) & (result["center_col"] == 46)]
    if len(target) > 0:
        print(target.iloc[0]["hit_conditions"])