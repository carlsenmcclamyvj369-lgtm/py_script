import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV, ParameterGrid
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from param_init import load_param
from show_debug_path import show_label_predict, show_heatmap, show_feature_importance
import os
import json

# 参数设置
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'


# 主函数
def main():
    # load param
    param_cfg, param_ai_ne, param_visual = load_param()

    # write
    train_data_set_path = param_cfg['train_data_set_path']
    train_data_dir = os.path.dirname(train_data_set_path)
    # test_data_set_path = param_cfg['test_data_set_path']
    test_data_set_path = train_data_set_path
    data_start_col = param_cfg['data_start_col']
    data_end_col = param_cfg['data_end_col']
    target_start_col = param_cfg['target_start_col']
    target_end_col = param_cfg['target_end_col']
    save_model_name = os.path.splitext(os.path.basename(train_data_set_path))[0] + "_ai_ne_model.txt"

    lgb_tree_num = param_ai_ne['lgb_tree_num']
    learning_rate_val = param_ai_ne['learning_rate_val']
    max_tree_depth = param_ai_ne['max_tree_depth']
    # min_data_in_leaf_num = param_ai_ne['min_data_in_leaf']
    num_leaves_val = param_ai_ne['num_leaves']
    test_size_ratio = param_ai_ne['test_size_ratio']
    random_seed = param_ai_ne['random_seed']
    max_pred_val = param_ai_ne['max_pred_val']
    match_accuracy_tolerance = param_ai_ne['match_accuracy_tolerance']
    early_stoppint_rounds_val = param_ai_ne['early_stopping_rounds']

    save_tree_plot_en = param_visual['save_tree_plot_en']
    plot_importance_max_num = param_visual['plot_importance_max_num']
    show_heatmap_en = param_visual['show_heatmap_en']

    # 1. 准备数据
    train_df = pd.read_csv(train_data_set_path)
    # train_df.describe().to_excel(os.path.join(train_data_dir, 'data_stats.xlsx'))
    train_df.columns = train_df.columns.str.replace(r'[^A-Za-z0-9_]', '_', regex=True)

    X = train_df.iloc[:, data_start_col: data_end_col + 1]
    y = train_df.iloc[:, target_start_col: target_end_col + 1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, stratify=y, random_state=random_seed)

    if param_cfg['randomized_search_param_mode'] == 0:
        model = lgb.LGBMRegressor(
            n_estimators=lgb_tree_num,
            learning_rate=learning_rate_val,
            max_depth=max_tree_depth,
            random_state=random_seed,
            num_leaves=num_leaves_val,
            # min_child_samples=min_data_in_leaf_num,
            metric=['mse', 'mae'],
            n_jobs=-1)

        model.fit(X_train,
                  y_train,
                  eval_set=[(X_test, y_test)],  # 提供验证集
                  eval_metric=['mse', 'mae'],  # 监控 MSE（也可以选 "rmse", "mae", "r2"）
                  callbacks=[lgb.early_stopping(stopping_rounds=early_stoppint_rounds_val)])  # 如果验证集 MSE 在 50 轮内没有提升，则停止训练
    elif param_cfg['randomized_search_param_mode'] == 1:
        fit_params = {
            'eval_metric': ['mse', 'R2'],
            'eval_set': [(X_test, y_test)],
            'eval_names': ['valid'],
            'categorical_feature': 'auto'
        }

        params_test = {
            'learning_rate': [0.1, 0.01, 0.05],
            'n_estimators': np.arange(50, 110, 10),
            'max_depth': np.arange(2, 7),
            # 'min_child_samples': np.arange(5, 25, 5)
        }

        n_lr = len(params_test['learning_rate'])
        n_est = len(params_test['n_estimators'])
        n_depth = len(params_test['max_depth'])

        n_iter = min(1000, n_lr * n_est * n_depth)

        model = lgb.LGBMRegressor(
            random_state=random_seed,
            n_jobs=-1)

        grid_search = RandomizedSearchCV(estimator=model,
                                         param_distributions=params_test,
                                         n_iter=n_iter,
                                         scoring=['mse', 'mae'],
                                         refit='r2',
                                         random_state=random_seed,
                                         verbose=True,
                                         n_jobs=-1)

        grid_search.fit(X_train, y_train, **fit_params, callbacks=[lgb.early_stopping(stopping_rounds=early_stoppint_rounds_val)])

        print("最优参数：\n", grid_search.best_params_)
        print("最优得分：\n", grid_search.best_score_)
        print("最优模型：\n", grid_search.best_estimator_)

        model = grid_search.best_estimator_
    else:
        param_grid = {
            'learning_rate': [0.05, 0.1],
            # 'min_data_in_leaf': np.arange(5, 25, 5),
            'num_leaves': [31, 63],
            'n_estimators': np.arange(100, 160, 10),
            'max_depth': np.arange(6, 11),
        }

        alpha1 = 0.6
        alpha2 = 1 - alpha1

        with open("supervise.txt", 'w') as f:
            f.write("")

        # 手动遍历所有参数组合
        for params in ParameterGrid(param_grid):
            current_params = {
                'learning_rate': params['learning_rate'],
                'n_estimators': params['n_estimators'],
                'num_leaves': params['num_leaves'],
                'max_depth': params['max_depth'],
                # 'min_child_samples': params['min_data_in_leaf'],
                'random_state': random_seed
            }
            print(params)

            # 训练模型
            model = lgb.LGBMRegressor(**current_params)
            model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)],
                      eval_metric=['mse', 'mae'],
                      callbacks=[lgb.early_stopping(stopping_rounds=early_stoppint_rounds_val)])

            # 计算训练集和测试集R2
            y_pred_train = model.predict(X_train)
            y_pred_train = y_pred_train.round().clip(0, max_pred_val)
            r2_train = r2_score(y_train, y_pred_train)
            mse_train = mean_squared_error(y_train, y_pred_train)
            rmse_train = np.sqrt(mse_train)
            mae_train = mean_absolute_error(y_train, y_pred_train)

            y_pred_test = model.predict(X_test)
            y_pred_test = y_pred_test.round().clip(0, max_pred_val)
            r2_test = r2_score(y_test, y_pred_test)
            mse_test = mean_squared_error(y_test, y_pred_test)
            rmse_test = np.sqrt(mse_test)
            mae_test = mean_absolute_error(y_test, y_pred_test)

            rule_train = alpha1 * mse_train + alpha2 * mae_train
            rule_test = alpha1 * mse_test + alpha2 * mae_test

            rule_diff = abs(rule_train - rule_test)
            rule_tmp = max(rule_train, rule_test)

            with open("supervise.txt", "a", encoding="utf-8") as f:
                f.write(f"参数:{current_params}, 训练集:{rule_train:.4f}, 测试集:{rule_test:.4f}, 下限值:{rule_tmp:.4f}, 差异:{rule_diff:.4f}\n")

        # 读取supervise.txt，按照下限值排序后，写入supervise_sort.txt
        best_params = sort_supervise_by_lower_bound('asc')

        print("\n最优参数组合:")
        print(best_params)

        current_params = {
            'learning_rate': best_params['learning_rate'],
            'n_estimators': best_params['n_estimators'],
            'num_leaves': best_params['num_leaves'],
            'max_depth': best_params['max_depth'],
            # 'min_child_samples': params['min_data_in_leaf'],
            'random_state': random_seed
        }

        print("\n最优参数组合跑代码:")
        model = lgb.LGBMRegressor(**current_params)
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  eval_metric=['mse', 'mae'],
                  callbacks=[lgb.early_stopping(stopping_rounds=early_stoppint_rounds_val)])

    # 保存模型
    model.booster_.save_model(save_model_name)
    print("Save Light GBM Model Successfully!")

    # 4. 可视化(可选)
    if save_tree_plot_en:
        os.makedirs('plots', exist_ok=True)

        for tree_idx in range(model.booster_.num_trees()):
            print(tree_idx)
            lgb.plot_tree(model, tree_index=tree_idx, figsize=(20, 10))
            plt.savefig(f'plots/lgbm_tree_{tree_idx}.png', dpi=300, bbox_inches='tight')  # 保存为PNG
            plt.close()  # 关闭图形，避免显示

    # 5. 特征重要性分析
    show_feature_importance(X_train, model, plot_importance_max_num)

    # 训练集结果
    y_pred_train = model.predict(X_train)
    y_pred_train = y_pred_train.round().clip(0, max_pred_val)
    mse = mean_squared_error(y_train, y_pred_train)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_train, y_pred_train)
    r2 = r2_score(y_train, y_pred_train)

    match_percentage = np.mean(np.abs(y_train.values.flatten() - y_pred_train) <= match_accuracy_tolerance) * 100

    print(f"训练集 MSE: {mse:.4f}")
    print(f"训练集 RMSE: {rmse:.4f}")
    print(f"训练集 MAE: {mae:.4f}")
    print(f"训练集 R²: {r2:.4f}")
    print(f"训练集 准确率: {match_percentage:.2f}%")
    print("\n")

    y_train_list = y_train.values.flatten().tolist()
    y_pred_train_list = y_pred_train.tolist()
    show_label_predict(y_train_list, y_pred_train_list, 'Train Confusion Matrix')

    # 测试集预测
    y_pred_test = model.predict(X_test)
    y_pred_test = y_pred_test.round().clip(0, max_pred_val)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    match_percentage = np.mean(np.abs(y_test.values.flatten() - y_pred_test) <= match_accuracy_tolerance) * 100

    print(f"测试集 MSE: {mse:.4f}")
    print(f"测试集 RMSE: {rmse:.4f}")
    print(f"测试集 MAE: {mae:.4f}")
    print(f"测试集 R²: {r2:.4f}")
    print(f"测试集 准确率: {match_percentage:.2f}%")

    y_test_list = y_test.values.flatten().tolist()
    y_pred_test_list = y_pred_test.tolist()
    show_label_predict(y_test_list, y_pred_test_list, 'Test Confusion Matrix')

    # 7. 对所有数据进行预测
    print("正在对所有数据集进行预测...")
    test_df = pd.read_csv(test_data_set_path)
    test_df.columns = test_df.columns.str.replace(r'[^A-Za-z0-9_]', '_', regex=True)

    true_labels = test_df['pred']
    true_labels = true_labels.tolist()
    X_new = test_df.iloc[:, data_start_col:data_end_col + 1]

    y_new_pred = model.predict(X_new)
    y_new_pred = y_new_pred.round()
    y_new_pred = [int(x) for x in y_new_pred]
    test_df['y_pred'] = y_new_pred
    input_name = os.path.splitext(os.path.basename(test_data_set_path))[0]
    prediction_csv_path = os.path.join(train_data_dir, input_name + "_prediction.csv")
    test_df.to_csv(prediction_csv_path, index=False)
    compare_pred_columns(prediction_csv_path, input_name)

    mse = mean_squared_error(true_labels, y_new_pred)
    r2 = r2_score(true_labels, y_new_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_labels, y_new_pred)

    match_percentage = np.mean(np.abs(np.array(true_labels) - np.array(y_new_pred)) <= match_accuracy_tolerance) * 100

    print(f"所有数据集 MSE: {mse:.4f}")
    print(f"所有数据集 RMSE: {rmse:.4f}")
    print(f"所有数据集 MAE: {mae:.4f}")
    print(f"所有数据集 R²: {r2:.4f}")
    print(f"所有数据集 准确率: {match_percentage:.2f}%")

    show_label_predict(true_labels, y_new_pred, 'Confusion Matrix')

    # 保存混淆矩阵图片
    cm_path = os.path.join(train_data_dir, os.path.basename(test_data_set_path).replace('.csv', '_confusion_matrix.png'))
    plt.savefig(cm_path, bbox_inches='tight', dpi=300)
    print(f'Confusion matrix saved to {cm_path}')

    # 热力图
    if show_heatmap_en:
        show_heatmap(test_data_set_path)

    # 保存param
    param_file_path = os.path.join(
        os.path.dirname(test_data_set_path),
        os.path.splitext(save_model_name)[0] + '_param.txt'
    )
    with open(param_file_path, 'w', encoding='utf-8') as f:
        json.dump({
            'param_cfg': param_cfg,
            'param_ai_ne': param_ai_ne
        }, f, indent=4, ensure_ascii=False)
    print(f'Parameters saved to {param_file_path}')

    # plt.show()
    plt.close()


def compare_pred_columns(input_file, input_name):
    if not os.path.exists(input_file):
        print(f"错误：文件 {input_file} 不存在")
        return

    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 查找pred和y_pred列
    if 'pred' not in df.columns or 'y_pred' not in df.columns:
        print("错误：找不到pred或y_pred列")
        return

    # 收集所有差异数据
    diff_data = []
    for idx, row in df.iterrows():
        pred_value = row['pred']
        y_pred_value = row['y_pred']

        if pred_value != y_pred_value:
            # 获取第一列的值（txt_name）
            txt_name = row.iloc[0]

            # 计算绝对值差异
            diff = int(abs(float(pred_value) - float(y_pred_value)))

            # 添加到差异数据列表
            diff_data.append((txt_name, pred_value, y_pred_value, diff, idx + 2))

    # 按差异值从大到小排序
    diff_data.sort(key=lambda x: x[3], reverse=True)

    # 创建新的CSV文件用于输出结果
    output_file = os.path.join(os.path.dirname(input_file), input_name + "_pred_false.csv")

    # 创建DataFrame并保存
    result_df = pd.DataFrame(diff_data, columns=["txt_name", "target", "pred", "diff", "row"])
    result_df.to_csv(output_file, index=False)

    # 保存修改后的原始CSV文件（不包含高亮，因为CSV不支持）
    df.to_csv(input_file, index=False)
    print(f"比较完成，差异已记录到 {output_file}")


def sort_supervise_by_lower_bound(sort_mode='asc'):
    """
    读取supervise.txt并按下限值排序后写入supervise_sort.txt
    sort_mode: 'asc'表示升序(小到大)，'desc'表示降序(大到小)
    返回: 差异值最小行的current_params
    """
    with open("supervise.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 解析每行的数据
    parsed_data = []
    for line in lines:
        if "下限值:" in line:
            lower_bound = float(line.split("下限值:")[1].split(",")[0])
            diff_value = float(line.split("差异:")[1].strip())
            params_str = line.split("参数:")[1].split(", 训练集:")[0]
            parsed_data.append({
                'lower_bound': lower_bound,
                'diff': diff_value,
                'params_str': params_str,
                'line': line
            })

    # 按指定方向排序
    reverse_sort = (sort_mode == 'desc')
    parsed_data.sort(key=lambda x: x['lower_bound'], reverse=reverse_sort)
    
    # 筛选前10%
    top_count = max(1, int(len(parsed_data) * 0.1))
    top_data = parsed_data[:top_count]
    
    # 对top_data按diff值升序排序并获取排序号
    top_data_sorted_by_diff = sorted(top_data, key=lambda x: x['diff'])
    for i, item in enumerate(top_data_sorted_by_diff):
        item['diff_sort_num'] = i

    # 对top_data按下限值按sort_mode排序并获取排序号
    reverse_sort = (sort_mode == 'desc')
    top_data_sorted_by_val = sorted(top_data, key=lambda x: x['lower_bound'], reverse=reverse_sort)
    for i, item in enumerate(top_data_sorted_by_val):
        item['val_sort_num'] = i

    # 计算final_sort_num = diff_sort_num + val_sort_num
    for item in top_data:
        item['final_sort_num'] = item['diff_sort_num'] + item['val_sort_num']

    # 将排序结果写入supervise_sort.txt
    with open("supervise_sort.txt", "w", encoding="utf-8") as f:
        for item in top_data:
            f.write(item['line'])
    
    # 找到final_sort_num最小的行，相同则选val_sort_num小的
    best_item = min(top_data, key=lambda x: (x['final_sort_num'], x['val_sort_num']))
    
    # 将参数字符串转换为字典
    params_str = best_item['params_str'].strip()
    params_dict = {}
    if params_str.startswith("{") and params_str.endswith("}"):
        try:
            params_dict = eval(params_str)
        except:
            params_dict = {'params_str': params_str}
    else:
        params_dict = {'params_str': params_str}
    
    return params_dict


if __name__ == '__main__':
    main()
