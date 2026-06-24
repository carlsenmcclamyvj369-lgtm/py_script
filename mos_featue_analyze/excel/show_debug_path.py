from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import lightgbm as lgb


def show_label_predict(true_labels, pred_labels, title_name):
    unique_labels = sorted(set(true_labels + pred_labels))

    cm = confusion_matrix(true_labels, pred_labels)
    print(f"\n--- {title_name} ---")
    print(cm)
    # 计算并打印各项指标
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    total = tp + tn + fp + fn
    print(f"TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    if (tp + fp) > 0:
        print(f"Precision: {tp/(tp+fp):.4f}")
    if (tp + fn) > 0:
        print(f"Recall:    {tp/(tp+fn):.4f}")
    if total > 0:
        print(f"Accuracy:  {(tp+tn)/total:.4f}")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Class {i}' for i in unique_labels],
                yticklabels=[f'Class {i}' for i in unique_labels])
    plt.title(title_name)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # plt.savefig(f'{title_name}.png', dpi=300, bbox_inches='tight')  # 保存为PNG


def show_heatmap(data_path):
    test_df = pd.read_csv(data_path)
    data_heat = test_df.iloc[:, 1: -1].corr()

    plt.figure(figsize=(10, 8))
    # sns.clustermap(data_heat, standard_scale=1, method='average')
    sns.heatmap(data_heat, annot=None)


def show_feature_importance(X_train, model, plot_importance_max_num):
    feature_importances = model.feature_importances_
    feature_names = X_train.columns  # 确保X_train是DataFrame以自动获取特征名
    # 创建DataFrame并排序
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values(by='importance', ascending=False)

    print("特征重要性排行：")
    print(importance_df)
    lgb.plot_importance(model, max_num_features=plot_importance_max_num, importance_type='split')