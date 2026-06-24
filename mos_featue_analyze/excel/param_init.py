import os

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_param():
    param_cfg = {
                 'train_data_set_path': os.path.join(SCRIPT_DIR, "all_data_norm.csv"),
                 'test_data_set_path': os.path.join(SCRIPT_DIR, "all_data_norm.csv"),
                 # 'train_data_set_path': os.path.join(SCRIPT_DIR, "train_data.csv"),
                 # 'test_data_set_path': os.path.join(SCRIPT_DIR, "test_data.csv"),
                 'randomized_search_param_mode': 2,
                 'data_start_col': 1,       # col 0 = id, cols 1-45 = features
                 'data_end_col': 45,
                 'target_start_col': 46,    # col 46 = label
                 'target_end_col': 46}

    param_ai_ne = {'lgb_tree_num': 130,
                   'learning_rate_val': 0.1,
                   'test_size_ratio': 0.2,
                   'max_tree_depth': 8,
                   'num_leaves': 31,
                   'random_seed': 618,
                   'early_stopping_rounds': 50,
                   'max_pred_val': 1,
                   'match_accuracy_tolerance': 0,
                   'reg_alpha': 0.1,
                   'reg_lambda': 1.0}

    param_visual = {'save_tree_plot_en': 0,
                    'plot_importance_max_num': 10,
                    'show_heatmap_en': 0}

    return param_cfg, param_ai_ne, param_visual
