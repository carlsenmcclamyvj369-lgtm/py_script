def load_param():
    param_cfg = {'train_data_set_path': r"C:\IPS\AI_NE\data\260402\rst\train_data_tnr_glb_gain_155.csv",
                 #train_data_me_155, train_data_nr_155, train_data_tnr_glb_gain_155
                 # 'test_data_set_path': r"E:\projection\1AI_PTH\AI_NE\0811\rst\train_data_me_155.xlsx",
                 'randomized_search_param_mode': 2,
                 'data_start_col': 1,       #1
                 'data_end_col': 155,       #155
                 'target_start_col': 156,   #156
                 'target_end_col': 156}     #156

    param_ai_ne = {'lgb_tree_num': 130,
                   'learning_rate_val': 0.1,
                   'test_size_ratio': 0.2,
                   'max_tree_depth': 69,
                   'num_leaves': 31,
                   # 'min_data_in_leaf': 20,
                   'random_seed': 618,
                   'early_stopping_rounds': 50,
                   'max_pred_val': 255,
                   'match_accuracy_tolerance': 3}

    param_visual = {'save_tree_plot_en': 0,
                    'plot_importance_max_num': 10,
                    'show_heatmap_en': 0}

    return param_cfg, param_ai_ne, param_visual
