import os
import random
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset

from nni.algorithms.compression.pytorch.quantization.am_dorefa_quantizer import AmDoReFaQuantizer
from nni.algorithms.compression.pytorch.quantization.qat_quantizer import QAT_Quantizer
from nni.compression.pytorch.quantization.settings import set_quant_scheme_dtype

from CNN_DM.results.results_costdown_cmp.CNN_CD_no_BN.predict_cnn import COST_DOWN
from dm_cnn import MosquitoDenoiseCNN, features_list, NORM_DIV, MosquitoPatchDataset, load_data_dirs
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
GS = 8
COST_DOWN = True

def model_train():
    COST_DOWN = True
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    DATA_DIR = os.path.dirname(__file__) if '__file__' in dir() else '.'

    if GS == 8:
        print("GS 8 TRAIN:")
        dm_datasets = [
            # MosquitoPatchDataset(os.path.join(DATA_DIR, "9x9_dm.csv"), label=1),
            # MosquitoPatchDataset(os.path.join(DATA_DIR, "9x9_dm_merged.csv"), label=1),
            # MosquitoPatchDataset(os.path.join(DATA_DIR, "9x9_dm_SR_x3.csv"), label=1),
            # MosquitoPatchDataset(os.path.join(DATA_DIR, "9x9_dm_SR_4k_0707.csv"), label=1),
            # MosquitoPatchDataset(os.path.join(DATA_DIR, "9x9_dm_seq_0710.csv"), label=1),
            # MosquitoPatchDataset(os.path.join(DATA_DIR, "9x9_dm_test_data_append_0715.csv"), label=1),
            MosquitoPatchDataset(os.path.join(DATA_DIR, "grid_8_dm_9x9.csv"), label=1),

        ]
        not_dm_datasets = [
            # MosquitoPatchDataset(os.path.join(DATA_DIR, "9x9_not_dm.csv"), label=0),
            # MosquitoPatchDataset(os.path.join(DATA_DIR, "9x9_not_dm_merged.csv"), label=0),
            # MosquitoPatchDataset(os.path.join(DATA_DIR, "9x9_not_dm_SR_x3.csv"), label=0),
            # MosquitoPatchDataset(os.path.join(DATA_DIR, "9x9_not_dm_SR_x2_0707.csv"), label=0),
            # MosquitoPatchDataset(os.path.join(DATA_DIR, "9x9_not_dm_SR_4k_0707.csv"), label=0),
            # MosquitoPatchDataset(os.path.join(DATA_DIR, "9x9_not_dm_seq_0710.csv"), label=0),
            # MosquitoPatchDataset(os.path.join(DATA_DIR, "9x9_not_dm_test_data_append_0715.csv"), label=0),
            MosquitoPatchDataset(os.path.join(DATA_DIR, "grid_8_not_dm_9x9.csv"), label=0),
        ]
    else:
        print("GS 16 TRAIN:")
        # txt_file = "grid_16_dataset_paths.txt"
        # data_dirs = load_data_dirs(txt_file)
        # dm_datasets = []
        # not_dm_datasets = []
        #
        # for data_dir in data_dirs:
        #     dm_csv_path = os.path.join(data_dir, "grid_16_dm_9x9.csv")
        #     if os.path.exists(dm_csv_path):
        #         dm_datasets.append(MosquitoPatchDataset(dm_csv_path, label=1))
        #
        #     not_dm_csv_path = os.path.join(data_dir, "grid_16_not_dm_9x9.csv")
        #     if os.path.exists(not_dm_csv_path):
        #         not_dm_datasets.append(MosquitoPatchDataset(not_dm_csv_path, label=0))
        dm_datasets = [
            MosquitoPatchDataset(os.path.join(DATA_DIR, "grid_16_dm_9x9.csv"), label=1),
        ]
        not_dm_datasets = [
            MosquitoPatchDataset(os.path.join(DATA_DIR, "grid_16_not_dm_9x9.csv"), label=0),
        ]

    dm_dataset = ConcatDataset(dm_datasets)
    not_dm_dataset = ConcatDataset(not_dm_datasets)

    val_ratio = 0.2
    dm_size = len(dm_dataset)
    not_dm_size = len(not_dm_dataset)

    dm_val = int(dm_size * val_ratio)
    not_dm_val = int(not_dm_size * val_ratio)

    dm_indices = np.arange(dm_size)
    not_dm_indices = np.arange(not_dm_size)
    np.random.shuffle(dm_indices)
    np.random.shuffle(not_dm_indices)

    train_idx = list(dm_indices[dm_val:]) + [dm_size + i for i in not_dm_indices[not_dm_val:]]
    val_idx = list(dm_indices[:dm_val]) + [dm_size + i for i in not_dm_indices[:not_dm_val]]

    full_dataset = ConcatDataset([dm_dataset, not_dm_dataset])
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)

    print(f"Train: {len(train_dataset)} patches, Val: {len(val_dataset)} patches")

    # =========================
    # 6. 训练
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = MosquitoDenoiseCNN(cost_down=COST_DOWN).to(device)

    if COST_DOWN:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        max_grad_norm = 1.0
        label_smoothing = 0.05
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-5
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        max_grad_norm = None
        label_smoothing = 0.0
        scheduler = None
    criterion = nn.BCELoss()

    epochs = 400
    best_f1 = 0.0

    for epoch in range(epochs):
        # ─── Train ───
        model.train()
        total_loss = 0.0
        total_count = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            if label_smoothing > 0:
                y = y * (1 - label_smoothing) + label_smoothing / 2
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

        avg_loss = total_loss / total_count

        # ─── Validation ───
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                all_preds.append(pred.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        all_preds = np.concatenate(all_preds).flatten()
        all_labels = np.concatenate(all_labels).flatten()

        thresholds = [0.5]
        best_th = 0.5
        best_f1_epoch = 0.0
        best_cm = np.zeros((2, 2), dtype=np.int64)
        for th in thresholds:
            pb = (all_preds > th).astype(np.int64)
            tp = np.sum((pb == 1) & (all_labels == 1))
            tn = np.sum((pb == 0) & (all_labels == 0))
            fp = np.sum((pb == 1) & (all_labels == 0))
            fn = np.sum((pb == 0) & (all_labels == 1))
            prec = tp / (tp + fp + 1e-10)
            rec = tp / (tp + fn + 1e-10)
            f1_th = 2 * prec * rec / (prec + rec + 1e-10)
            if f1_th > best_f1_epoch:
                best_f1_epoch = f1_th
                best_th = th
                best_cm = np.array([[tn, fp], [fn, tp]])

        acc = (best_cm[0, 0] + best_cm[1, 1]) / best_cm.sum()
        prec = best_cm[1, 1] / (best_cm[1, 1] + best_cm[0, 1] + 1e-10)
        rec = best_cm[1, 1] / (best_cm[1, 1] + best_cm[1, 0] + 1e-10)

        print(f"Epoch [{epoch + 1}/{epochs}]  Loss: {avg_loss:.6f}  "
              f"Val Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {best_f1_epoch:.4f}  "
              f"th={best_th:.2f}")
        print(f"  Confusion Matrix:")
        print(f"    TN={best_cm[0, 0]:>5d}  FP={best_cm[0, 1]:>5d}")
        print(f"    FN={best_cm[1, 0]:>5d}  TP={best_cm[1, 1]:>5d}")

        if best_f1_epoch > best_f1:
            best_f1 = best_f1_epoch
            suffix = "_cost_down" if COST_DOWN else ""
            model_dir = os.path.join(DATA_DIR, "model")
            os.makedirs(model_dir, exist_ok=True)
            # torch.save(model.state_dict(), os.path.join(model_dir, f"mosquito_denoise_cnn{suffix}_grid_{GS}_qat.pth"))
            # np.save(f"best_th{suffix}.npy", np.array(best_th))
            torch.save(model, os.path.join(model_dir, f"mosquito_denoise_cnn_qat_grid_{GS}.pth"))

            print(f"  >>> Model saved (F1 improved to {best_f1_epoch:.4f} @ th={best_th:.2f})")

        if scheduler is not None:
            scheduler.step(best_f1_epoch)

        if scheduler is not None:
            scheduler.step(best_f1_epoch)

def main():
    # Two things should be kept in mind when set this configure_list:
    # 1. When deploying model on backend, some layers will be fused into one layer. For example, the consecutive
    # conv + bn + relu layers will be fused into one big layer. If we want to execute the big layer in quantization
    # mode, we should tell the backend the quantization information of the input, output, and the weight tensor of
    # the big layer, which correspond to conv's input, conv's weight and relu's output.
    # 2. Same tensor should be quantized only once. For example, if a tensor is the output of layer A and the input
    # of the layer B, you should configure either {'quant_types': ['input'], 'op_names': ['b']} or
    # {'quant_types': ['output'], 'op_names': ['a']} in the configure_list.

    #model_pth = "./model_save/"
    # model_save_path = "./model_save/"

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    SCRIPT_DIR = os.path.dirname(__file__)
    MODEL_PATH = os.path.join(SCRIPT_DIR, f"./model/mosquito_denoise_cnn_cost_down_grid_{GS}.pth")
    model_save_path = "./model/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### change
    #-----------------------------
    configure_list = [
        {
            'quant_types': ['weight', 'input'],
            'quant_bits': {'weight': 8, 'input': 8},
            'op_names': ['conv1']
        }, {
            'quant_types': ['output'],
            'quant_bits': {'output': 8},
            'op_names': ['relu1', 'relu2', 'relu3', 'sigmoid']
        },
        {
            'quant_types': ['weight'],
            'quant_bits': {'weight': 8},
            'op_names': ['conv2', 'conv3', 'conv4']
        }
    ]

    output_range_dict = {"relu1": [0, 1], "relu2": [0, 1], "relu3": [0, 1], "sigmoid": [0, 1]}

    ### replace with your own model
    model = MosquitoDenoiseCNN(C).to(device)
    # state = torch.load(os.path.join(model_pth, 'model.pth')) l
    # print(model)
    # print(state)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    # model.to(device)

    if COST_DOWN:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-6, weight_decay=1e-4)
        max_grad_norm = 1.0
        label_smoothing = 0.05
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-5
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        max_grad_norm = None
        label_smoothing = 0.0
        scheduler = None

    ### don't modify
    quantizer = AmDoReFaQuantizer(model, configure_list, optimizer, output_range_dict)
    quantizer.compress()
    # print(model)

    ### change
    model_train(model, device, epochs=10)  # 200

    # torch.save(model, os.path.join(model_save_path, f"mosquito_denoise_cnn_gat_grid_{GS}.pth"))


if __name__ == '__main__':
    main()
