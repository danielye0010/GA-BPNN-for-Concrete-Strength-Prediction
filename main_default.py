import os
import random
import sys

import dill
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
# plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

OUTPUTS_DIR = "outputs_default"  # 输出路径
os.makedirs(OUTPUTS_DIR, exist_ok=True)  # 创建输出路径

FEATURE_SCALER_PATH = OUTPUTS_DIR + "/" + "feature_scaler.pkl"  # 特征缩放器路径
TARGET_SCALER_PATH = OUTPUTS_DIR + "/" + "target_scaler.pkl"  # 目标缩放器路径
MODEL_PATH = OUTPUTS_DIR + "/" + "model.pkl"  # 模型路径
R2_VISUALIZATION_PATH = OUTPUTS_DIR + "/" + "r2_visualization.png"  # R2可视化路径
R2_VISUALIZATION_CSV_PATH = OUTPUTS_DIR + "/" + "r2_visualization.csv"  # R2可视化csv路径
LOSS_VISUALIZATION_PATH = OUTPUTS_DIR + "/" + "loss_visualization.png"  # 损失可视化路径
LOSS_VISUALIZATION_CSV_PATH = OUTPUTS_DIR + "/" + "loss_visualization.csv"  # 损失可视化csv路径
RESIDUAL_VISUALIZATION_PATH = OUTPUTS_DIR + "/" + "residual_visualization.png"  # 残差可视化路径
COMPARISON_VISUALIZATION_PATH = OUTPUTS_DIR + "/" + "comparison_visualization.png"  # 对比可视化路径
EVALUATE_RESULT_PATH = OUTPUTS_DIR + "/" + "evaluate_result.txt"  # 评估结果路径
RESULT_VALUES_PATH = OUTPUTS_DIR + "/" + "result_values.csv"  # 结果值路径

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 优先使用gpu
print(f"DEVICE: {DEVICE}")  # 设备信息


class Datasets(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        assert len(self.X) == len(self.y)
        return len(self.y)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])


class Model(nn.Module):
    def __init__(self, inputs_size=4, hidden1_size=32, hidden2_size=32, outputs_size=1, dropout=0.2):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(inputs_size, hidden1_size)  # 输入层 -> 隐藏层
        self.linear2 = nn.Linear(hidden1_size, hidden2_size)  # 隐藏层 -> 隐藏层
        self.linear3 = nn.Linear(hidden2_size, outputs_size)  # 隐藏层 -> 输出层
        self.activation = nn.PReLU()  # 激活层
        self.dropout = nn.Dropout(dropout)  # dropout层

    def forward(self, inputs):  # torch.Size([B, 4])
        outputs = self.linear1(inputs)  # torch.Size([B, 32])
        outputs = self.activation(outputs)  # torch.Size([B, 32])

        outputs = self.dropout(outputs)  # torch.Size([B, 32])
        outputs = self.linear2(outputs)  # torch.Size([B, 32])
        outputs = self.activation(outputs)  # torch.Size([B, 32])

        outputs = self.dropout(outputs)  # torch.Size([B, 32])
        outputs = self.linear3(outputs)  # torch.Size([B, 1])

        return outputs


def setup_seed(seed=42):
    # 随机因子 保证在不同电脑上模型的复现性
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def save_pkl(filepath, data):
    # 保存模型
    with open(filepath, "wb") as fw:
        dill.dump(data, fw)
    print(f"[{filepath}] data saving...")


def load_pkl(filepath):
    # 加载模型
    with open(filepath, "rb") as fr:
        data = dill.load(fr, encoding="utf-8")
    print(f"[{filepath}] data loading...")
    return data


def save_txt(filepath, data):
    # 保存txt
    with open(filepath, "w", encoding="utf-8") as fw:
        fw.write(data)
    print(f"{filepath} saving...")


def standard_scaler(values, scaler_path, mode="train"):
    # 标准化
    if mode == "train":
        scaler = StandardScaler()  # 定义标准化模型
        scaler.fit(values)  # 训练
        save_pkl(scaler_path, scaler)  # 保存
    else:
        scaler = load_pkl(scaler_path)  # 加载模型
    return scaler.transform(values)  # 转换


def standard_scaler_inverse(values, scaler_path):
    # 反标准化
    scaler = load_pkl(scaler_path)  # 加载模型
    return scaler.inverse_transform(values)  # 转换


def load_data():
    # 加载数据
    data = pd.read_csv("D:/data.csv", header=None, names=["x1", "x2", "x3", "x4", "y"], usecols=[0, 1, 2, 3, 4])  # 读取数据

    X = data[["x1", "x2", "x3", "x4"]].values  # 获取特征
    y = data[["y"]].values  # 获取标签

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 划分数据集

    X_train = standard_scaler(np.array(X_train), FEATURE_SCALER_PATH, mode="train").tolist()  # 特征标准化 训练集训练
    X_test = standard_scaler(np.array(X_test), FEATURE_SCALER_PATH, mode="val").tolist()  # 特征标准化 测试集应用

    y_train = standard_scaler(np.array(y_train), TARGET_SCALER_PATH, mode="train").tolist()  # 目标值标准化 训练集训练
    y_test = standard_scaler(np.array(y_test), TARGET_SCALER_PATH, mode="val").tolist()  # 目标值标准化 测试集应用

    return X_train, X_test, y_train, y_test


def regression_evaluate(y_true, y_pred):
    # 回归模型的性能指标
    evaluate_result = ""
    evaluate_result += f"The evaluation indicator is:"
    evaluate_result += f"\nMAE: {round(mean_absolute_error(y_true, y_pred), 4)}"
    evaluate_result += f"\nRMSE: {round(pow(mean_squared_error(y_true, y_pred), 0.5), 4)}"
    evaluate_result += f"\nMAPE: {round(mean_absolute_percentage_error(y_true, y_pred), 4)}"
    evaluate_result += f"\nR2: {round(r2_score(y_true, y_pred), 4)}"
    print(evaluate_result)
    save_txt(EVALUATE_RESULT_PATH, evaluate_result)


def epoch_visualization(y1, y2, name, output_path):
    # epoch变化图
    plt.figure(figsize=(16, 9), dpi=100)  # 定义画布
    plt.plot(y1, marker="", linestyle="-", linewidth=2, label=f"Train {name}")  # 绘制曲线
    plt.plot(y2, marker="", linestyle="-", linewidth=2, label=f"Test {name}")  # 绘制曲线
    plt.title(f"{name} change map during training", fontsize=24)  # 标题
    plt.xlabel("Epoch", fontsize=20)  # x轴标签
    plt.ylabel(name, fontsize=20)  # y轴标签
    plt.tick_params(labelsize=16)  # 设置坐标轴轴刻度大小
    plt.legend(loc="best", prop={"size": 20})  # 图例
    plt.savefig(output_path)  # 保存图像
    # plt.show()  # 显示图像
    plt.close()  # 关闭图像


def residual_visualization(y_real, y_pred, output_path, fitting=False):
    # 绘制预测值和真实值的对比图
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9), dpi=100)  # 定义画布

    ax.text(
        min(y_real),
        max(y_real),
        f"$MAE={round(mean_absolute_error(y_real, y_pred), 4)}$"
        f"\n$RMSE={round(pow(mean_squared_error(y_real, y_pred), 0.5), 4)}$"
        f"\n$MAPE={round(mean_absolute_percentage_error(y_real, y_pred), 4)}$"
        f"\n$R^2={round(r2_score(y_real, y_pred), 4)}$",
        verticalalignment="top",
        fontdict={"size": 16, "color": "k"},
    )  # 左上角显示模型性能指标
    ax.scatter(y_real, y_pred, c="none", marker="o", edgecolors="k")  # 绘制散点图
    if fitting:
        from sklearn.linear_model import LinearRegression

        fitting_model = LinearRegression()
        fitting_model.fit([[item] for item in y_real], y_pred)
        ax.plot(
            [min(y_real), max(y_real)],
            [
                fitting_model.predict([[min(y_real)]]).item(),
                fitting_model.predict([[max(y_real)]]).item(),
            ],
            linewidth=2,
            linestyle="--",
            color="r",
            label="Fitting curve",
        )  # 拟合曲线
    ax.plot(
        [min(y_real), max(y_real)],
        [min(y_real), max(y_real)],
        linewidth=2,
        linestyle="-",
        color="r",
        label="Reference curve",
    )  # 参考曲线
    ax.set_title("Real value and predictive value residue diagram", fontsize=24)  # 标题
    ax.set_xlabel("Real values", fontsize=20)  # x轴标签
    ax.set_ylabel("Predict values", fontsize=20)  # y轴标签
    ax.tick_params(labelsize=16)  # 设置坐标轴轴刻度大小
    ax.legend(loc="lower right", prop={"size": 20})  # 图例

    plt.tight_layout()  # 防重叠
    plt.savefig(output_path)  # 保存图像
    # plt.show()  # 显示图像
    plt.close()  # 关闭图像


def comparison_visualization(y_real, y_pred, output_path):
    # 绘制预测值与真实值的直观对比图
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9), dpi=100)  # 定义画布

    ax.plot(y_real, marker="", linestyle="-", linewidth=2, label="Real values")  # 画真实值
    ax.plot(y_pred, marker="", linestyle="-", linewidth=2, label="Predict values")  # 画预测值
    ax.set_title("Comparison of real values and prediction values", fontsize=24)  # 标题
    ax.set_xlabel("Data dot", fontsize=20)  # x轴标签
    ax.set_ylabel("Values", fontsize=20)  # y轴标签
    ax.tick_params(labelsize=16)  # 设置坐标轴轴刻度大小
    ax.legend(loc="best", prop={"size": 20})  # 图例

    plt.tight_layout()  # 防重叠
    plt.savefig(output_path)  # 保存图像
    # plt.show()  # 显示图像
    plt.close()  # 关闭图像


def train_epoch(train_loader, model, optimizer, criterion, epoch, epochs):
    model.train()  # 训练模式
    real_sets = []  # 真实值
    pred_sets = []  # 预测值
    train_loss_records = []  # loss
    for idx, batch_data in enumerate(tqdm(train_loader, file=sys.stdout)):  # 遍历
        inputs, targets = batch_data  # 输入 输出

        outputs = model(inputs.to(DEVICE))  # 前向传播
        loss = criterion(outputs, targets.to(DEVICE))  # 计算loss
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        real_sets.extend(targets.numpy().tolist())  # 记录真实值
        pred_sets.extend(outputs.detach().cpu().numpy().tolist())  # 记录预测值
        train_loss_records.append(loss.item())  # 记录loss

    train_mae = round(mean_absolute_error(real_sets, pred_sets), 4)  # 计算MAE
    train_r2 = round(r2_score(real_sets, pred_sets), 4)  # 计算R2
    train_loss = round(sum(train_loss_records) / len(train_loss_records), 4)  # 求loss均值
    print(f"[train] Epoch: {epoch} / {epochs}, mae: {train_mae}, r2: {train_r2}, loss: {train_loss}")
    return train_mae, train_r2, train_loss


def evaluate(test_loader, model, criterion, epoch, epochs):
    model.eval()  # 测试模式
    real_sets = []  # 真实值
    pred_sets = []  # 预测值
    test_loss_records = []  # loss
    for idx, batch_data in enumerate(test_loader):  # 遍历
        inputs, targets = batch_data  # 输入 输出

        outputs = model(inputs.to(DEVICE))  # 前向传播
        loss = criterion(outputs, targets.to(DEVICE))  # 计算loss

        real_sets.extend(targets.numpy().tolist())  # 记录真实值
        pred_sets.extend(outputs.detach().cpu().numpy().tolist())  # 记录预测值
        test_loss_records.append(loss.item())  # 记录loss

    test_mae = round(mean_absolute_error(real_sets, pred_sets), 4)  # 计算MAE
    test_r2 = round(r2_score(real_sets, pred_sets), 4)  # 计算R2
    test_loss = round(sum(test_loss_records) / len(test_loss_records), 4)  # 求loss均值
    print(f"[test]  Epoch: {epoch} / {epochs}, mae: {test_mae}, r2: {test_r2}, loss: {test_loss}")
    return test_mae, test_r2, test_loss


def train(
    train_loader,
    test_loader,
    model,
    optimizer,
    criterion,
    model_path,
    r2_visualization_path,
    r2_visualization_csv_path,
    loss_visualization_path,
    loss_visualization_csv_path,
    epochs,
):
    train_r2_records = []  # 训练r2
    train_loss_records = []  # 训练loss
    test_r2_records = []  # 测试r2
    test_loss_records = []  # 测试loss
    for epoch in range(1, epochs + 1):
        train_mae, train_r2, train_loss = train_epoch(train_loader, model, optimizer, criterion, epoch, epochs)  # 训练
        test_mae, test_r2, test_loss = evaluate(test_loader, model, criterion, epoch, epochs)  # 测试

        train_r2_records.append(train_r2)  # 记录
        train_loss_records.append(train_loss)  # 记录
        test_r2_records.append(test_r2)  # 记录
        test_loss_records.append(test_loss)  # 记录

        torch.save(model.state_dict(), model_path)  # 保存模型

        if epoch == epochs:
            print(f"best test r2: {test_r2}, training finished!")
            break

    epoch_visualization(train_r2_records, test_r2_records, "R2", r2_visualization_path)  # 绘制r2图
    pd.DataFrame(
        {
            "epoch": list(range(1, len(train_r2_records) + 1)),
            "train r2": train_r2_records,
            "test r2": test_r2_records,
        }
    ).to_csv(
        r2_visualization_csv_path, index=False
    )  # 保存r2数据

    epoch_visualization(train_loss_records, test_loss_records, "Loss", loss_visualization_path)  # 绘制loss图
    pd.DataFrame(
        {
            "epoch": list(range(1, len(train_loss_records) + 1)),
            "train loss": train_loss_records,
            "test loss": test_loss_records,
        }
    ).to_csv(
        loss_visualization_csv_path, index=False
    )  # 保存loss数据

    return -test_r2_records[-1]


def train_run(batch_size, lr, epochs, hidden1_size, hidden2_size, dropout):
    setup_seed(seed=42)  # 设置随机种子

    X_train, X_test, y_train, y_test = load_data()  # 加载数据

    train_datasets = Datasets(X_train, y_train)  # 训练数据集
    test_datasets = Datasets(X_test, y_test)  # 测试数据集

    train_loader = DataLoader(
        train_datasets,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )  # 训练数据加载器
    test_loader = DataLoader(
        test_datasets,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )  # 测试数据加载器

    model = Model(hidden1_size=hidden1_size, hidden2_size=hidden2_size, dropout=dropout).to(DEVICE)  # 定义模型
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 优化器
    criterion = torch.nn.MSELoss()  # 损失函数

    return train(
        train_loader,
        test_loader,
        model,
        optimizer,
        criterion,
        MODEL_PATH,
        R2_VISUALIZATION_PATH,
        R2_VISUALIZATION_CSV_PATH,
        LOSS_VISUALIZATION_PATH,
        LOSS_VISUALIZATION_CSV_PATH,
        epochs,
    )  # 开始训练


def test_run(batch_size, lr, epochs, hidden1_size, hidden2_size, dropout):
    X_train, X_test, y_train, y_test = load_data()  # 加载数据

    test_datasets = Datasets(X_test, y_test)  # 测试数据集

    test_loader = DataLoader(
        test_datasets,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )  # 测试数据加载器

    model = Model(hidden1_size=hidden1_size, hidden2_size=hidden2_size, dropout=dropout)  # 定义模型
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))  # 加载模型参数
    model.to(DEVICE)  # 使用CPU/GPU
    model.eval()  # 测试模式

    real_sets = []  # 真实值
    pred_sets = []  # 预测值
    for idx, batch_data in enumerate(test_loader):  # 遍历
        inputs, targets = batch_data  # 输入 输出
        outputs = model(inputs.to(DEVICE))  # 前向传播

        real_sets.extend(targets.numpy().tolist())  # 记录真实值
        pred_sets.extend(outputs.detach().cpu().numpy().tolist())  # 记录预测值

    real_sets = standard_scaler_inverse(np.array(real_sets), TARGET_SCALER_PATH).reshape(-1).tolist()  # 反归一化
    pred_sets = standard_scaler_inverse(np.array(pred_sets), TARGET_SCALER_PATH).reshape(-1).tolist()  # 反归一化

    regression_evaluate(real_sets, pred_sets)  # 打印模型的性能指标
    residual_visualization(real_sets, pred_sets, RESIDUAL_VISUALIZATION_PATH, fitting=True)  # 残差图
    comparison_visualization(real_sets, pred_sets, COMPARISON_VISUALIZATION_PATH)  # 预测图
    pd.DataFrame({"real values": real_sets, "pred values": pred_sets}).to_csv(RESULT_VALUES_PATH)  # 保存预测值和真实值


if __name__ == "__main__":
    train_run(batch_size=16, lr=0.0003, epochs=200, hidden1_size=32, hidden2_size=32, dropout=0.2)  # 训练
    test_run(batch_size=16, lr=0.0003, epochs=200, hidden1_size=32, hidden2_size=32, dropout=0.2)  # 测试
