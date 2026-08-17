import pandas as pd
import glob
import os
import chardet
from torch_geometric.data import HeteroData
import numpy as np
import torch.nn as nn
from torch_geometric.nn import HGTConv
import torch.nn.functional as F
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import mean_squared_error
import networkx as nx
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch_geometric.nn import GATConv


'''
# 文档合并
# 设置包含CSV文件的目录路径
folder_path = r'E:\西铁院\列车数据\二次处理-提取范围'

# 检查路径是否存在
if not os.path.exists(folder_path):
    print(f"错误：路径 '{folder_path}' 不存在")
    exit()

# 获取所有CSV文件路径
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
print(f"找到 {len(csv_files)} 个CSV文件")

if not csv_files:
    print("错误：没有找到任何CSV文件")
    exit()

# 读取并合并文件
dfs = []
success_count = 0
failure_count = 0

for file_path in csv_files:
    file_name = os.path.basename(file_path)

    try:
        # 检测文件编码
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # 读取前10000字节检测编码
            result = chardet.detect(raw_data)
            file_encoding = result['encoding']
            confidence = result['confidence']

            # 处理低置信度或None编码
            if not file_encoding or confidence < 0.7:
                file_encoding = 'gbk'  # 默认使用中文编码

        print(f"尝试读取: {file_name} (检测编码: {file_encoding}, 置信度: {confidence:.2f})")

        # 尝试读取文件
        df = pd.read_csv(file_path, encoding=file_encoding, engine='python', on_bad_lines='warn')

        # 添加来源文件名列
        df['source_file'] = file_name

        dfs.append(df)
        success_count += 1
        print(f"√ 成功加载: {file_name} ({len(df)} 行)\n")

    except Exception as e:
        failure_count += 1
        print(f"× 读取文件 {file_name} 失败: {str(e)}")

        # 尝试其他常见中文编码
        retry_encodings = ['gbk', 'gb2312', 'gb18030', 'big5', 'utf-16', 'latin1']
        for encoding in retry_encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, engine='python', on_bad_lines='warn')
                df['source_file'] = file_name
                dfs.append(df)
                success_count += 1
                failure_count -= 1
                print(f"√ 使用备用编码 {encoding} 成功读取: {file_name} ({len(df)} 行)\n")
                break
            except:
                continue

print(f"\n文件读取总结: 成功 {success_count} 个, 失败 {failure_count} 个")

if not dfs:
    print("错误：没有成功读取任何CSV文件")
    exit()

# 合并所有DataFrame
try:
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"\n合并完成! 总行数: {len(merged_df)}")
    print(f"总列数: {len(merged_df.columns)}")

    # 显示前几行数据
    print("\n合并数据预览:")
    print(merged_df.head(3))

except Exception as e:
    print(f"合并文件时出错: {str(e)}")
    exit()

# 保存结果
output_path = os.path.join(folder_path, 'merged_all_files.csv')
try:
    # 使用支持中文的编码保存
    merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n成功保存合并文件到: {output_path}")

except Exception as e:
    print(f"保存文件时出错: {str(e)}")
    # 尝试保存到当前工作目录
    current_dir_path = os.path.join(os.getcwd(), 'merged_all_files.csv')
    try:
        merged_df.to_csv(current_dir_path, index=False, encoding='utf-8-sig')
        print(f"已将文件保存到当前目录: {current_dir_path}")
    except Exception as e2:
        print(f"保存到当前目录也失败: {str(e2)}")'''


# 安全标准化函数 (防止除零)
def safe_standardize(X):
    if X.size == 0:
        return X
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std < 1e-6] = 1.0  # 避免除零
    return (X - mean) / std


# 时间转换函数 (增强健壮性)
def time_to_min(tstr):
    try:
        if isinstance(tstr, (int, float)):
            return int(tstr)
        # 处理可能的非标准格式
        parts = str(tstr).strip().split(':')
        if len(parts) == 1:
            return int(parts[0])  # 只有分钟的情况
        h, m = map(int, parts)
        return h * 60 + m
    except:
        return 0  # 返回0而不是NaN


# 读取数据
df = pd.read_csv(r'E:\\西铁院\\列车晚点研究\\二次处理-提取范围\\6-4.merged_files.csv', encoding='utf-8')

# 确保数据中没有NaN
df = df.dropna(subset=['发站', '到站', '预计发时', '预计到时', '实际发时', '实际到时', '里程', '历时'])
dates_list = []
graph_list = []
day_dfs = []
for date, df_day in df.groupby('日期'):
    # 按预计发时排序确保时序正确
    df_day = df_day.sort_values(by='预计发时').reset_index(drop=True)

    data = HeteroData()
    stations = list(set(df_day['发站']) | set(df_day['到站']))
    station_id = {s: i for i, s in enumerate(stations)}

    # ─── 1. 准备车站特征 ───────────────────────────────
    depart_counts = df_day['发站'].value_counts().to_dict()
    arrive_counts = df_day['到站'].value_counts().to_dict()
    weather_vals = df_day['强度'].unique()
    weather_map = {w: i for i, w in enumerate(weather_vals)}

    raw_station_feats = []
    for s in stations:
        dep_cnt = depart_counts.get(s, 0)
        arr_cnt = arrive_counts.get(s, 0)

        # 处理天气数据
        w_data = df_day[(df_day['发站'] == s) | (df_day['到站'] == s)]['强度']
        ws = weather_map.get(w_data.iloc[0], 0) if not w_data.empty else 0

        # 处理车站等级
        lvl_data = df_day[(df_day['发站'] == s) | (df_day['到站'] == s)]['车站等级']
        if not lvl_data.empty:
            lvl_map = {'一等站': 1, '二等站': 2, '三等站': 3}
            sl = lvl_map.get(lvl_data.iloc[0], 0)
        else:
            sl = 0

        # 在车站特征中加入更多信息
        raw_station_feats.append([
            dep_cnt,
            arr_cnt,
            ws,
            sl,
            len(df_day[df_day['发站'] == s]),  # 发车次数
            len(df_day[df_day['到站'] == s])  # 到达次数
        ])

    # 安全标准化
    raw_station_feats = np.array(raw_station_feats)

    station_feats_norm = safe_standardize(raw_station_feats)

    data['station_node'].x = torch.from_numpy(station_feats_norm.astype(np.float32))

    # ─── 2. 准备列车节点特征 (移除目标值) ────────────────
    train_ids = df_day.index.tolist()
    raw_train_feats = []
    raw_train_delays = []

    for _, row in df_day.iterrows():
        mi = float(row['里程'])
        du = time_to_min(row['历时'])
        dl = float(row['晚点时长'])
        tl = 1 if row['列车等级'] == '高铁' else 0

        # 特征中不包含目标值 (晚点时长)
        raw_train_feats.append([mi, du, tl])
        raw_train_delays.append(dl)

    # 安全标准化 (只使用特征)
    raw_train_feats = np.array(raw_train_feats)
    train_feats_norm = safe_standardize(raw_train_feats)
    data['train_node'].x = torch.from_numpy(train_feats_norm.astype(np.float32))
    data['train_node'].y = torch.tensor(raw_train_delays, dtype=torch.float32)

    # ─── 3. 构建边 ──────────────────────────────────
    src = list(range(len(raw_train_feats)))

    # 发车边
    dep_planned = np.array(df_day['预计发时'].apply(time_to_min).tolist(), dtype=np.float32)
    dep_actual = np.array(df_day['实际发时'].apply(time_to_min).tolist(), dtype=np.float32)
    raw_dep_attr = np.vstack([dep_planned, dep_actual]).T
    dep_attr_norm = safe_standardize(raw_dep_attr)

    data['train_node', 'depart', 'station_node'].edge_index = torch.tensor([
        src,
        [station_id[s] for s in df_day['发站']]
    ], dtype=torch.long)

    data['station_node', 'rev_depart', 'train_node'].edge_index = data[
        'train_node', 'depart', 'station_node'
    ].edge_index.flip(0)

    data['train_node', 'depart', 'station_node'].edge_attr = torch.from_numpy(dep_attr_norm.astype(np.float32))

    # 到达边
    arr_planned = np.array(df_day['预计到时'].apply(time_to_min).tolist(), dtype=np.float32)
    arr_actual = np.array(df_day['实际到时'].apply(time_to_min).tolist(), dtype=np.float32)
    raw_arr_attr = np.vstack([arr_planned, arr_actual]).T
    arr_attr_norm = safe_standardize(raw_arr_attr)

    data['train_node', 'arrive', 'station_node'].edge_index = torch.tensor([
        src,
        [station_id[s] for s in df_day['到站']]
    ], dtype=torch.long)

    # 增加 station_node -> train_node 的 arrive 反向边
    data['station_node', 'rev_arrive', 'train_node'].edge_index = data[
        'train_node', 'arrive', 'station_node'
    ].edge_index.flip(0)

    data['train_node', 'arrive', 'station_node'].edge_attr = torch.from_numpy(arr_attr_norm.astype(np.float32))

    # 前后继边
    follows, headways = [[], []], []
    for _, grp in df_day.groupby('始发-终到'):
        if len(grp) < 2:
            continue

        grp = grp.sort_values(by='预计发时')
        prev_idx, prev_time = None, None
        for ridx, row in grp.iterrows():
            cur = grp.index.get_loc(ridx)
            t = time_to_min(row['预计发时'])
            if prev_idx is not None:
                follows[0].append(prev_idx)
                follows[1].append(cur)
                headways.append(t - prev_time)
            prev_idx, prev_time = cur, t

    if follows[0]:
        headways = np.array(headways, dtype=np.float32).reshape(-1, 1)
        hw_norm = safe_standardize(headways)
        data['train_node', 'follows', 'train_node'].edge_index = torch.tensor(follows, dtype=torch.long)

        # 再添加反向 follows⁻¹
        data['train_node', 'rev_follows', 'train_node'].edge_index = data[
            'train_node', 'follows', 'train_node'
        ].edge_index.flip(0)

        data['train_node', 'follows', 'train_node'].edge_attr = torch.from_numpy(hw_norm.astype(np.float32))
    else:
        # 创建空边
        data['train_node', 'follows', 'train_node'].edge_index = torch.empty((2, 0), dtype=torch.long)
        data['train_node', 'follows', 'train_node'].edge_attr = torch.empty((0, 1), dtype=torch.float32)

    graph_list.append(data)
    day_dfs.append(df_day)
    dates_list.append(date)


# 获取特征维度
sample = graph_list[0]
in_dim_train = sample['train_node'].x.shape[1]  # 3 (移除了晚点时长)
in_dim_station = sample['station_node'].x.shape[1]  # 4


# 修改模型：添加层归一化和更小的初始学习率
class DelayPredictor(torch.nn.Module):
    def __init__(self, in_dim_train, in_dim_station, hidden_dim, num_heads):
        super().__init__()
        self.lin_train = torch.nn.Linear(in_dim_train, hidden_dim)
        self.lin_station = torch.nn.Linear(in_dim_station, hidden_dim)

        # 添加层归一化
        self.norm_train = torch.nn.LayerNorm(hidden_dim)
        self.norm_station = torch.nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.5)  # 新增
        self.hgt1 = HGTConv(
            hidden_dim, hidden_dim,
            metadata=(['train_node', 'station_node'],
                      [('train_node', 'depart', 'station_node'),
                       ('train_node', 'arrive', 'station_node'),
                       ('station_node', 'rev_depart', 'train_node'),
                       ('station_node', 'rev_arrive', 'train_node'),
                       ('train_node', 'follows', 'train_node'),
                       ('train_node', 'rev_follows', 'train_node'),
                       ]),
            heads=num_heads
        )
        self.hgt2 = HGTConv(
            hidden_dim, hidden_dim,
            metadata=(['train_node', 'station_node'],
                      [('train_node', 'depart', 'station_node'),
                       ('train_node', 'arrive', 'station_node'),
                       ('station_node', 'rev_depart', 'train_node'),
                       ('station_node', 'rev_arrive', 'train_node'),
                       ('train_node', 'follows', 'train_node'),
                       ('train_node', 'rev_follows', 'train_node'),
                       ]),
            heads=num_heads
        )
        self.gat_follows = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            concat=False,
            add_self_loops=False,
        )# 保持与 HGT 卷积相同的拓扑
        self.gru = torch.nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )


    def forward(self, data, hidden_state):
        x_dict = {
            'train_node': self.norm_train(self.lin_train(data['train_node'].x)),
            'station_node': self.norm_station(self.lin_station(data['station_node'].x))
        }

        x_dict = self.hgt1(x_dict, data.edge_index_dict)
        x_dict = {k: self.dropout(v) for k, v in x_dict.items()}
        x_dict = self.hgt2(x_dict, data.edge_index_dict)
        train_feat = x_dict['train_node']

        # 处理空序列
        if train_feat.size(0) == 0:
            return torch.tensor([], device=train_feat.device), None

        train_feat_seq = train_feat.unsqueeze(0)
        gru_out, h = self.gru(train_feat_seq, hidden_state)
        gru_out = gru_out.squeeze(0)
        pred_delay = self.fc(gru_out).squeeze(-1)
        edge_index = data['train_node','follows','train_node'].edge_index
        _, (edge_idx, att_weights) = self.gat_follows(
                train_feat, edge_index, return_attention_weights=True
        )
        self.latest_att = {
            'edge_index': edge_idx.cpu().numpy(),        # shape [2, E]
            'att_weights': att_weights.detach().cpu().numpy()  # shape [E, heads]
        }

        return pred_delay, h

    def get_follows_attention(self):
        """返回最近一次 forward 时计算得到的 follows 边 attention"""
        return self.latest_att
# ======== 添加目标值标准化代码 ========

# 步骤1：收集所有图中的目标值
all_delays = []
for data in graph_list:
    all_delays.extend(data['train_node'].y.tolist())

# 步骤2：计算全局均值和标准差
delay_mean = np.mean(all_delays)
delay_std = np.std(all_delays) + 1e-6  # 防止除零

print(f"全局晚点均值: {delay_mean:.2f}, 标准差: {delay_std:.2f}")

# 步骤3：标准化所有图中的目标值
for data in graph_list:
    data['train_node'].y = (data['train_node'].y - delay_mean) / delay_std

# 划分训练集和验证集 (80% 训练, 20% 验证)
split_idx = int(len(graph_list) * 0.8)
train_graphs = graph_list[:split_idx]
val_graphs = graph_list[split_idx:]

# 初始化损失记录器
train_loss_history = []
val_loss_history = []
# ======== 模型初始化 ========
sample = graph_list[0]
in_dim_train = sample['train_node'].x.shape[1]
in_dim_station = sample['station_node'].x.shape[1]

# 初始化模型 (更小的隐藏维度)
model = DelayPredictor(
    in_dim_train=in_dim_train,
    in_dim_station=in_dim_station,
    hidden_dim=16,
    num_heads=4
)

# 使用更小的学习率和梯度裁剪
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
loss_fn = torch.nn.MSELoss()
# loss_fn = torch.nn.L1Loss()  # 即 MAE


num_epochs = 20
loss_history = []

for epoch in range(num_epochs):
    # ======== 训练阶段 ========
    model.train()
    epoch_train_loss = 0.0
    hidden = None

    for data in train_graphs:
        # 重置隐藏状态
        if hidden is not None and hidden.size(1) != data['train_node'].x.size(0):
            hidden = None

        optimizer.zero_grad()
        pred, hidden = model(data, hidden)

        # 跳过空预测
        if pred.numel() == 0:
            continue

        # 分离隐藏状态防止梯度爆炸
        hidden = hidden.detach()

        loss = loss_fn(pred, data['train_node'].y)
        epoch_train_loss += loss.item()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        loss.backward()
        optimizer.step()

    avg_train_loss = epoch_train_loss / len(train_graphs)
    train_loss_history.append(avg_train_loss)

    # ======== 验证阶段 ========
    model.eval()
    epoch_val_loss = 0.0

    with torch.no_grad():
        for data in val_graphs:
            # 重置隐藏状态
            hidden_val = None

            pred, hidden_val = model(data, hidden_val)

            # 跳过空预测
            if pred.numel() == 0:
                continue

            loss = loss_fn(pred, data['train_node'].y)
            epoch_val_loss += loss.item()

    avg_val_loss = epoch_val_loss / len(val_graphs)
    val_loss_history.append(avg_val_loss)

    # ======== 打印和记录损失 ========
    print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}")

    # 保存每个epoch的损失到CSV
    with open('loss_history.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        # 如果是第一个epoch，写入表头
        if epoch == 0:
            writer.writerow(['epoch', 'train_loss', 'val_loss'])
        writer.writerow([epoch, avg_train_loss, avg_val_loss])

    #注意力系数计算
    model.eval()
    with torch.no_grad():
        target_date = '2022.6.23'
        # 1) 找到 target_date 在 dates_list 中的索引
        idx = dates_list.index(target_date)
         # 2) 取出对应的异构图
        data_graph = graph_list[idx]
         # 3) 前向推理，获取 attention
        pred, _ = model(data_graph, None)

    att_data = model.get_follows_attention()
    edge_idx = att_data['edge_index']  # numpy array [2, E]
    att_weights = att_data['att_weights']  # numpy array [E, heads]

    # 将多头 attention 合并为单头（取平均）
    att_mean = att_weights.mean(axis=1)  # shape [E]

    # 保存到 CSV，列出每条 follows 边及其 attention
    import csv

    with open('follows_attention.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['src_train', 'dst_train', 'att_weight'])
        for i in range(edge_idx.shape[1]):
            src = int(edge_idx[0, i])
            dst = int(edge_idx[1, i])
            w = float(att_mean[i])
            writer.writerow([src, dst, f"{w:.4f}"])
    print("✅ 已保存 follows 边的注意力系数到 follows_attention.csv")

# ======== 绘制损失曲线 ========
plt.figure(figsize=(10, 6))
plt.plot(range(num_epochs), train_loss_history, label='Training Loss', marker='o')
plt.plot(range(num_epochs), val_loss_history, label='Validation Loss', marker='x')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('train_val_loss_curve.png', dpi=150)
plt.show()

print("✅ 训练和验证损失已保存至 loss_history.csv，损失曲线保存为 train_val_loss_curve.png")

# 2. 在训练完成后，用全数据或留出集计算 MSE 和 RMSE
all_true = []
all_pred = []
model.eval()
with torch.no_grad():
    for data in graph_list:
        pred, _ = model(data, None)
        if pred.numel() == 0: continue
        # 反标准化预测值和真实值
        true = data['train_node'].y * delay_std + delay_mean
        p = pred * delay_std + delay_mean
        all_true.extend(true.cpu().numpy().tolist())
        all_pred.extend(p.cpu().numpy().tolist())

mse = mean_squared_error(all_true, all_pred)
rmse = mse ** 0.5
# mae  = mean_absolute_error(all_true, all_pred)
# rmse = mean_squared_error(all_true, all_pred, squared=False)  # 直接得到 RMSE
print(f"模型评估结果：MAE = {mse:.3f} 分钟，RMSE = {rmse:.3f} 分钟")


# 3. 计算传播速度和影响范围（源站为“杭州”）
import os


# 确保目标目录存在
os.makedirs("propagation_results", exist_ok=True)

# 设定日期
target_date = '2022.6.23'

try:
    idx = dates_list.index(target_date)
except ValueError:
    raise ValueError(f"未在 dates_list 中找到目标日期 {target_date}")

df_day = day_dfs[idx].copy()
data_graph = graph_list[idx]

# 模型预测
model.eval()
with torch.no_grad():
    pred_norm, _ = model(data_graph, None)
pred = (pred_norm.cpu().numpy() * delay_std) + delay_mean

# 合并预测值
df_day = df_day.copy().reset_index(drop=True)
df_day['pred_delay'] = pred

# 所有发站
# --------- 新增：可视化函数（pyvis & plotly） ----------
def visualize_propagation_network_pyvis(df_day, att_edges=None, pred_col='pred_delay', out_html='propagation_network.html'):
    """
    使用 pyvis 生成交互式传播网络（HTML）。
    - 节点：每个车次（按实际发时排序）
    - 节点颜色/大小：预测延误大小
    - 边：follows 边，若提供 att_edges (dict of (src,dst)->weight)，用权重着色/宽度
    """
    try:
        from pyvis.network import Network
    except Exception as e:
        print("pyvis 未安装，请 pip install pyvis 来生成交互 HTML。", e)
        return

    net = Network(height='700px', width='100%', notebook=False, directed=True)
    # 选当天车次排序
    nodes = df_day.reset_index(drop=True)
    nodes['act_dep_min'] = nodes['实际发时'].apply(time_to_min)
    nodes = nodes.sort_values('act_dep_min').reset_index(drop=True)

    # 添加节点
    max_delay = max(nodes[pred_col].max(), 1e-6)
    for idx, row in nodes.iterrows():
        nid = int(idx)
        label = f"{row['车次']}\\n{row['发站']}->{row['到站']}\\n{row['实际发时']}"
        delay = float(row[pred_col])
        size = 8 + (delay / max_delay) * 30
        color = 'red' if delay > 0 else 'lightgray'
        title = f"pred_delay: {delay:.2f} min<br>planned dep: {row['预计发时']}<br>actual dep: {row['实际发时']}"
        net.add_node(nid, label=label, title=title, color=color, size=size)

    # 添加 follows 边（若存在）
    # match by same '始发-终到' route and earlier->later actual dep
    for route, grp in df_day.groupby('始发-终到'):
        grp = grp.sort_values(by='实际发时', key=lambda s: s.apply(time_to_min)).reset_index(drop=True)
        for i in range(len(grp)-1):
            src = nodes[nodes['车次'] == grp.loc[i, '车次']].index[0]
            dst = nodes[nodes['车次'] == grp.loc[i+1, '车次']].index[0]
            w = 1.0
            if att_edges is not None:
                # att_edges expected as dict {(src_train, dst_train): weight}
                key = (grp.loc[i, '车次'], grp.loc[i+1, '车次'])
                w = att_edges.get(key, 1.0)
            net.add_edge(int(src), int(dst), value=float(w), title=f"weight:{w:.4f}")

    net.show(out_html)
    print(f"Interactive propagation network saved to {out_html}")


def plot_atr_sar_plotly(summary_list, out_html='atr_sar_summary.html'):
    """
    summary_list: list of dict, each dict includes:
        {'station':..., 'ATR':..., 'SAR_km_min':..., 'SAR_km_h':..., 'n_trains':..., 'n_stations':...}
    生成交互式条形图（Plotly），并保存为 HTML。
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except:
        print("plotly 未安装，请 pip install plotly 来生成交互图。")
        return

    import pandas as pd
    df = pd.DataFrame(summary_list)
    if df.empty:
        print("No summary to plot.")
        return

    # ATR (train/min) 条形图
    fig1 = px.bar(df, x='station', y='ATR', title='Affected Train Rate (trains/min) by departure station',
                  labels={'ATR':'trains/min', 'station':'Departure Station'})
    fig1.write_html(out_html.replace('.html', '_ATR.html'))
    print(f"Saved ATR chart to {out_html.replace('.html', '_ATR.html')}")

    # SAR (km/h) 条形图
    fig2 = px.bar(df, x='station', y='SAR_km_h', title='Spatial Affected Rate (km/h) by departure station',
                  labels={'SAR_km_h':'km/h', 'station':'Departure Station'})
    fig2.write_html(out_html.replace('.html', '_SAR.html'))
    print(f"Saved SAR chart to {out_html.replace('.html', '_SAR.html')}")

# --------- 传播计算与保存（替换原块） ----------
os.makedirs("propagation_results", exist_ok=True)
target_date = '2022.6.23'
try:
    idx = dates_list.index(target_date)
except ValueError:
    raise ValueError(f"未在 dates_list 中找到目标日期 {target_date}")

df_day = day_dfs[idx].copy().reset_index(drop=True)
data_graph = graph_list[idx]

# 模型预测并反标准化
model.eval()
with torch.no_grad():
    pred_norm, _ = model(data_graph, None)
pred = (pred_norm.cpu().numpy() * delay_std) + delay_mean

df_day['pred_delay'] = pred

unique_stations = df_day['发站'].unique()
summary_list = []  # 用于可视化汇总

# 封装辅助函数：把 (车次对) Attention edges 转为 dict（若可用）
def build_att_dict(edge_idx_np, att_mean_np, df_day_local):
    # edge_idx_np: [2, E] pairs of train-node indices (these indices relate to train-node ordering in data_graph)
    # We will map them to actual train IDs by index mapping: data_graph order equals df_day order (constructed earlier)
    att_dict = {}
    # if mapping differs, user must adapt
    for i in range(edge_idx_np.shape[1]):
        src_idx = int(edge_idx_np[0, i])
        dst_idx = int(edge_idx_np[1, i])
        if src_idx < len(df_day_local) and dst_idx < len(df_day_local):
            src_train = df_day_local.loc[src_idx, '车次']
            dst_train = df_day_local.loc[dst_idx, '车次']
            att_dict[(src_train, dst_train)] = float(att_mean_np[i])
    return att_dict

# 尝试取得 attention edges（若模型计算过）
att_dict_global = {}
try:
    att_data = model.get_follows_attention()
    edge_idx = att_data['edge_index']  # numpy [2, E]
    att_weights = att_data['att_weights']
    att_mean = att_weights.mean(axis=1)
    att_dict_global = build_att_dict(edge_idx, att_mean, df_day)
except Exception as e:
    # 无 attention 数据也可以继续
    att_dict_global = {}
    print("No follows attention available or mapping mismatch:", e)

for station in unique_stations:
    df_src_all = df_day[df_day['发站'] == station].copy()
    if df_src_all.empty:
        continue
    # Sort by actual dep time, fallback to planned if missing
    df_src_all['act_dep_min'] = df_src_all['实际发时'].apply(time_to_min)
    df_src_all = df_src_all.sort_values(by='act_dep_min').reset_index(drop=True)

    # only keep those with pred_delay > 0 as candidates (for seed search)
    if not (df_src_all['pred_delay'] > 0).any():
        continue

    # choose first delayed train as seed (earliest actual dep with delay>0)
    # choose first delayed train as seed (earliest actual dep with delay>0)
    seed_row = df_src_all[df_src_all['pred_delay'] > 0].iloc[0]
    seed_time = time_to_min(seed_row['实际发时'])
    seed_train = seed_row['车次']

    # affected: trains with actual dep > seed_time and pred_delay>0 (exclude seed itself)
    affected = df_src_all[
        (df_src_all['实际发时'].apply(time_to_min) > seed_time) & (df_src_all['pred_delay'] > 0)].copy()
    n_affected = len(affected)

    # prepare records: include seed first
    records = []
    records.append([target_date, seed_row['车次'], seed_row['发站'], seed_row['到站'], f"{seed_row['pred_delay']:.1f}"])

    # initialize affected sets (include seed if you want seed counted as affected station/train)
    affected_stations = set()
    affected_trains = set()
    affected_stations.add(seed_row['到站'])
    affected_trains.add(seed_row['车次'])

    if n_affected == 0:
        # no subsequent affected trains -> keep records containing just the seed
        avg_speed_min = 0.0  # we set avg_speed_min to ATR (trains/min) => 0
        avg_speed_hr = 0.0  # we set avg_speed_hr to SAR (km/h) => 0
    else:
        # 填充 records 并统计时间与里程
        # 计算 ATR: n_affected / (max_dep - seed_time)
        affected['act_dep_min'] = affected['实际发时'].apply(time_to_min)
        max_dep = int(affected['act_dep_min'].max())
        delta_T = max_dep - seed_time  # minutes window
        ATR = n_affected / delta_T if delta_T > 0 else 0.0  # trains per minute

        # 计算 SAR: sum(distances) / sum(time_from_seed)
        affected['time_from_seed'] = affected['act_dep_min'] - seed_time
        sum_time_from_seed = float(affected['time_from_seed'].sum())
        sum_dist = float(affected['里程'].astype(float).sum())
        SAR_km_min = sum_dist / sum_time_from_seed if sum_time_from_seed > 0 else 0.0


        # 填充 records 与 affected sets
        for _, r in affected.iterrows():
            records.append([target_date, r['车次'], r['发站'], r['到站'], f"{r['pred_delay']:.1f}"])
            affected_trains.add(r['车次'])
            affected_stations.add(r['到站'])

        # 将变量映射为原脚本期望的名字（保持 CSV 写入段落不变）
        # avg_speed_min 将承载 ATR（trains/min），avg_speed_hr 承载 SAR(km/h)
        avg_speed_min = ATR
        avg_speed_hr = SAR_km_min

    # ---------- 此处开始保持你原有的不变 CSV 写入段落（**完全逐字未改**） ----------
    filename = f"propagation_results/{target_date.replace('.', '-')}_{station}_传播统计.csv"
    with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['日期', '车次', '发站', '到站', '预测延误(min)'])
        writer.writerows(records)
        writer.writerow([])
        writer.writerow(['DPR(km/min)', '', '', '', f"{avg_speed_min:.2f}"])
        writer.writerow(['STDS(km/h)', '', '', '', f"{avg_speed_hr:.1f}"])
        writer.writerow(['受影响车站数', '', '', '', len(affected_stations)])
        writer.writerow(['受影响列车数', '', '', '', len(affected_trains)])

    print(f"✅ 结果已保存至 {filename}\n")

