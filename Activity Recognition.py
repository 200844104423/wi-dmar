"""
Wi-DMAR: 跨域活动识别 — 改进版
主要改进：
  1. Hard Triplet Mining（解决 loss=1.0 的塌陷问题）
  2. 域对抗训练 DANN（Domain Adversarial Neural Network）
  3. MMD 正则化（最大均值差异，拉近源域/目标域分布）
  4. Prototypical Network 推理（比最近邻更鲁棒）
  5. 数据增强（时序抖动 + 高斯噪声 + 子载波dropout）
  6. 修复 N_CLASSES=9（实际9类）
  7. 支持集/评估集分离（不再用同一批数据）
  8. 目标域无标签数据用于域适应
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
from collections import defaultdict

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("[ERROR] h5py 未安装，请运行: pip install h5py")


# ═══════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════
class Config:
    SOURCE_ROOT  = "./csi-processed-data22"
    TARGET_ROOT  = "./processed_csi_data22"
    SOURCE_USERS = ["user15", "user16", "user17"]
    PSEUDO_USER  = "zengqiang40"

    N_SUBCARRIERS = 27
    TIME_STEPS    = 500
    N_CLASSES     = 9        # ← 修复：实际9类

    CNN_CHANNELS  = [64, 128, 256]
    LSTM_HIDDEN   = 256
    EMBED_DIM     = 256
    MARGIN        = 0.5      # ← 降低 margin，避免梯度消失

    BATCH_SIZE    = 64       # ← 增大 batch，Hard Mining 需要更多样本
    EPOCHS        = 150
    LR            = 5e-4
    WEIGHT_DECAY  = 1e-4
    PATIENCE      = 30

    # 域适应权重（逐步增大）
    LAMBDA_DOMAIN = 0.1      # 域对抗损失权重
    LAMBDA_MMD    = 0.1      # MMD 损失权重

    # 分类头（辅助损失，加速收敛）
    LAMBDA_CLS    = 0.5

    N_SHOT        = 5
    SEED          = 42
    DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cfg = Config()
random.seed(cfg.SEED)
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)


# ═══════════════════════════════════════════════════════
# .mat 文件读取（复用原版）
# ═══════════════════════════════════════════════════════
def _struct_to_amp(arr):
    real = arr['real'].astype(np.float64)
    imag = arr['imag'].astype(np.float64)
    return np.sqrt(real ** 2 + imag ** 2).astype(np.float32)

def _complex_to_amp(arr):
    return np.abs(arr).astype(np.float32)

def load_mat_file(filepath):
    try:
        mat = sio.loadmat(filepath)
        keys = [k for k in mat.keys() if not k.startswith('_')]
        for k in keys:
            v = np.array(mat[k])
            if np.iscomplexobj(v) and v.ndim == 2:
                amp = _complex_to_amp(v)
                if amp.shape[0] < amp.shape[1] and amp.shape[0] <= 120:
                    amp = amp.T
                return amp
            elif v.ndim == 2 and v.size > 100:
                arr = v.real.astype(np.float32)
                if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 120:
                    arr = arr.T
                return arr
    except Exception:
        pass

    if not HAS_H5PY:
        raise RuntimeError("需要 h5py: pip install h5py")

    with h5py.File(filepath, 'r') as f:
        candidate_keys = ['csi_data', 'processed_csi_data', 'csi', 'data',
                          'amp', 'CSIamp', 'csiData', 'amplitude', 'CSI']
        for key in candidate_keys:
            if key not in f:
                continue
            obj = f[key]
            if not isinstance(obj, h5py.Dataset):
                continue
            raw  = obj[()]
            dtype = obj.dtype
            if dtype.names and 'real' in dtype.names and 'imag' in dtype.names:
                amp = _struct_to_amp(raw)
                if amp.shape[0] < amp.shape[1] and amp.shape[0] <= 120:
                    amp = amp.T
                return amp
            if np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.complexfloating):
                amp = _complex_to_amp(raw) if np.iscomplexobj(raw) else raw.astype(np.float32)
                if amp.shape[0] < amp.shape[1] and amp.shape[0] <= 120:
                    amp = amp.T
                return amp
        raise ValueError(f"未找到可读变量。根键: {list(f.keys())}")


def parse_label(filename):
    basename = os.path.splitext(os.path.basename(filename))[0]
    parts = basename.split('-')
    try:
        return int(parts[1]) - 1
    except (IndexError, ValueError):
        return 0


def normalize(arr):
    mean = arr.mean(axis=0, keepdims=True)
    std  = arr.std(axis=0, keepdims=True) + 1e-8
    return (arr - mean) / std

def resize_time(arr, T):
    if arr.shape[0] == T:
        return arr
    x_old = np.linspace(0, 1, arr.shape[0])
    x_new = np.linspace(0, 1, T)
    out = np.zeros((T, arr.shape[1]), dtype=np.float32)
    for s in range(arr.shape[1]):
        out[:, s] = np.interp(x_new, x_old, arr[:, s])
    return out

def process(arr, n_subcarriers, time_steps):
    if arr.shape[1] > n_subcarriers:
        arr = arr[:, :n_subcarriers]
    elif arr.shape[1] < n_subcarriers:
        pad = np.zeros((arr.shape[0], n_subcarriers - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    arr = resize_time(arr, time_steps)
    arr = normalize(arr)
    return arr

def load_folder(folder_path, n_subcarriers, time_steps, tag=""):
    if not os.path.exists(folder_path):
        print(f"[WARN] 文件夹不存在: {folder_path}")
        return []
    mat_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.mat')])
    if not mat_files:
        return []
    samples, skipped = [], 0
    for fname in mat_files:
        fpath = os.path.join(folder_path, fname)
        label = parse_label(fname)
        try:
            arr = load_mat_file(fpath)
            arr = process(arr, n_subcarriers, time_steps)
            samples.append((arr, label))
        except Exception as e:
            skipped += 1
    name = tag or os.path.basename(folder_path)
    print(f"  {name}: {len(samples)} 样本" + (f"（跳过{skipped}）" if skipped else ""))
    return samples

def load_target_domain(root, n_subcarriers, time_steps):
    if not os.path.exists(root):
        return []
    entries = os.listdir(root)
    has_subdir = any(os.path.isdir(os.path.join(root, e)) for e in entries)
    if has_subdir:
        samples = []
        for d in sorted(e for e in entries if os.path.isdir(os.path.join(root, e))):
            samples.extend(load_folder(os.path.join(root, d), n_subcarriers, time_steps, tag=f"target/{d}"))
        return samples
    else:
        return load_folder(root, n_subcarriers, time_steps, tag="target")


# ═══════════════════════════════════════════════════════
# 数据增强
# ═══════════════════════════════════════════════════════
def augment(arr, training=True):
    """arr: (T, S) numpy → 增强后 (T, S)"""
    if not training:
        return arr
    # 1. 高斯噪声
    if random.random() < 0.5:
        arr = arr + np.random.randn(*arr.shape).astype(np.float32) * 0.05
    # 2. 时序抖动（随机时移）
    if random.random() < 0.5:
        shift = random.randint(-20, 20)
        arr = np.roll(arr, shift, axis=0)
    # 3. 子载波 Dropout
    if random.random() < 0.3:
        drop_idx = random.sample(range(arr.shape[1]), k=max(1, arr.shape[1] // 8))
        arr = arr.copy()
        arr[:, drop_idx] = 0.0
    # 4. 幅度缩放
    if random.random() < 0.5:
        scale = random.uniform(0.8, 1.2)
        arr = arr * scale
    return arr


# ═══════════════════════════════════════════════════════
# Dataset：Hard Triplet Mining
# ═══════════════════════════════════════════════════════
class TripletDataset(Dataset):
    """返回 (anchor, positive, negative, label)，支持在线 Hard Mining。"""
    def __init__(self, samples, training=True):
        self.data     = [s[0] for s in samples]
        self.labels   = [s[1] for s in samples]
        self.training = training
        self.cls_idx  = defaultdict(list)
        for i, lbl in enumerate(self.labels):
            self.cls_idx[lbl].append(i)
        if len(self.cls_idx) < 2:
            raise ValueError(f"需要至少 2 个类别，当前: {list(self.cls_idx.keys())}")

    def __len__(self):
        return len(self.data)

    def _t(self, idx, aug=True):
        arr = self.data[idx]
        if aug and self.training:
            arr = augment(arr)
        return torch.tensor(arr.T, dtype=torch.float32)  # (S, T)

    def __getitem__(self, idx):
        a_lbl = self.labels[idx]
        pp = [i for i in self.cls_idx[a_lbl] if i != idx]
        pi = random.choice(pp) if pp else idx
        nl = random.choice([l for l in self.cls_idx if l != a_lbl])
        ni = random.choice(self.cls_idx[nl])
        return self._t(idx), self._t(pi), self._t(ni), a_lbl


class UnlabeledDataset(Dataset):
    """目标域无标签数据集（用于域适应）。"""
    def __init__(self, samples):
        self.data = [s[0] for s in samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        arr = augment(self.data[idx])
        return torch.tensor(arr.T, dtype=torch.float32)  # (S, T)


# ═══════════════════════════════════════════════════════
# 梯度反转层（DANN 域适应核心）
# ═══════════════════════════════════════════════════════
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.tensor(alpha))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        alpha = ctx.saved_tensors[0].item()
        return -alpha * grad_output, None

class GradientReversal(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, alpha=1.0):
        return GradientReversalFunction.apply(x, alpha)


# ═══════════════════════════════════════════════════════
# 模型
# ═══════════════════════════════════════════════════════
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, pool=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=k // 2),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.MaxPool1d(pool),
        )
    def forward(self, x):
        return self.block(x)


class CNNBiLSTMExtractor(nn.Module):
    """特征提取器：(B, S, T) → (B, embed_dim) L2-normalized。"""
    def __init__(self):
        super().__init__()
        ch = cfg.CNN_CHANNELS
        H  = cfg.LSTM_HIDDEN
        E  = cfg.EMBED_DIM
        S  = cfg.N_SUBCARRIERS

        layers, ic = [], S
        for oc in ch:
            layers.append(ConvBlock(ic, oc))
            ic = oc
        self.cnn = nn.Sequential(*layers)

        self.bilstm1 = nn.LSTM(ch[-1], H, batch_first=True, bidirectional=True)
        self.cnn2    = nn.Sequential(
            nn.Conv1d(H * 2, H, 3, padding=1),
            nn.BatchNorm1d(H),
            nn.GELU(),
        )
        self.bilstm2 = nn.LSTM(H, H, batch_first=True, bidirectional=True)
        self.drop    = nn.Dropout(0.4)
        self.fc      = nn.Sequential(
            nn.Linear(H * 2, E),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(E, E),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.bilstm1(x)
        x = self.drop(x)
        x = x.permute(0, 2, 1)
        x = self.cnn2(x)
        x = x.permute(0, 2, 1)
        x, _ = self.bilstm2(x)
        x = self.drop(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


class ActivityClassifier(nn.Module):
    """辅助分类头，加速嵌入空间学习。"""
    def __init__(self, embed_dim, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )
    def forward(self, x):
        return self.fc(x)


class DomainDiscriminator(nn.Module):
    """域判别器：区分源域(0) vs 目标域(1)。"""
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )
    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════════════
# 损失函数
# ═══════════════════════════════════════════════════════
class HardTripletLoss(nn.Module):
    """
    Batch Hard Triplet Loss：
    在同一 batch 中为每个 anchor 找最难的 positive 和 negative。
    这是解决 loss=1.0 塌陷的关键！
    """
    def __init__(self, margin=cfg.MARGIN):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        B = embeddings.size(0)
        # 成对距离矩阵
        dist = torch.cdist(embeddings, embeddings, p=2)  # (B, B)

        # 掩码
        labels = labels.to(embeddings.device)
        same   = labels.unsqueeze(0) == labels.unsqueeze(1)   # (B, B)
        diff   = ~same

        # 对角线排除
        eye = torch.eye(B, dtype=torch.bool, device=embeddings.device)
        same = same & ~eye

        # Hard Positive: 同类中距离最大
        dist_ap = (dist * same.float()).max(dim=1).values   # (B,)
        # Hard Negative: 异类中距离最小（加大数避免0掩蔽）
        dist_an = (dist + (~diff).float() * 1e9).min(dim=1).values  # (B,)

        loss = F.relu(dist_ap - dist_an + self.margin).mean()
        return loss


def mmd_loss(source_feat, target_feat):
    """
    最大均值差异（RBF 核）：拉近源域和目标域的特征分布。
    """
    def rbf_kernel(x, y, sigma=1.0):
        n, m = x.size(0), y.size(0)
        xx = x.unsqueeze(1).expand(n, m, -1)
        yy = y.unsqueeze(0).expand(n, m, -1)
        return torch.exp(-((xx - yy) ** 2).sum(-1) / (2 * sigma ** 2))

    Kss = rbf_kernel(source_feat, source_feat).mean()
    Ktt = rbf_kernel(target_feat, target_feat).mean()
    Kst = rbf_kernel(source_feat, target_feat).mean()
    return Kss + Ktt - 2 * Kst


# ═══════════════════════════════════════════════════════
# 训练循环
# ═══════════════════════════════════════════════════════
def train_epoch(extractor, classifier, domain_disc, grl,
                src_loader, tgt_loader,
                opt_feat, opt_cls, opt_dom,
                triplet_loss, cls_loss_fn,
                device, epoch, total_epochs):

    extractor.train()
    classifier.train()
    domain_disc.train()

    # DANN 的 alpha 随训练进行逐步增大
    p = epoch / total_epochs
    alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

    tgt_iter = iter(tgt_loader)
    total_trip, total_cls, total_dom, total_mmd = 0, 0, 0, 0
    n_batch = 0

    for batch in src_loader:
        xa, xp, xn, labels = batch
        xa, xp, xn = xa.to(device), xp.to(device), xn.to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)

        # 目标域无标签 batch
        try:
            xt = next(tgt_iter).to(device)
        except StopIteration:
            tgt_iter = iter(tgt_loader)
            xt = next(tgt_iter).to(device)

        # ── 特征提取 ──
        ea = extractor(xa)   # anchor embeddings
        ep = extractor(xp)
        en = extractor(xn)
        et = extractor(xt)   # target embeddings

        # 1. Hard Triplet Loss（对 anchor batch 计算）
        all_emb = torch.cat([ea, ep, en], dim=0)
        all_lbl = torch.cat([labels, labels, labels], dim=0)
        l_trip  = triplet_loss(all_emb, all_lbl)

        # 2. 辅助分类损失（仅源域有标签）
        logits = classifier(ea)
        l_cls  = cls_loss_fn(logits, labels)

        # 3. 域对抗损失（DANN）
        n_src, n_tgt = ea.size(0), et.size(0)
        domain_feats  = torch.cat([ea, et], dim=0)
        domain_labels = torch.cat([
            torch.zeros(n_src, dtype=torch.long, device=device),
            torch.ones(n_tgt,  dtype=torch.long, device=device),
        ])
        rev_feats = grl(domain_feats, alpha)
        domain_logits = domain_disc(rev_feats)
        l_dom = cls_loss_fn(domain_logits, domain_labels)

        # 4. MMD Loss
        l_mmd = mmd_loss(ea, et)

        # ── 总损失 ──
        loss = (l_trip
                + cfg.LAMBDA_CLS    * l_cls
                + cfg.LAMBDA_DOMAIN * l_dom
                + cfg.LAMBDA_MMD    * l_mmd)

        opt_feat.zero_grad()
        opt_cls.zero_grad()
        opt_dom.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(extractor.parameters(),    1.0)
        torch.nn.utils.clip_grad_norm_(classifier.parameters(),   1.0)
        torch.nn.utils.clip_grad_norm_(domain_disc.parameters(),  1.0)
        opt_feat.step()
        opt_cls.step()
        opt_dom.step()

        total_trip += l_trip.item()
        total_cls  += l_cls.item()
        total_dom  += l_dom.item()
        total_mmd  += l_mmd.item()
        n_batch    += 1

    return (total_trip / n_batch, total_cls / n_batch,
            total_dom / n_batch, total_mmd / n_batch)


@torch.no_grad()
def val_epoch(extractor, classifier, src_loader, device):
    extractor.eval()
    classifier.eval()
    total_trip, total_acc, n_batch = 0.0, 0.0, 0
    triplet_loss = HardTripletLoss()
    cls_loss_fn  = nn.CrossEntropyLoss()

    for batch in src_loader:
        xa, xp, xn, labels = batch
        xa, xp, xn = xa.to(device), xp.to(device), xn.to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)

        ea = extractor(xa)
        all_emb = torch.cat([ea, extractor(xp), extractor(xn)], dim=0)
        all_lbl = torch.cat([labels, labels, labels], dim=0)
        l_trip = triplet_loss(all_emb, all_lbl)

        logits = classifier(ea)
        acc = (logits.argmax(1) == labels).float().mean()

        total_trip += l_trip.item()
        total_acc  += acc.item()
        n_batch    += 1

    return total_trip / n_batch, total_acc / n_batch


# ═══════════════════════════════════════════════════════
# Prototypical Network 分类器
# ═══════════════════════════════════════════════════════
class ProtoNetClassifier:
    """
    Prototypical Network：每个类的原型 = 支持集嵌入的均值。
    比最近邻分类器更鲁棒。
    """
    def __init__(self, model, device):
        self.model     = model
        self.device    = device
        self.prototypes = None  # (C, E)
        self.proto_labels = None

    @torch.no_grad()
    def build_support(self, samples, n_shot=cfg.N_SHOT):
        self.model.eval()
        cls_pool = defaultdict(list)
        for arr, lbl in samples:
            cls_pool[lbl].append(arr)

        protos, proto_labels = [], []
        for lbl, arrs in sorted(cls_pool.items()):
            chosen = random.sample(arrs, min(n_shot, len(arrs)))
            embeds = []
            for arr in chosen:
                x = torch.tensor(arr.T, dtype=torch.float32).unsqueeze(0).to(self.device)
                e = self.model(x).squeeze(0).cpu().numpy()
                embeds.append(e)
            proto = np.mean(embeds, axis=0)  # 原型 = 均值
            protos.append(proto)
            proto_labels.append(lbl)

        self.prototypes   = np.stack(protos)       # (C, E)
        self.proto_labels = np.array(proto_labels)
        print(f"[支持集] {len(proto_labels)} 个原型 | 类别: {sorted(proto_labels)}")

    @torch.no_grad()
    def predict(self, arr):
        self.model.eval()
        x = torch.tensor(arr.T, dtype=torch.float32).unsqueeze(0).to(self.device)
        e = self.model(x).squeeze(0).cpu().numpy()
        dists = np.sum((self.prototypes - e) ** 2, axis=1)
        return self.proto_labels[np.argmin(dists)]

    def evaluate(self, samples):
        correct, total = 0, 0
        per_class = defaultdict(lambda: [0, 0])
        for arr, lbl in samples:
            pred = self.predict(arr)
            per_class[lbl][1] += 1
            if pred == lbl:
                correct += 1
                per_class[lbl][0] += 1
            total += 1
        acc = correct / total if total else 0.0
        print(f"\n[评估] 准确率: {acc*100:.2f}%  ({correct}/{total})")
        print("各类准确率:")
        for lbl in sorted(per_class):
            c, t = per_class[lbl]
            print(f"  类别 {lbl+1}: {c}/{t} = {c/t*100:.1f}%")
        return acc


# ═══════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════
def main():
    device = cfg.DEVICE
    print(f"设备: {device}\n")

    # ── 1. 加载数据 ──
    print("[1] 加载数据...")
    source_samples = []
    for user in cfg.SOURCE_USERS:
        folder  = os.path.join(cfg.SOURCE_ROOT, user)
        samples = load_folder(folder, cfg.N_SUBCARRIERS, cfg.TIME_STEPS, tag=user)
        source_samples.extend(samples)

    pseudo_samples = load_folder(
        os.path.join(cfg.SOURCE_ROOT, cfg.PSEUDO_USER),
        cfg.N_SUBCARRIERS, cfg.TIME_STEPS, tag=cfg.PSEUDO_USER
    )
    train_pool = source_samples + pseudo_samples
    print(f"\n  源域: {len(source_samples)} | 伪数据: {len(pseudo_samples)} | 训练池: {len(train_pool)}")

    print("\n  加载目标域...")
    target_samples = load_target_domain(cfg.TARGET_ROOT, cfg.N_SUBCARRIERS, cfg.TIME_STEPS)
    print(f"  目标域: {len(target_samples)}")

    if not train_pool:
        raise RuntimeError(f"训练数据为空！SOURCE_ROOT={cfg.SOURCE_ROOT}")
    if not target_samples:
        raise RuntimeError(f"目标域数据为空！TARGET_ROOT={cfg.TARGET_ROOT}")

    # ── 2. 划分数据集 ──
    # 目标域：20% 作支持集，80% 作评估集（不再重叠）
    random.shuffle(target_samples)
    n_support = max(cfg.N_SHOT * cfg.N_CLASSES, int(0.2 * len(target_samples)))
    support_set = target_samples[:n_support]
    eval_set    = target_samples[n_support:]
    print(f"\n  目标域支持集: {len(support_set)} | 评估集: {len(eval_set)}")

    random.shuffle(train_pool)
    split   = int(0.85 * len(train_pool))
    train_s = train_pool[:split]
    val_s   = train_pool[split:]
    print(f"  源域训练: {len(train_s)} | 验证: {len(val_s)}")

    train_dl = DataLoader(TripletDataset(train_s, training=True),
                          batch_size=cfg.BATCH_SIZE, shuffle=True,  num_workers=0, drop_last=True)
    val_dl   = DataLoader(TripletDataset(val_s,   training=False),
                          batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0, drop_last=True)
    tgt_dl   = DataLoader(UnlabeledDataset(target_samples),
                          batch_size=cfg.BATCH_SIZE, shuffle=True,  num_workers=0, drop_last=True)

    # ── 3. 构建模型 ──
    extractor   = CNNBiLSTMExtractor().to(device)
    classifier  = ActivityClassifier(cfg.EMBED_DIM, cfg.N_CLASSES).to(device)
    domain_disc = DomainDiscriminator(cfg.EMBED_DIM).to(device)
    grl         = GradientReversal()
    triplet_loss = HardTripletLoss()
    cls_loss_fn  = nn.CrossEntropyLoss()

    opt_feat = torch.optim.AdamW(extractor.parameters(),   lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    opt_cls  = torch.optim.AdamW(classifier.parameters(),  lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    opt_dom  = torch.optim.AdamW(domain_disc.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    sched_feat = torch.optim.lr_scheduler.CosineAnnealingLR(opt_feat, T_max=cfg.EPOCHS)
    sched_cls  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_cls,  T_max=cfg.EPOCHS)

    n_params = sum(p.numel() for p in extractor.parameters() if p.requires_grad)
    print(f"\n[3] 特征提取器参数量: {n_params:,}")

    # ── 4. 训练 ──
    print("\n[4] 开始训练...")
    best_val_trip = float('inf')
    patience_cnt  = 0
    best_weights  = None

    for epoch in range(1, cfg.EPOCHS + 1):
        tr_trip, tr_cls, tr_dom, tr_mmd = train_epoch(
            extractor, classifier, domain_disc, grl,
            train_dl, tgt_dl,
            opt_feat, opt_cls, opt_dom,
            triplet_loss, cls_loss_fn,
            device, epoch, cfg.EPOCHS
        )
        vl_trip, vl_acc = val_epoch(extractor, classifier, val_dl, device)
        sched_feat.step()
        sched_cls.step()

        if vl_trip < best_val_trip:
            best_val_trip = vl_trip
            patience_cnt  = 0
            best_weights  = {k: v.clone() for k, v in extractor.state_dict().items()}
            torch.save(best_weights, "best_recognition_model.pth")
        else:
            patience_cnt += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{cfg.EPOCHS} | "
                  f"Trip: {tr_trip:.4f} | Cls: {tr_cls:.4f} | "
                  f"Dom: {tr_dom:.4f} | MMD: {tr_mmd:.4f} | "
                  f"Val_Trip: {vl_trip:.4f} | Val_Acc: {vl_acc*100:.1f}% | "
                  f"LR: {sched_feat.get_last_lr()[0]:.2e}")

        if patience_cnt >= cfg.PATIENCE:
            print(f"  提前停止 (epoch {epoch})")
            break

    extractor.load_state_dict(best_weights)
    print(f"\n最佳验证 Triplet Loss: {best_val_trip:.4f}")
    print("模型已保存: best_recognition_model.pth")

    # ── 5. 构建支持集 ──
    print("\n[5] 用目标域支持集构建原型...")
    clf = ProtoNetClassifier(extractor, device)
    clf.build_support(support_set, n_shot=cfg.N_SHOT)

    # ── 6. 在独立评估集上测试 ──
    print("\n[6] 在目标域评估集上测试...")
    if len(eval_set) == 0:
        print("[WARN] 评估集为空，改用全部目标域数据")
        acc = clf.evaluate(target_samples)
    else:
        acc = clf.evaluate(eval_set)

    print(f"\n✓ 跨域活动识别准确率: {acc*100:.2f}%")
    return extractor, clf


if __name__ == "__main__":
    main()