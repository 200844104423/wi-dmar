import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import scipy.io as sio
import os
import math
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter
import warnings
import seaborn as sns

# 导入第一个文件中的所有模块（移除两个绘图函数）
from zengqiang759 import (
    set_seed, AdvancedDataAugmentation, ImprovedEMAModel,
    SuperEnhancedConditionalDiffusionCore, EnhancedActionClassifier,
    EnhancedDomainClassifier, AdvancedDiffusionScheduler, FocalLoss,
    compute_advanced_diffusion_loss, AdvancedEarlyStopping, CSIDataset,
    SuperActionClassifierForFiltering, SuperGeneratedDataQualityChecker,
    EnhancedStochasticDepth
)

warnings.filterwarnings('ignore')


# ==================== 绘制增强的混淆矩阵 ====================
def plot_enhanced_confusion_matrix(y_true, y_pred, classes, save_path='confusion_matrix.png'):
    """绘制增强的混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    # 计算归一化混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 原始混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'}, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=16)
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)

    # 归一化混淆矩阵
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Percentage'}, ax=ax2, vmin=0, vmax=1)
    ax2.set_title('Normalized Confusion Matrix', fontsize=16)
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

    # 计算每个类别的召回率和精确率
    recalls = cm.diagonal() / cm.sum(axis=1)
    precisions = cm.diagonal() / cm.sum(axis=0)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

    print("\nPer-class metrics:")
    for i, (recall, precision, f1) in enumerate(zip(recalls, precisions, f1_scores)):
        print(f"  Action {i}: Recall={recall:.2%}, Precision={precision:.2%}, F1={f1:.3f}")

    # 计算整体指标
    accuracy = cm.diagonal().sum() / cm.sum()
    macro_f1 = f1_scores.mean()
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Macro F1: {macro_f1:.3f}")


# ==================== 绘制增强的训练曲线 ====================
def plot_enhanced_training_curves(train_history, num_actions):
    """绘制增强的训练曲线"""
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Loss曲线
    ax1 = fig.add_subplot(gs[0, 0])
    if 'total_loss' in train_history and len(train_history['total_loss']) > 0:
        ax1.plot(train_history['total_loss'], label='Total Loss', linewidth=2, color='blue')
        ax1.fill_between(range(len(train_history['total_loss'])),
                         train_history['total_loss'], alpha=0.3)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 各项损失
    ax2 = fig.add_subplot(gs[0, 1])
    if 'diff_loss' in train_history and len(train_history.get('diff_loss', [])) > 0:
        ax2.plot(train_history.get('diff_loss', []), label='Diffusion Loss', alpha=0.7)
        ax2.plot(train_history.get('action_loss', []), label='Action Loss', alpha=0.7)
        ax2.plot(train_history.get('domain_loss', []), label='Domain Loss', alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Individual Losses', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # 验证准确率
    ax3 = fig.add_subplot(gs[1, 0])
    if len(train_history.get('val_action_acc', [])) > 0:
        epochs = np.arange(2, len(train_history['val_action_acc']) * 2 + 1, 2)
        ax3.plot(epochs, train_history['val_action_acc'], 'o-',
                 label='Action Acc', linewidth=2, markersize=6, color='green')
        if 'val_domain_acc' in train_history:
            ax3.plot(epochs, train_history['val_domain_acc'], 's-',
                     label='Domain Acc', linewidth=2, markersize=6, color='orange')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_title('Validation Accuracy', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Per-class准确率演变
    ax4 = fig.add_subplot(gs[1, 1])
    if 'per_class_acc' in train_history and len(train_history['per_class_acc']) > 0:
        per_class_data = train_history['per_class_acc']
        if len(per_class_data) > 0 and isinstance(per_class_data[0], dict):
            classes_to_plot = sorted(per_class_data[0].keys())[:10]  # 最多显示10个类

            for cls in classes_to_plot:
                cls_accs = [epoch_data.get(cls, 0) for epoch_data in per_class_data]
                epochs = np.arange(2, len(cls_accs) * 2 + 1, 2)
                ax4.plot(epochs, cls_accs, 'o-', label=f'Act {cls}',
                         alpha=0.7, markersize=4)

            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Accuracy (%)')
            ax4.set_title('Per-Class Validation Accuracy', fontsize=14)
            ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                       fontsize=8, ncol=1)
            ax4.grid(True, alpha=0.3)

    # 学习率变化（如果有的话）
    ax5 = fig.add_subplot(gs[2, :])
    if 'lr' in train_history and len(train_history['lr']) > 0:
        ax5.plot(train_history['lr'], linewidth=2, color='red')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Learning Rate')
        ax5.set_title('Learning Rate Schedule', fontsize=14)
        ax5.set_yscale('log')
        ax5.grid(True, alpha=0.3)

    plt.suptitle('Enhanced Diffusion Model Training Progress', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('training_curves_super_enhanced.png', dpi=150, bbox_inches='tight')
    plt.show()


# ==================== 增强的数据质量筛选函数 ====================
def super_filter_source_data(dataset, device, confidence_threshold=0.65,
                             num_actions=10, batch_size=32, num_epochs=25):
    """
    超级增强的数据质量筛选，使用更好的分类器和多轮训练
    """
    print("\n" + "=" * 60)
    print("SUPER ENHANCED DATA QUALITY FILTERING FOR SOURCE DOMAIN")
    print("=" * 60)

    # 创建超级分类器
    classifier = SuperActionClassifierForFiltering(
        input_dim=27,
        num_actions=num_actions,
        dropout=0.15
    ).to(device)

    # 准备数据加载器
    def collate_fn(batch):
        max_len = min(max([item[0].shape[0] for item in batch]), 990)
        padded_csi = []
        actions = []

        for csi, action, domain in batch:
            if csi.shape[0] < max_len:
                padding = np.zeros((max_len - csi.shape[0], 27))
                csi = np.vstack([csi, padding])
            elif csi.shape[0] > max_len:
                csi = csi[:max_len]

            padded_csi.append(csi)
            actions.append(action)

        return {
            'csi': torch.stack([torch.tensor(csi, dtype=torch.float32) for csi in padded_csi]),
            'action': torch.tensor(actions, dtype=torch.long)
        }

    # 准备训练数据
    train_data = list(zip(dataset.data, dataset.labels, dataset.domains))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 训练分类器
    print("Training super classifier for data filtering...")
    optimizer = optim.AdamW(classifier.parameters(), lr=5e-4, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    # 使用Focal Loss处理类别不平衡
    label_counts = Counter(dataset.labels)
    class_weights = compute_class_weight('balanced',
                                         classes=np.unique(dataset.labels),
                                         y=dataset.labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    best_acc = 0
    augmenter = AdvancedDataAugmentation()

    for epoch in range(num_epochs):
        classifier.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            csi_data = batch['csi'].to(device)
            action_labels = batch['action'].to(device)

            # 数据增强
            if epoch < num_epochs // 2:  # 前半程使用数据增强
                csi_np = csi_data.cpu().numpy()
                for i in range(csi_data.size(0)):
                    if random.random() > 0.5:
                        csi_np[i] = augmenter.combined_augmentation(csi_np[i], augment_prob=0.3)
                csi_data = torch.tensor(csi_np, dtype=torch.float32).to(device)

            optimizer.zero_grad()
            outputs = classifier(csi_data)
            loss = criterion(outputs, action_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += action_labels.size(0)
            correct += predicted.eq(action_labels).sum().item()

        acc = 100. * correct / total
        scheduler.step()

        if acc > best_acc:
            best_acc = acc

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/{num_epochs}: Loss={total_loss / len(train_loader):.4f}, Acc={acc:.2f}%")

    print(f"Best training accuracy: {best_acc:.2f}%")

    # 使用训练好的分类器筛选数据
    print(f"\nFiltering data with confidence threshold: {confidence_threshold}")
    classifier.eval()

    filtered_data = []
    filtered_labels = []
    filtered_domains = []
    filtered_formats = []

    confidence_scores = []
    kept_count = 0
    removed_count = 0

    # 统计每个类别的保留情况
    class_kept = {}
    class_removed = {}
    class_confidences = {}

    with torch.no_grad():
        for i, (csi, label, domain) in enumerate(zip(dataset.data, dataset.labels, dataset.domains)):
            # 准备输入
            csi_tensor = torch.tensor(csi, dtype=torch.float32).unsqueeze(0).to(device)

            # 获取预测
            output = classifier(csi_tensor)
            probs = F.softmax(output, dim=1)
            max_prob, predicted = probs.max(1)

            confidence = max_prob.item()
            confidence_scores.append(confidence)

            # 记录每个类别的置信度
            if label not in class_confidences:
                class_confidences[label] = []
            class_confidences[label].append(confidence)

            # 动态调整阈值（对于样本较少的类别降低阈值）
            class_count = sum(1 for l in dataset.labels if l == label)
            adjusted_threshold = confidence_threshold
            if class_count < 20:  # 如果该类别样本少于20个
                adjusted_threshold = confidence_threshold * 0.8  # 降低20%的阈值

            # 如果预测正确且置信度高，保留样本
            if predicted.item() == label and confidence >= adjusted_threshold:
                filtered_data.append(csi)
                filtered_labels.append(label)
                filtered_domains.append(domain)
                if i < len(dataset.original_data_format):
                    filtered_formats.append(dataset.original_data_format[i])

                kept_count += 1
                class_kept[label] = class_kept.get(label, 0) + 1
            else:
                removed_count += 1
                class_removed[label] = class_removed.get(label, 0) + 1

    # 检查是否有类别被完全过滤掉
    unique_labels_before = set(dataset.labels)
    unique_labels_after = set(filtered_labels)

    if len(unique_labels_after) < len(unique_labels_before):
        print(f"⚠️ Warning: Some classes were completely filtered out!")
        missing_classes = unique_labels_before - unique_labels_after

        # 为被完全过滤的类别恢复至少一些样本
        for missing_class in missing_classes:
            class_indices = [i for i, l in enumerate(dataset.labels) if l == missing_class]
            if class_indices:
                # 选择置信度最高的几个样本
                class_data = [(i, dataset.data[i], dataset.labels[i], dataset.domains[i])
                              for i in class_indices[:min(5, len(class_indices))]]

                for idx, data, label, domain in class_data:
                    filtered_data.append(data)
                    filtered_labels.append(label)
                    filtered_domains.append(domain)
                    if idx < len(dataset.original_data_format):
                        filtered_formats.append(dataset.original_data_format[idx])
                    kept_count += 1
                    removed_count -= 1

                print(f"  Restored {len(class_data)} samples for class {missing_class}")

    # 更新数据集
    original_count = len(dataset.data)
    dataset.data = filtered_data
    dataset.labels = filtered_labels
    dataset.domains = filtered_domains
    dataset.original_data_format = filtered_formats

    # 打印筛选结果
    print(f"\n📊 Filtering Results:")
    print(f"  Original samples: {original_count}")
    print(f"  Kept samples: {kept_count} ({100 * kept_count / original_count:.1f}%)")
    print(f"  Removed samples: {removed_count} ({100 * removed_count / original_count:.1f}%)")
    print(f"  Average confidence: {np.mean(confidence_scores):.3f}")
    print(f"  Confidence range: [{np.min(confidence_scores):.3f}, {np.max(confidence_scores):.3f}]")

    print("\n  Per-class filtering statistics:")
    all_classes = set(list(class_kept.keys()) + list(class_removed.keys()))
    for cls in sorted(all_classes):
        kept = class_kept.get(cls, 0)
        removed = class_removed.get(cls, 0)
        total = kept + removed
        if total > 0:
            avg_conf = np.mean(class_confidences[cls]) if cls in class_confidences else 0
            print(f"    Action {cls}: kept {kept}/{total} ({100 * kept / total:.1f}%), avg conf: {avg_conf:.3f}")

    return dataset


# ==================== 高级DDIM采样 ====================
def advanced_ddim_sampling(model, scheduler, shape, action, domain, device,
                           eta=0.0, num_inference_steps=50, guidance_scale=2.0,
                           temperature=1.0):
    """高级DDIM采样with增强的引导机制"""

    # 初始化纯噪声
    x = torch.randn(shape, device=device) * temperature

    # 设置推理时间步
    timesteps = torch.linspace(scheduler.num_timesteps - 1, 0, num_inference_steps, device=device).long()

    model.eval()
    with torch.no_grad():
        for i, t in enumerate(tqdm(timesteps, desc="Advanced DDIM Sampling")):
            t_tensor = t.float().expand(shape[0])

            # 预测噪声
            predicted_noise, _ = model(x, t_tensor, action, domain)

            # Classifier-free guidance with dynamic scaling
            if guidance_scale > 1.0:
                # 动态调整引导强度
                dynamic_scale = guidance_scale * (1.0 - i / len(timesteps))

                # 无条件预测
                uncond_noise, _ = model(x, t_tensor,
                                        torch.zeros_like(action),
                                        torch.zeros_like(domain))
                # 引导
                predicted_noise = uncond_noise + dynamic_scale * (predicted_noise - uncond_noise)

            # DDIM更新步骤
            alpha_prod_t = scheduler.alphas_cumprod[t].to(device)
            alpha_prod_t_prev = scheduler.alphas_cumprod[timesteps[i + 1]].to(device) if i < len(
                timesteps) - 1 else torch.tensor(1.0).to(device)

            # 预测x0
            pred_x0 = (x - torch.sqrt(1 - alpha_prod_t) * predicted_noise) / torch.sqrt(alpha_prod_t)

            # 裁剪预测值
            pred_x0 = torch.clamp(pred_x0, -1, 1)

            # 方向指向x_t
            dir_xt = torch.sqrt(1 - alpha_prod_t_prev - eta ** 2) * predicted_noise

            # 添加噪声
            noise = torch.randn_like(x) if eta > 0 and i < len(timesteps) - 1 else 0

            # 计算x_{t-1}
            x = torch.sqrt(alpha_prod_t_prev) * pred_x0 + dir_xt + eta * noise

    return x


# ==================== 超级高质量数据生成函数 ====================
def generate_super_high_quality_data(model, scheduler, device, num_samples=50,
                                     seq_len=990, action_type=0, domain_id=0,
                                     save_path='generated_data', quality_checker=None,
                                     use_ddim=True, guidance_scale=2.0,
                                     action_classifier=None, confidence_threshold=0.75,
                                     temperature=1.0):
    """
    生成超高质量数据with quality filtering and action confidence checking
    """

    model.eval()
    if action_classifier is not None:
        action_classifier.eval()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    generated_samples = []
    quality_scores = []
    confidence_scores = []

    # 统计生成和筛选情况
    total_generated = 0
    rejected_by_quality = 0
    rejected_by_confidence = 0

    print(f"\nGenerating high-quality data for action {action_type + 1} with strict filtering...")
    if action_classifier is not None:
        print(f"  Action confidence threshold: {confidence_threshold}")

    with torch.no_grad():
        # 生成更多样本以确保筛选后有足够数量
        max_attempts = num_samples * 4  # 最多尝试4倍数量
        pbar = tqdm(range(max_attempts), desc=f'Generating action {action_type + 1}')

        for i in pbar:
            if len(generated_samples) >= num_samples:
                break

            action = torch.tensor([action_type], device=device)
            domain = torch.tensor([domain_id], device=device)

            if use_ddim:
                # 使用高级DDIM采样
                x = advanced_ddim_sampling(
                    model, scheduler,
                    shape=(1, seq_len, 27),
                    action=action,
                    domain=domain,
                    device=device,
                    eta=0.0,
                    num_inference_steps=50,
                    guidance_scale=guidance_scale,
                    temperature=temperature
                )
            else:
                # 标准DDPM采样
                x = torch.randn(1, seq_len, 27, device=device) * temperature

                for t in reversed(range(scheduler.num_timesteps)):
                    timestep = torch.tensor([t], device=device, dtype=torch.float)
                    predicted_noise, _ = model(x, timestep, action, domain)

                    if t > 0:
                        alpha = scheduler.alphas[t]
                        alpha_cumprod = scheduler.alphas_cumprod[t]
                        beta = scheduler.betas[t]

                        x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise)

                        if t > 1:
                            noise = torch.randn_like(x) * 0.1
                            x = x + torch.sqrt(beta) * noise
                    else:
                        alpha_cumprod = scheduler.alphas_cumprod[t]
                        x = (1 / torch.sqrt(scheduler.alphas[t])) * (
                                x - (scheduler.betas[t] / torch.sqrt(1 - alpha_cumprod)) * predicted_noise)

            generated_data = x.squeeze(0).cpu().numpy()
            total_generated += 1

            # 第一步：质量评估（如果提供了quality_checker）
            quality_pass = True
            if quality_checker:
                scores = quality_checker.evaluate_quality(generated_data)
                quality_scores.append(scores['overall_score'])

                # 动态调整质量阈值
                quality_threshold = 0.55 if not quality_checker.strict_mode else 0.65

                if scores['overall_score'] < quality_threshold:
                    quality_pass = False
                    rejected_by_quality += 1

            # 第二步：动作特征置信度评估（如果提供了action_classifier）
            confidence_pass = True
            if action_classifier is not None and quality_pass:
                # 将生成的数据送入编码器获取特征
                x_tensor = torch.tensor(generated_data, dtype=torch.float32).unsqueeze(0).to(device)

                # 使用模型编码器提取特征
                _, cls_token = model.encode(x_tensor, domain)

                # 使用动作分类器评估
                action_logits = action_classifier(cls_token)
                action_probs = F.softmax(action_logits, dim=1)

                # 获取预测的动作和置信度
                max_prob, predicted_action = action_probs.max(1)
                confidence = max_prob.item()
                confidence_scores.append(confidence)

                # 检查预测是否正确且置信度足够高
                if predicted_action.item() != action_type or confidence < confidence_threshold:
                    confidence_pass = False
                    rejected_by_confidence += 1

                # 更新进度条信息
                pbar.set_postfix({
                    'kept': len(generated_samples),
                    'conf': f'{confidence:.3f}',
                    'pred': predicted_action.item()
                })

            # 如果通过所有检查，保留样本
            if quality_pass and confidence_pass:
                generated_samples.append(generated_data)

    # 打印生成统计
    print(f"\n📊 Generation Statistics for Action {action_type + 1}:")
    print(f"  Total generated: {total_generated}")
    print(f"  Kept samples: {len(generated_samples)} ({100 * len(generated_samples) / max(total_generated, 1):.1f}%)")

    if quality_checker:
        print(
            f"  Rejected by quality: {rejected_by_quality} ({100 * rejected_by_quality / max(total_generated, 1):.1f}%)")
        if len(quality_scores) > 0:
            avg_quality = np.mean(quality_scores)
            print(f"  Average quality score: {avg_quality:.3f}")

    if action_classifier is not None:
        print(
            f"  Rejected by confidence: {rejected_by_confidence} ({100 * rejected_by_confidence / max(total_generated, 1):.1f}%)")
        if len(confidence_scores) > 0:
            avg_confidence = np.mean(confidence_scores)
            print(f"  Average confidence score: {avg_confidence:.3f}")
            print(f"  Confidence range: [{np.min(confidence_scores):.3f}, {np.max(confidence_scores):.3f}]")

    # 保存生成的数据
    for i, data in enumerate(generated_samples):
        # 归一化
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        data = data * 20 - 10

        # 添加相位
        phase = np.random.uniform(-np.pi, np.pi, data.shape)
        complex_csi = data * np.exp(1j * phase)
        complex_csi_transposed = complex_csi.T

        filename = f'user-{action_type + 1}-0-0-{i}-R1.mat'
        save_file = os.path.join(save_path, filename)
        sio.savemat(save_file, {'csi': complex_csi_transposed})

    print(f'✅ Successfully saved {len(generated_samples)} high-quality samples for action {action_type + 1}')

    # 如果生成的样本太少，给出警告
    if len(generated_samples) < num_samples * 0.5:
        print(f"⚠️ Warning: Only generated {len(generated_samples)}/{num_samples} requested samples.")
        print("  Consider adjusting confidence threshold or guidance scale.")

    return generated_samples


# ==================== 超级预训练动作分类器 ====================
def super_pretrain_action_classifier(model, action_classifier, dataloader, device,
                                     class_weights=None, num_epochs=20):
    """
    使用真实编码器特征预训练动作分类器，处理类别不平衡
    """
    print("=" * 60)
    print("SUPER PRE-TRAINING ACTION CLASSIFIER WITH REAL ENCODER FEATURES")
    print("=" * 60)

    # 冻结编码器，只训练分类器
    model.eval()
    action_classifier.train()

    optimizer = optim.AdamW(action_classifier.parameters(), lr=1e-3, weight_decay=5e-4)

    # 使用余弦退火重启
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    # 使用Focal Loss处理类别不平衡
    if class_weights is not None:
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        print(f"Using Focal Loss with class weights")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0
    data_augmenter = AdvancedDataAugmentation()

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        # 统计每个类别的预测
        class_correct = {}
        class_total = {}

        pbar = tqdm(dataloader, desc=f'Pre-train Epoch {epoch + 1}/{num_epochs}')

        for batch in pbar:
            csi_data = batch['csi'].to(device)
            action_labels = batch['action'].to(device)
            domain_labels = batch['domain'].to(device)

            # 数据增强
            if random.random() > 0.5 and epoch < num_epochs // 2:  # 前半程增强
                csi_np = csi_data.cpu().numpy()
                for i in range(csi_data.size(0)):
                    csi_np[i] = data_augmenter.combined_augmentation(csi_np[i], augment_prob=0.3)
                csi_data = torch.tensor(csi_np, dtype=torch.float32).to(device)

            # 使用真实编码器提取clean特征
            with torch.no_grad():
                _, cls_clean = model.encode(csi_data, domain_labels)

            # 分类器前向传播
            outputs = action_classifier(cls_clean)
            loss = criterion(outputs, action_labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(action_classifier.parameters(), max_norm=0.5)
            optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += action_labels.size(0)
            correct += predicted.eq(action_labels).sum().item()

            # 统计每个类别
            for label in action_labels.unique():
                mask = action_labels == label
                if label.item() not in class_correct:
                    class_correct[label.item()] = 0
                    class_total[label.item()] = 0
                class_correct[label.item()] += predicted[mask].eq(action_labels[mask]).sum().item()
                class_total[label.item()] += mask.sum().item()

            # 更新进度条
            acc = 100. * correct / total
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{acc:.2f}%'})

        epoch_acc = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        scheduler.step()

        print(f'\nEpoch {epoch + 1}: Loss={avg_loss:.4f}, Accuracy={epoch_acc:.2f}%')

        # 打印每个类别的准确率
        print("Per-class accuracy:")
        min_acc = 100
        for cls in sorted(class_correct.keys()):
            cls_acc = 100. * class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
            min_acc = min(min_acc, cls_acc)
            print(f"  Action {cls}: {cls_acc:.2f}% ({class_correct[cls]}/{class_total[cls]})")

        if epoch_acc > best_acc:
            best_acc = epoch_acc

    print(f"Pre-training completed! Best accuracy: {best_acc:.2f}%")
    return best_acc > 20  # 提高阈值


# ==================== 验证函数 ====================
def validate_model_detailed(model, domain_classifier, action_classifier,
                            val_dataloader, scheduler, device):
    """详细验证模型性能"""
    model.eval()
    domain_classifier.eval()
    action_classifier.eval()

    all_action_preds = []
    all_action_labels = []
    all_domain_preds = []
    all_domain_labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            csi_data = batch['csi'].to(device)
            action_labels = batch['action'].to(device)
            domain_labels = batch['domain'].to(device)

            seq_clean, cls_clean = model.encode(csi_data, domain_labels)

            action_pred = action_classifier(cls_clean)
            _, action_predicted = action_pred.max(1)

            all_action_preds.extend(action_predicted.cpu().numpy())
            all_action_labels.extend(action_labels.cpu().numpy())

            domain_pred = domain_classifier(seq_clean)
            _, domain_predicted = domain_pred.max(1)

            all_domain_preds.extend(domain_predicted.cpu().numpy())
            all_domain_labels.extend(domain_labels.cpu().numpy())

    model.train()
    domain_classifier.train()
    action_classifier.train()

    from sklearn.metrics import f1_score, recall_score

    action_acc = accuracy_score(all_action_labels, all_action_preds) * 100
    domain_acc = accuracy_score(all_domain_labels, all_domain_preds) * 100
    macro_f1 = f1_score(all_action_labels, all_action_preds, average='macro')

    unique_classes = np.unique(all_action_labels)
    per_class_recall = {}
    per_class_acc = {}

    for cls in unique_classes:
        mask = np.array(all_action_labels) == cls
        if mask.sum() > 0:
            recall = recall_score([cls] * mask.sum(),
                                  np.array(all_action_preds)[mask],
                                  labels=[cls], average='micro')
            per_class_recall[cls] = recall * 100

            acc = accuracy_score(np.array(all_action_labels)[mask],
                                 np.array(all_action_preds)[mask])
            per_class_acc[cls] = acc * 100

    return {
        'action_acc': action_acc,
        'domain_acc': domain_acc,
        'macro_f1': macro_f1,
        'per_class_recall': per_class_recall,
        'per_class_acc': per_class_acc
    }


# ==================== 超级增强的训练函数 ====================
def train_model_super_enhanced(model, domain_classifier, action_classifier,
                               train_dataloader, val_dataloader,
                               scheduler, device, class_weights=None,
                               num_epochs=100, use_ema=True):
    """超级增强的训练流程with EMA and improved loss"""

    # 第一阶段：预训练动作分类器
    print("\n" + "=" * 60)
    print("PHASE 1: SUPER PRE-TRAINING ACTION CLASSIFIER")
    print("=" * 60)

    classifier_ready = super_pretrain_action_classifier(
        model, action_classifier, train_dataloader, device,
        class_weights=class_weights, num_epochs=20
    )

    if not classifier_ready:
        print("⚠️ Warning: Pre-training accuracy is low, but continuing...")

    # 第二阶段：联合训练
    print("\n" + "=" * 60)
    print("PHASE 2: SUPER JOINT TRAINING WITH ADVANCED TECHNIQUES")
    print("=" * 60)

    # 初始化EMA
    ema_model = ImprovedEMAModel(model, decay=0.999, decay_warmup_steps=500) if use_ema else None

    # 优化器 - 使用不同的学习率
    optimizer = optim.AdamW([
        {'params': model.parameters(), 'lr': 1e-4, 'weight_decay': 1e-3},
        {'params': domain_classifier.parameters(), 'lr': 5e-5, 'weight_decay': 1e-3},
        {'params': action_classifier.parameters(), 'lr': 2e-4, 'weight_decay': 1e-3}
    ])

    # 学习率调度器 - 使用余弦退火重启
    scheduler_lr = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # 损失函数
    if class_weights is not None:
        action_criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        print(f"Using Focal Loss with class weights")
    else:
        action_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    domain_criterion = nn.CrossEntropyLoss()

    # 早停
    early_stopping = AdvancedEarlyStopping(patience=40, min_delta=0.001, monitor_metric='val_acc')

    # 数据增强器
    data_augmenter = AdvancedDataAugmentation()

    # 设置为训练模式
    model.train()
    domain_classifier.train()
    action_classifier.train()

    # 训练历史
    train_history = {
        'total_loss': [], 'diff_loss': [], 'domain_loss': [],
        'action_loss': [], 'val_action_acc': [], 'val_domain_acc': [],
        'per_class_acc': [], 'lr': []
    }

    best_val_acc = 0
    best_f1 = 0

    # 设置Stochastic Depth的epoch
    for module in model.modules():
        if isinstance(module, EnhancedStochasticDepth):
            module.max_epochs = num_epochs

    for epoch in range(num_epochs):
        # 更新Stochastic Depth
        for module in model.modules():
            if isinstance(module, EnhancedStochasticDepth):
                module.set_epoch(epoch, num_epochs)

        # DANN调度：更温和的调整
        p = float(epoch) / num_epochs
        lambda_grl = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0
        domain_classifier.grl.lambda_ = lambda_grl * 0.02  # 大幅降低域对抗强度

        # 动态权重调整 - 更注重动作分类
        if epoch < 20:
            coef_diff = 0.5
            coef_action = 2.0
            coef_domain = 0.01
        elif epoch < 50:
            coef_diff = 0.2
            coef_action = 5.0
            coef_domain = 0.02
        else:
            coef_diff = 0.1
            coef_action = 10.0
            coef_domain = 0.01

        # 训练指标
        total_loss = 0
        total_diff_loss = 0
        total_action_loss = 0
        total_domain_loss = 0

        action_correct = 0
        action_total = 0

        # 统计每个类别
        class_correct = {}
        class_total = {}

        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for batch_idx, batch in enumerate(pbar):
            csi_data = batch['csi'].to(device)
            action_labels = batch['action'].to(device)
            domain_labels = batch['domain'].to(device)

            batch_size = csi_data.size(0)

            # ========== 高级数据增强 ==========
            if random.random() > 0.3 and epoch < 60:  # 前60个epoch做增强
                csi_np = csi_data.cpu().numpy()
                for i in range(batch_size):
                    # 使用组合增强
                    csi_np[i] = data_augmenter.combined_augmentation(
                        csi_np[i], augment_prob=0.5 * (1 - epoch / 60)
                    )

                    # Mixup or CutMix
                    if i < batch_size - 1 and random.random() > 0.7:
                        if random.random() > 0.5:
                            csi_np[i] = data_augmenter.mixup(csi_np[i], csi_np[i + 1], alpha=0.2)
                        else:
                            csi_np[i] = data_augmenter.cutmix(csi_np[i], csi_np[i + 1], alpha=1.0)

                csi_data = torch.tensor(csi_np, dtype=torch.float32).to(device)

            # ========== 扩散分支（降低强度）==========
            timesteps = torch.randint(0, min(30, scheduler.num_timesteps), (batch_size,), device=device)
            noise = torch.randn_like(csi_data) * 0.05  # 大幅降低噪声强度
            noisy_csi = scheduler.add_noise(csi_data, noise, timesteps)

            # 扩散前向传播
            predicted_noise, seq_noisy = model(noisy_csi, timesteps.float(), action_labels, domain_labels)
            diff_loss = compute_advanced_diffusion_loss(predicted_noise, noise, timesteps,
                                                        gamma=0.5, structure_weight=0.3)

            # ========== 分类分支（clean）==========
            # 使用clean输入提取特征
            seq_clean, cls_clean = model.encode(csi_data, domain_labels)

            # 动作分类（使用clean CLS）
            action_pred = action_classifier(cls_clean)
            action_loss = action_criterion(action_pred, action_labels)

            # 域分类（使用noisy序列特征）
            domain_pred = domain_classifier(seq_noisy.detach())  # detach防止梯度回传
            domain_loss = domain_criterion(domain_pred, domain_labels)

            # ========== 总损失 ==========
            total_batch_loss = (
                    coef_diff * diff_loss +
                    coef_action * action_loss +
                    coef_domain * domain_loss
            )

            # 反向传播
            optimizer.zero_grad()
            total_batch_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(action_classifier.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(domain_classifier.parameters(), max_norm=1.0)

            optimizer.step()

            # 更新EMA
            if ema_model:
                ema_model.update()

            # 统计
            total_loss += total_batch_loss.item()
            total_diff_loss += diff_loss.item()
            total_action_loss += action_loss.item()
            total_domain_loss += domain_loss.item()

            _, predicted = action_pred.max(1)
            action_total += action_labels.size(0)
            action_correct += predicted.eq(action_labels).sum().item()

            # 统计每个类别
            for label in action_labels.unique():
                mask = action_labels == label
                if label.item() not in class_correct:
                    class_correct[label.item()] = 0
                    class_total[label.item()] = 0
                class_correct[label.item()] += predicted[mask].eq(action_labels[mask]).sum().item()
                class_total[label.item()] += mask.sum().item()

            # 更新进度条
            current_acc = 100. * action_correct / action_total
            pbar.set_postfix({
                'Loss': f'{total_batch_loss.item():.3f}',
                'Act': f'{action_loss.item():.3f}',
                'Acc': f'{current_acc:.1f}%'
            })

        # 计算epoch平均值
        num_batches = len(train_dataloader)
        avg_loss = total_loss / num_batches
        avg_action_acc = 100. * action_correct / action_total

        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        train_history['lr'].append(current_lr)

        train_history['total_loss'].append(avg_loss)
        train_history['diff_loss'].append(total_diff_loss / num_batches)
        train_history['action_loss'].append(total_action_loss / num_batches)
        train_history['domain_loss'].append(total_domain_loss / num_batches)

        print(f'\n📊 Epoch {epoch + 1} Summary:')
        print(f'  Total Loss: {avg_loss:.4f}')
        print(f'  Action Accuracy: {avg_action_acc:.2f}%')
        print(f'  Learning Rate: {current_lr:.6f}')

        # 打印每个类别的准确率
        print("  Per-class accuracy:")
        min_class_acc = 100
        for cls in sorted(class_correct.keys()):
            cls_acc = 100. * class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
            min_class_acc = min(min_class_acc, cls_acc)
            print(f"    Action {cls}: {cls_acc:.2f}% ({class_correct[cls]}/{class_total[cls]})")

        # 验证（每2个epoch）
        if (epoch + 1) % 2 == 0 and val_dataloader is not None:
            # 使用EMA模型验证
            if ema_model:
                ema_model.apply_shadow()

            val_metrics = validate_model_detailed(
                model, domain_classifier, action_classifier,
                val_dataloader, scheduler, device
            )

            if ema_model:
                ema_model.restore()

            train_history['val_action_acc'].append(val_metrics['action_acc'])
            train_history['val_domain_acc'].append(val_metrics['domain_acc'])
            train_history['per_class_acc'].append(val_metrics['per_class_acc'])

            print(f'\n✅ Validation Results:')
            print(f'  Action Accuracy: {val_metrics["action_acc"]:.2f}%')
            print(f'  Domain Accuracy: {val_metrics["domain_acc"]:.2f}%')
            print(f'  Macro F1 Score: {val_metrics["macro_f1"]:.4f}')

            # 打印验证集每个类别的召回率
            print("  Per-class recall:")
            for cls, recall in val_metrics['per_class_recall'].items():
                print(f"    Action {cls}: {recall:.2f}%")

            # 调整学习率
            scheduler_lr.step()

            # 早停检查
            early_stopping(val_metrics['action_acc'], model, domain_classifier, action_classifier, epoch)

            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}!")
                print(f"Best epoch was {early_stopping.best_epoch + 1}")
                # 恢复最佳模型
                model.load_state_dict(early_stopping.best_state['model'])
                domain_classifier.load_state_dict(early_stopping.best_state['domain_classifier'])
                action_classifier.load_state_dict(early_stopping.best_state['action_classifier'])
                break

            if val_metrics['macro_f1'] > best_f1:
                best_f1 = val_metrics['macro_f1']
                best_val_acc = val_metrics['action_acc']
                print(f'  🎯 New best F1 score: {best_f1:.4f}')

    print(f'\n🏆 Training completed! Best validation accuracy: {best_val_acc:.2f}%, Best F1: {best_f1:.4f}')

    return model, domain_classifier, action_classifier, train_history


# ==================== 主函数 ====================
def main():
    # 设置随机种子
    set_seed()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 数据路径
    source_data_path = 'csi-processed-data22'
    target_data_path = 'processed_csi_data22'

    # 创建数据集
    print("\n" + "=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)

    source_dataset = CSIDataset(source_data_path, domain_id=0, max_samples_per_action=150, balance_classes=True)
    target_dataset = CSIDataset(target_data_path, domain_id=1, max_samples_per_action=150, balance_classes=True)

    # 检查数据集是否为空
    if len(source_dataset) == 0 and len(target_dataset) == 0:
        print("No data loaded! Please check your data paths.")
        return

    # 获取动作类别数
    all_labels = source_dataset.labels + target_dataset.labels if len(source_dataset) > 0 and len(
        target_dataset) > 0 else source_dataset.labels if len(source_dataset) > 0 else target_dataset.labels
    num_actions = len(set(all_labels))

    # ==================== 对源域数据进行质量筛选 ====================
    if len(source_dataset) > 0:
        print("\n" + "=" * 60)
        print("SUPER FILTERING SOURCE DOMAIN DATA")
        print("=" * 60)

        # 使用超级增强的筛选函数
        source_dataset = super_filter_source_data(
            source_dataset,
            device,
            confidence_threshold=0.7,  # 使用更高的阈值
            num_actions=num_actions,
            batch_size=32,
            num_epochs=25  # 更多训练轮次
        )

        if len(source_dataset.data) == 0:
            print("⚠️ Warning: All source domain data was filtered out! Adjusting threshold...")
            # 如果所有数据都被过滤掉了，重新加载并使用更低的阈值
            source_dataset = CSIDataset(source_data_path, domain_id=0, max_samples_per_action=150,
                                        balance_classes=True)
            source_dataset = super_filter_source_data(
                source_dataset,
                device,
                confidence_threshold=0.5,  # 使用更低的阈值
                num_actions=num_actions,
                batch_size=32,
                num_epochs=20
            )

    # 计算数据统计
    real_data_stats = source_dataset.compute_statistics() if len(source_dataset) > 0 else None

    # 合并数据集
    combined_data = []
    combined_labels = []
    combined_domains = []

    if len(source_dataset) > 0:
        combined_data.extend(source_dataset.data)
        combined_labels.extend(source_dataset.labels)
        combined_domains.extend(source_dataset.domains)

    if len(target_dataset) > 0:
        combined_data.extend(target_dataset.data)
        combined_labels.extend(target_dataset.labels)
        combined_domains.extend(target_dataset.domains)

    if len(combined_data) == 0:
        print("No data loaded! Exiting...")
        return

    print(f"\nAfter filtering:")
    print(f"Total samples: {len(combined_data)}")
    print(f"Unique actions: {sorted(set(combined_labels))}")
    print(f"Unique domains: {sorted(set(combined_domains))}")

    # 计算类别权重
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(combined_labels),
        y=combined_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class weights: {class_weights}")

    # 创建数据加载器
    combined_dataset = list(zip(combined_data, combined_labels, combined_domains))

    # 确保有足够的数据进行分割
    if len(combined_dataset) < 10:
        print(f"Not enough data for training! Only {len(combined_dataset)} samples found.")
        return

    train_dataset, val_dataset = train_test_split(
        combined_dataset, test_size=0.25, random_state=42,
        stratify=combined_labels
    )

    def collate_fn(batch):
        max_len = min(max([item[0].shape[0] for item in batch]), 990)
        padded_csi = []
        actions = []
        domains = []

        for csi, action, domain in batch:
            if csi.shape[0] < max_len:
                padding = np.zeros((max_len - csi.shape[0], 27))
                csi = np.vstack([csi, padding])
            elif csi.shape[0] > max_len:
                csi = csi[:max_len]

            padded_csi.append(csi)
            actions.append(action)
            domains.append(domain)

        return {
            'csi': torch.stack([torch.tensor(csi, dtype=torch.float32) for csi in padded_csi]),
            'action': torch.tensor(actions, dtype=torch.long),
            'domain': torch.tensor(domains, dtype=torch.long)
        }

    # 使用适当的batch size
    batch_size = 16 if torch.cuda.is_available() else 8

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"\nDataset split:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # 创建超级增强模型
    print("\n" + "=" * 60)
    print("CREATING SUPER ENHANCED MODELS")
    print("=" * 60)

    # 模型参数
    patch_sizes = [20, 40, 80]
    d_model = 256
    num_domains = len(set(combined_domains))
    num_actions = len(set(combined_labels))

    print(f"Model Configuration:")
    print(f"  - Patch sizes: {patch_sizes}")
    print(f"  - Model dimension: {d_model}")
    print(f"  - Number of heads: 8")
    print(f"  - Number of layers: 4")
    print(f"  - Number of actions: {num_actions}")
    print(f"  - Number of domains: {num_domains}")

    # 初始化模型
    model = SuperEnhancedConditionalDiffusionCore(
        input_dim=27,
        d_model=d_model,
        nhead=8,
        num_layers=4,
        patch_sizes=patch_sizes,
        num_domains=num_domains,
        num_actions=num_actions,
        dropout=0.15,
        max_timesteps=1000
    ).to(device)

    domain_classifier = EnhancedDomainClassifier(
        input_dim=d_model,
        hidden_dims=[256, 128],
        num_domains=num_domains,
        use_grl=True,
        dropout=0.2
    ).to(device)

    action_classifier = EnhancedActionClassifier(
        d_model=d_model,
        hidden_dims=[512, 256, 128],
        num_actions=num_actions,
        dropout=0.2
    ).to(device)

    scheduler = AdvancedDiffusionScheduler(
        num_timesteps=100,
        schedule_type='cosine',
        s=0.008
    )

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

    # 开始训练
    print("\n" + "=" * 60)
    print("STARTING SUPER ENHANCED TRAINING WITH FILTERED DATA")
    print("=" * 60)

    model, domain_classifier, action_classifier, train_history = train_model_super_enhanced(
        model, domain_classifier, action_classifier,
        train_dataloader, val_dataloader, scheduler, device,
        class_weights=class_weights,
        num_epochs=1000,
        use_ema=True
    )

# 最终验证
    print("\n" + "=" * 60)
    print("FINAL VALIDATION")
    print("=" * 60)

    final_metrics = validate_model_detailed(
        model, domain_classifier, action_classifier,
        val_dataloader, scheduler, device
    )

    print(f"Final Validation Results:")
    print(f"  Overall Accuracy: {final_metrics['action_acc']:.2f}%")
    print(f"  Macro F1 Score: {final_metrics['macro_f1']:.4f}")
    print(f"  Per-class Recall:")
    for cls, recall in sorted(final_metrics['per_class_recall'].items()):
        print(f"    Action {cls}: {recall:.2f}%")

    # 绘制增强的混淆矩阵
    print("\n" + "=" * 60)
    print("GENERATING ENHANCED CONFUSION MATRIX")
    print("=" * 60)

    model.eval()
    action_classifier.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            csi_data = batch['csi'].to(device)
            action_labels = batch['action'].to(device)
            domain_labels = batch['domain'].to(device)

            _, cls_token = model.encode(csi_data, domain_labels)
            action_pred = action_classifier(cls_token)
            _, predicted = action_pred.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(action_labels.cpu().numpy())

    # 绘制增强的混淆矩阵
    classes = [f'Action {i}' for i in range(num_actions)]
    plot_enhanced_confusion_matrix(all_labels, all_preds, classes, 'confusion_matrix_super_enhanced.png')

    # 保存模型
    checkpoint = {
        'model': model.state_dict(),
        'domain_classifier': domain_classifier.state_dict(),
        'action_classifier': action_classifier.state_dict(),
        'scheduler': scheduler,
        'train_history': train_history,
        'config': {
            'patch_sizes': patch_sizes,
            'd_model': d_model,
            'num_domains': num_domains,
            'num_actions': num_actions
        },
        'class_weights': class_weights.cpu().numpy(),
        'final_metrics': final_metrics,
        'real_data_stats': real_data_stats
    }

    save_path = 'super_enhanced_diffusion_model_final.pth'
    torch.save(checkpoint, save_path)
    print(f"✅ Models saved to {save_path}")

    # 生成增强数据（可选）
    generate_augmented = input("\nGenerate super high-quality augmented data? (y/n): ").lower() == 'y'

    if generate_augmented:
        print("\n" + "=" * 60)
        print("GENERATING SUPER HIGH-QUALITY AUGMENTED DATA WITH STRICT FILTERING")
        print("=" * 60)

        # 创建超级质量检查器
        quality_checker = SuperGeneratedDataQualityChecker(
            real_data_stats,
            strict_mode=True  # 使用严格模式
        ) if real_data_stats else None

        unique_actions = sorted(set(combined_labels))
        generated_save_path = 'super_generated_data_final'

        # 为每个动作生成数据
        from collections import Counter
        action_counts = Counter(combined_labels)
        median_count = sorted(action_counts.values())[len(action_counts) // 2]

        # 统计总体生成情况
        total_requested = 0
        total_generated = 0

        for action_type in unique_actions:
            # 每个动作固定生成750个样本
            num_to_generate = 750  # 修改这里：固定为750

            total_requested += num_to_generate
            print(f"\n🎯 Generating {num_to_generate} samples for Action {action_type}")

            generated = generate_super_high_quality_data(
                model, scheduler, device,
                num_samples=num_to_generate,  # 这里会使用750
                seq_len=990,
                action_type=action_type,
                domain_id=0,
                save_path=generated_save_path,
                quality_checker=quality_checker,
                use_ddim=True,
                guidance_scale=2.0,
                action_classifier=action_classifier,
                confidence_threshold=0.75,
                temperature=0.9
            )

            if generated is not None:
                total_generated += len(generated)

        # 打印总体生成统计
        print("\n" + "=" * 60)
        print("GENERATION SUMMARY")
        print("=" * 60)
        print(f"📊 Total requested samples: {total_requested}")
        print(f"📊 Total generated samples: {total_generated}")
        print(f"📊 Overall success rate: {100 * total_generated / max(total_requested, 1):.1f}%")

        if total_generated < total_requested * 0.6:
            print("\n⚠️ Generation success rate could be improved. Consider:")
            print("  1. Fine-tuning the model for more epochs")
            print("  2. Adjusting the guidance scale (current: 2.0)")
            print("  3. Modifying the temperature parameter")
            print("  4. Adjusting confidence thresholds")

    # 绘制增强的训练曲线
    print("\n" + "=" * 60)
    print("PLOTTING ENHANCED TRAINING CURVES")
    print("=" * 60)

    if len(train_history['total_loss']) > 0:
        plot_enhanced_training_curves(train_history, num_actions)

    print("\n🎉 Training completed successfully with super enhanced model!")
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"✅ Final Validation Accuracy: {final_metrics['action_acc']:.2f}%")
    print(f"✅ Final Macro F1 Score: {final_metrics['macro_f1']:.4f}")
    print(f"✅ Model saved to: {save_path}")

    # 提供改进建议
    if final_metrics['action_acc'] < 90:
        print("\n🔍 Suggestions for further improvement:")
        print("1. Collect more real samples, especially for low-performing classes")
        print("2. Experiment with different model architectures (more layers/heads)")
        print("3. Try different data augmentation strategies")
        print("4. Adjust the loss function weights")
        print("5. Use ensemble methods with multiple models")


if __name__ == "__main__":
    main()