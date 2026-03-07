clc
clear all;

% 创建输出目录结构
output_dir = 'csi-processed-data22';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 主数据目录
main_data_dir = '20181130_user15_16_17';
if ~exist(main_data_dir, 'dir')
    error('找不到20181130_user15_16_17文件夹，请确保该文件夹存在');
end

% 用户子文件夹
user_folders = {'user15', 'user16', 'user17'};

fprintf('开始处理CSI数据文件...\n');

% 遍历每个用户文件夹
for user_idx = 1:length(user_folders)
    user_folder = user_folders{user_idx};
    user_dir = fullfile(main_data_dir, user_folder);
    
    if ~exist(user_dir, 'dir')
        fprintf('警告: 找不到用户文件夹 %s，跳过\n', user_folder);
        continue;
    end
    
    % 创建用户输出目录
    user_output_dir = fullfile(output_dir, user_folder);
    if ~exist(user_output_dir, 'dir')
        mkdir(user_output_dir);
    end
    
    % 获取用户文件夹中的所有.dat文件
    dat_files = dir(fullfile(user_dir, '*.dat'));
    if isempty(dat_files)
        fprintf('警告: 在%s文件夹中未找到.dat文件\n', user_folder);
        continue;
    end
    
    fprintf('在%s文件夹中找到 %d 个.dat文件\n', user_folder, length(dat_files));
    
    % 处理每个.dat文件
    for file_idx = 1:length(dat_files)
        filename = dat_files(file_idx).name;
        processFile(filename, user_output_dir, user_dir);
    end
end

% 文件处理函数
function processFile(filename, target_dir, data_dir)
    filepath = fullfile(data_dir, filename);
    [~, name, ~] = fileparts(filename);
    
    fprintf('正在处理文件: %s\n', filename);
    
    try
        % 读取CSI数据文件
        csi_trace = read_bf_file(filepath);
        
        % 获取样本数量
        num_samples = min(length(csi_trace), 990);  % 限制最大样本数
        
        % 初始化存储结构：3个天线 × 30个子载波 × 样本点 (保存复数形式)
        antenna_complex_data = zeros(3, 30, num_samples);
        antenna_amplitude_data = zeros(3, 30, num_samples);
        
        % 处理所有样本点，保存复数CSI数据
        for i = 1:num_samples
            csi_entry = csi_trace{i};
            csi = get_scaled_csi(csi_entry);
            csi_matrix = squeeze(csi(1,:,:)).'; % 30×3 complex
            
            % 保存原始复数形式
            for k = 1:3
                antenna_complex_data(k, :, i) = csi_matrix(:, k);
            end
            
            % 转换为dB并应用限幅处理（用于特征选择）
            csi_abs = db(abs(csi_matrix));
            
            % 限幅处理
            csi_abs(csi_abs >= 35) = 35;
            csi_abs(csi_abs <= 1) = 1;
            
            % 存储幅度数据（仅用于特征选择）
            for k = 1:3
                antenna_amplitude_data(k, :, i) = csi_abs(:, k);
            end
        end
        
        % 使用幅度数据进行特征选择，但保留复数数据进行滤波
        all_amplitude_subcarriers = zeros(90, num_samples); % 90个子载波 (3天线 × 30子载波)
        all_complex_subcarriers = zeros(90, num_samples); % 保存复数数据
        subcarrier_count = 0;
        
        for k = 1:3
            % 初始化滤波后的结果
            filtered_amplitude_data = zeros(30, num_samples);
            filtered_complex_data = zeros(30, num_samples);
            
            % 对每个子载波应用hampel滤波、巴特沃斯低通滤波和移动平均滤波
            for sub = 1:30
                % 提取当前子载波的幅度数据（用于滤波参数确定）
                amplitude_data = squeeze(antenna_amplitude_data(k, sub, :));
                
                % 提取当前子载波的复数数据
                complex_data = squeeze(antenna_complex_data(k, sub, :));
                
                % 应用hampel滤波去除异常值（基于幅度数据确定异常值位置）
                [~, outlier_idx, ~, ~] = hampel(amplitude_data, 10, 4);
                
                % 对复数数据应用同样的异常值处理
                filtered_complex = complex_data;
                if ~isempty(outlier_idx)
                    % 用邻近值替换异常值
                    for out_i = 1:length(outlier_idx)
                        idx = outlier_idx(out_i);
                        if idx > 1 && idx < length(complex_data)
                            filtered_complex(idx) = (complex_data(idx-1) + complex_data(idx+1)) / 2;
                        elseif idx == 1 && length(complex_data) > 1
                            filtered_complex(idx) = complex_data(2);
                        elseif idx == length(complex_data) && length(complex_data) > 1
                            filtered_complex(idx) = complex_data(end-1);
                        end
                    end
                end
                
                % 应用巴特沃斯低通滤波器到复数数据
                fs = 1; % 采样频率
                cutoff = 0.1; % 截止频率
                order = 4; % 滤波器阶数
                
                % 设计巴特沃斯低通滤波器
                [b, a] = butter(order, cutoff/(fs/2), 'low');
                
                % 分别对实部和虚部应用滤波器
                real_filtered = filtfilt(b, a, real(filtered_complex));
                imag_filtered = filtfilt(b, a, imag(filtered_complex));
                butterworth_complex_filtered = complex(real_filtered, imag_filtered);
                
                % 应用移动平均滤波器到复数数据
                window_size = 5; % 移动平均窗口大小
                if length(butterworth_complex_filtered) >= window_size
                    real_movmean = movmean(real(butterworth_complex_filtered), window_size);
                    imag_movmean = movmean(imag(butterworth_complex_filtered), window_size);
                    final_complex_filtered = complex(real_movmean, imag_movmean);
                else
                    final_complex_filtered = butterworth_complex_filtered;
                end
                
                % 同时处理幅度数据用于特征选择
                [hampel_amplitude_filtered, ~, ~, ~] = hampel(amplitude_data, 10, 4);
                butterworth_amplitude_filtered = filtfilt(b, a, hampel_amplitude_filtered);
                if length(butterworth_amplitude_filtered) >= window_size
                    movmean_amplitude_filtered = movmean(butterworth_amplitude_filtered, window_size);
                else
                    movmean_amplitude_filtered = butterworth_amplitude_filtered;
                end
                
                filtered_amplitude_data(sub, :) = movmean_amplitude_filtered;
                filtered_complex_data(sub, :) = final_complex_filtered;
            end
            
            % 合并所有天线的子载波数据
            subcarrier_indices = (subcarrier_count+1):(subcarrier_count+30);
            all_amplitude_subcarriers(subcarrier_indices, :) = filtered_amplitude_data;
            all_complex_subcarriers(subcarrier_indices, :) = filtered_complex_data;
            subcarrier_count = subcarrier_count + 30;
        end
        
        % PCA特征选择 - 选择前5个主成分（基于幅度数据）
        data_for_pca = all_amplitude_subcarriers';
        [coeff, score, latent] = pca(data_for_pca);
        
        % 选择前4个主成分
        num_components = min(4, size(coeff, 2));
        top_components = coeff(:, 1:num_components);
        
        % 计算子载波与前4个主成分的相似度（使用互信息）
        sensitivity_matrix = zeros(size(all_amplitude_subcarriers, 1), 1);
        
        for i = 1:size(all_amplitude_subcarriers, 1)
            subcarrier_signal = all_amplitude_subcarriers(i, :);
            total_mi = 0;
            
            % 计算与前4个主成分的互信息之和
            for j = 1:num_components
                component_signal = score(:, j)';
                
                % 计算互信息
                mi_value = mutual_information(subcarrier_signal, component_signal);
                total_mi = total_mi + mi_value;
            end
            
            sensitivity_matrix(i) = total_mi;
        end
        
        % 选择灵敏度矩阵的前27个子载波
        num_selected = min(27, length(sensitivity_matrix));
        [~, idx] = sort(sensitivity_matrix, 'descend');
        selected_subcarriers = idx(1:num_selected);
        
        % 提取选中子载波的复数数据（这是最终要保存的数据）
        selected_complex_data = all_complex_subcarriers(selected_subcarriers, :);
        
        % 保存为.mat文件（只保存处理后的复数CSI数据）
        mat_name = [name, '.mat'];
        mat_path = fullfile(target_dir, mat_name);
        
        % 创建最终的CSI复数数据变量
        csi_data = selected_complex_data;
        
        % 只保存处理后的CSI复数数据
        save(mat_path, 'csi_data', '-v7.3');
        
        fprintf('成功处理文件: %s -> %s (保存了%d个选中子载波的复数CSI数据，数据维度: %dx%d)\n', ...
            filename, mat_name, num_selected, size(csi_data, 1), size(csi_data, 2));
        
    catch ME
        fprintf('处理文件 %s 时出错: %s\n', filename, ME.message);
    end
end

% 互信息计算函数
function mi = mutual_information(x, y)
    % 简化的互信息计算
    % 将连续信号离散化为直方图
    
    % 数据归一化
    x = (x - min(x)) / (max(x) - min(x) + eps);
    y = (y - min(y)) / (max(y) - min(y) + eps);
    
    % 设置直方图bins数量
    num_bins = min(50, floor(length(x)/10));
    
    % 计算联合直方图
    [joint_hist, ~] = histcounts2(x, y, num_bins);
    joint_hist = joint_hist / sum(joint_hist(:)); % 归一化为概率
    
    % 计算边际概率
    px = sum(joint_hist, 2);
    py = sum(joint_hist, 1);
    
    % 计算互信息
    mi = 0;
    for i = 1:size(joint_hist, 1)
        for j = 1:size(joint_hist, 2)
            if joint_hist(i,j) > 0 && px(i) > 0 && py(j) > 0
                mi = mi + joint_hist(i,j) * log2(joint_hist(i,j) / (px(i) * py(j)));
            end
        end
    end
    
    % 处理可能的负值或NaN
    if isnan(mi) || mi < 0
        mi = 0;
    end
end

fprintf('\n批量处理完成！\n');
fprintf('输出目录结构:\n');
fprintf('csi-processed-data/\n');
fprintf('├── user15/          (.mat文件)\n');
fprintf('├── user16/          (.mat文件)\n');
fprintf('└── user17/          (.mat文件)\n');

% 统计处理结果
fprintf('\n处理统计:\n');
for i = 1:length(user_folders)
    user_folder = user_folders{i};
    user_output_dir = fullfile(output_dir, user_folder);
    if exist(user_output_dir, 'dir')
        mat_files = dir(fullfile(user_output_dir, '*.mat'));
        fprintf('%s: 处理了 %d 个文件\n', user_folder, length(mat_files));
    end
end

fprintf('\n每个.mat文件包含:\n');
fprintf('- csi_data: 处理后的CSI复数数据 (选中子载波 × 时间点)\n');
fprintf('  - 数据类型: 复数矩阵\n');
fprintf('  - 维度: 27个选中子载波 × 时间采样点\n');
fprintf('  - 处理方法: hampel滤波 + 巴特沃斯低通滤波 + 移动平均滤波\n');
fprintf('  - 子载波选择: 基于PCA和互信息的特征选择\n');