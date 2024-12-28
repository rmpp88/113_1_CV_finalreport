close all;clear all; clc;
%%
% 讀取數據和標記圖像 %
% 設置路徑
imageDir = '.\images\train\image1';  % 輸入圖像資料夾
labelDir = '.\images\train\Label3ceilmean';  % 標記圖像資料夾

% 定義類別和標籤 (數值應該與標記圖像中的像素值對應)
classes = ["background", "pleura", "Reverberation", "Muscle"];
labelIDs = [0, 1, 2, 3];

% 創建圖像資料存儲
imds = imageDatastore(imageDir);

% 創建 pixelLabelDatastore (標記影像)
pxds = pixelLabelDatastore(labelDir, classes, labelIDs);

% 檢查影像與標籤數量是否一致
if numel(imds.Files) ~= numel(pxds.Files)
    error('原始影像數量與標記影像數量不一致，請檢查資料。');
end

% 檢查配對數據
preview(pxds);

%%
% % 1. 分割數據
% [imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');
% [pxdsTrain, pxdsVal] = splitEachLabel(pxds, 0.8, 'randomized');
% % 2. 創建訓練和驗證資料集
% trainingData = combine(imdsTrain, pxdsTrain);
% validationData = combine(imdsVal, pxdsVal);

%%
% 定義摺數
k = 5;
% 創建交叉驗證分區
cv = cvpartition(numel(imds.Files), 'KFold', k);
% 紀錄每個摺的訓練和驗證結果
allMetrics = cell(k, 1);
% 紀錄每個摺的模型和結果
foldResults = struct('Model', [], 'Metrics', [], 'Fold', []);  % 用於存儲每個摺的結果

bestDice = -inf;  % 用於追蹤最佳 Dice 分數
bestModel = [];   % 用於存儲最佳模型

% 迴圈執行每個摺的交叉驗證
for fold = 1:k
    % 分割訓練和驗證資料
    trainIdx = training(cv, fold);  % 訓練索引
    testIdx = test(cv, fold);      % 驗證索引
    
    % 建立訓練和驗證的 ImageDatastore
    imdsTrain = subset(imds, find(trainIdx));
    imdsVal = subset(imds, find(testIdx));
    
    % 創建 Pixel Label Datastore
    pxdsTrain = subset(pxds, find(trainIdx));
    pxdsVal = subset(pxds, find(testIdx));
    
    % 合併訓練和驗證資料
    trainingData = combine(imdsTrain, pxdsTrain);
    % 定義增強函數
    augmentedData = transform(trainingData, @(data) augmentData(data));
    % 验证数据不做增強
    validationData = combine(imdsVal, pxdsVal);
    
    % 設定 U-Net 架構和訓練選項
    lgraph = unetLayers([256 256 1], numel(classes));  % 假設單通道影像
    % lgraph = attentionUnet([256 256 1], numel(classes));


    % % Define the attention layer
    % attentionSize=[256 256 1];
    % attentionLayer = attentionLayer('AttentionSize', attentionSize);
    % % Create the rest of your deep learning model
    % layers = [
    %     imageInputLayer([inputImageSize])
    %     convolution2dLayer(3, 64, 'Padding', 'same')
    %     reluLayer
    %     attentionLayer
    %     fullyConnectedLayer(numClasses)
    %     softmaxLayer
    %     classificationLayer
    % ];
    % % Create the deep learning network
    % net = layerGraph(layers);
    % % Visualize the network
    % plot(net);


    % 使用自訂的 U-Net + 注意力機制
    % lgraph = unetWithAttention([256 256 1], numel(classes));
    % % 繪製網路結構以檢查
    % analyzeNetwork(lgraph);


    % options = trainingOptions('adam', ...
    %     'InitialLearnRate', 1e-4, ...
    %     'MaxEpochs', 50, ...
    %     'MiniBatchSize', 16, ...
    %     'Shuffle', 'every-epoch', ...
    %     'ValidationData', validationData, ...
    %     'ValidationFrequency', 50, ...
    %     'Plots', 'training-progress', ...
    %     'Verbose', false);

    options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...                       % 初始學習率
    'LearnRateSchedule', 'piecewise', ...               % 學習率調整方式
    'LearnRateDropFactor', 0.1, ...                     % 學習率下降因子
    'LearnRateDropPeriod', 10, ...                      % 學習率下降週期
    'MaxEpochs', 50, ...                                % 訓練最大迭代次數
    'MiniBatchSize', 16, ...                            % mini-batch 大小
    'ExecutionEnvironment', 'gpu', ...                  % 使用GPU加速 QQ
    'Shuffle', 'every-epoch', ...                       % 每個epoch隨機打亂資料
    'GradientThreshold', 1.0, ...                       % 梯度裁剪
    'Plots', 'training-progress', ...                   % 顯示訓練過程
    'Verbose', true ,...                                 % 顯示詳盡訊息
    'ValidationData', validationData, ...               % 驗證資料
    'ValidationFrequency', 50, ...                      % 驗證頻率
    'ValidationPatience', 5  );                     % 提前停止的容忍次數
    
    % 訓練模型
    disp(['Training Fold ', num2str(fold)]);
    net = trainNetwork(augmentedData, lgraph, options);
    
    % 驗證模型 (可選)
    metrics = evaluateModel(net, validationData);  % 用戶需要定義 evaluateModel 函數

    disp(['Fold ', num2str(fold), ' Dice Score: ', num2str(metrics.Dice)]);
    
    % 保存結果
    foldResults(fold).Model = net;
    foldResults(fold).Metrics = metrics;
    foldResults(fold).Fold = fold;
    
    % 檢查是否是最佳結果
    if metrics.Dice > bestDice
        bestDice = metrics.Dice;
        bestModel = net;  % 保存最佳模型
    end
end

% 顯示最佳結果
disp(['Best Fold Dice Score: ', num2str(bestDice)]);

% 保存最佳模型
modelPath = '.\images\模型架構\1226_bestModel.mat';  % 儲存路徑
save(modelPath, 'bestModel');

% 保存所有摺的結果
resultsPath = '.\images\模型架構\1226_crossValidationResults.mat';  % 儲存路徑
save(resultsPath, 'foldResults');

% 將結果轉為表格格式
foldTable = struct2table(foldResults);

% 保存為 CSV 檔案
resultsCsvPath = '.\images\模型架構\1226_crossValidationResults.csv';
writetable(foldTable, resultsCsvPath);

% % 設置 U-Net 網路架構 %
% % 定義輸入圖像大小和類別數量
% imageSize = [256 256 1];  % 假設單通道的 256x256 圖像
% numClasses = numel(classes);  % 背景、肋膜線、多重反射線
% 
% % 建立 U-Net 網路  
% lgraph = unetLayers(imageSize, numClasses);
% % disp(lgraph);
% % plot(lgraph);
%%
%  設置訓練選項 %
% options = trainingOptions('adam', ...
%     'InitialLearnRate', 1e-4, ...                       % 初始學習率
%     'LearnRateSchedule', 'piecewise', ...               % 學習率調整方式
%     'LearnRateDropFactor', 0.1, ...                     % 學習率下降因子
%     'LearnRateDropPeriod', 10, ...                      % 學習率下降週期
%     'MaxEpochs', 50, ...                                % 訓練最大迭代次數
%     'MiniBatchSize', 16, ...                            % mini-batch 大小
%     'ExecutionEnvironment', 'gpu', ...                  % 使用GPU加速 QQ
%     'Shuffle', 'every-epoch', ...                       % 每個epoch隨機打亂資料
%     'GradientThreshold', 1.0, ...                       % 梯度裁剪
%     'Plots', 'training-progress', ...                   % 顯示訓練過程
%     'Verbose', true); %,...                                 % 顯示詳盡訊息
    % 'ValidationData', validationData, ...               % 驗證資料
    % 'ValidationFrequency', 50, ...                      % 驗證頻率
    % 'ValidationPatience', 5                       % 提前停止的容忍次數

% 進行 U-Net 訓練 %
% 將圖像和標記資料存儲組合成 training data
% trainingData = combine(imds, pxds);
% % tds = transform(trainingData, @(data) preprocessTrainingData(data, imageSize));
% % augtds = transform(tds, @(data) augmentData(data));  % 加入增強
% data = readall(trainingData); % 針對 datastore 提取所有數據
% for i = 1:size(data, 1)
%     img = data{i, 1}; % 圖像
%     lbl = data{i, 2}; % 標籤
%     if ~isequal(size(img, 1:2), size(lbl, 1:2))
%         error('圖像和標籤的大小不一致！檢查數據集。');
%     end
% end
% 訓練 U-Net
% net = trainNetwork(trainingData, lgraph, options);

%%
% 保存模型和附加資訊
% save('C:\Users\226\OneDrive - 國立陽明交通大學\桌面\深度學習標記\模型架構\trainedUNetWithInfo_1129.mat', 'net', 'classes', 'labelIDs');
% 載入儲存的模型
% loadedInfo = load('C:\Users\226\OneDrive - 國立陽明交通大學\桌面\深度學習標記\模型架構\trainedUNetWithInfo_1128.mat');
% net = loadedInfo.net;  % 提取模型
% classes = loadedInfo.classes;
% labelIDs = loadedInfo.labelIDs;
%%
modelPath = '.\images\模型架構\1216_bestModel.mat';  % 儲存路徑
loadedInfo = load(modelPath);
net = loadedInfo.bestModel;  % 提取模型


%%
% % 定義類別與顏色
% classColors = [
%     0, 0, 0;        % 背景 (黑色)
%     255, 0, 0;      % 胸膜 (紅色)
%     0, 255, 0;      % 反射 (綠色)
%     0, 0, 255;      % 肌肉 (藍色)
% ] ./ 255;  % 歸一化到 [0, 1] 範圍
% % 創建顏色映射
% cmap = colorcube(numel(classes));
% cmap(1:size(classColors, 1), :) = classColors; % 將自定義顏色插入映射
% 
% % 測試和可視化分割結果 %
% % 測試分割
% testImage = imread('.\images\test\image\image50.png');
% segmentedImage = semanticseg(testImage, net,Classes=classes);
% 
% 
% % 視覺化
% figure;
% 
% % 疊加分割結果
% subplot(1, 2, 1);
% imshow(labeloverlay(testImage, segmentedImage, 'Colormap', cmap, 'Transparency', 0.5));
% title('分割結果與原始圖像疊加');
% 
% % 單獨顯示分割結果
% subplot(1, 2, 2);
% segImageColored = label2rgb(uint8(segmentedImage), cmap, 'k', 'noshuffle');
% imshow(segImageColored);
% title('彩色分割結果');
% 
% % 顯示圖例
% legendEntries = classes; % 類別名稱
% legendColors = classColors; % 類別顏色
% 
% for i = 1:numel(classes)
%     hold on;
%     scatter(NaN, NaN, 100, legendColors(i, :), 'filled', 'DisplayName', legendEntries(i));
% end
% legend('show', 'Location', 'southoutside', 'Orientation', 'horizontal');
%%
%%
testDir = '.\images\test\image';  % 輸入圖像資料夾
testdata = imageDatastore(testDir);

% 定義類別和標籤 (數值應該與標記圖像中的像素值對應)
classes = ["background", "pleura"];%, "Reverberation", "Muscle"
labelIDs = [0, 1];%, 3, 4

testlabelDir = '.\images\test\LabelPleura1216_median';  % 標記圖像資料夾
testlabel = pixelLabelDatastore(testlabelDir, classes, labelIDs);

testingData = combine(testdata, testlabel);

% 定義類別與顏色
classColors = [
    0, 0, 0;        % 背景 (黑色)
    255, 0, 0;      % 胸膜 (紅色)
    0, 255, 0;      % 反射 (綠色)
    0, 0, 255;      % 肌肉 (藍色)
] ./ 255;  % 歸一化到 [0, 1] 範圍
% 創建顏色映射
cmap = colorcube(numel(classes));
cmap(1:size(classColors, 1), :) = classColors; % 將自定義顏色插入映射

% 假設 testingData 是測試數據
diceScores = [];
% 用於存儲每筆數據的所有類別 Dice 系數
allDiceScores = [];  


% 測試數據總數
numTestingFiles = numel(testdata.Files);
% 存儲每筆數據的混淆矩陣
confusionMatrices = cell(1, numTestingFiles);  

% 遍歷測試數據集的每一筆
for i = 1:numTestingFiles
    % 使用 subset 提取第 i 筆數據
    subsetData = subset(testingData, i);
    data = read(subsetData);  % 返回值可能是 struct
    
    % 分離圖像和真實標籤
    image = data{1};       % 圖像
    trueLabel = data{2};   % 真實標籤
    
    % 預測標籤
    predictedLabel = semanticseg(image, net, 'ExecutionEnvironment', 'gpu');
    
    % figure
    % imshow(labeloverlay(image, predictedLabel, 'Colormap', cmap, 'Transparency', 0.5));
    % title('分割結果與原始圖像疊加');
    % % 保存標記圖像
    % filename_save=['.\images\分割測試結果\Label1pleura\image',num2str(i),'_label.png'];
    % imwrite(labeloverlay(image, predictedLabel, 'Colormap', cmap, 'Transparency', 0.5), filename_save);
    % close

    % 計算 Dice 系數
    diceScore = dice(predictedLabel, trueLabel);
    allDiceScores = [allDiceScores; diceScore(:)'];  % 每筆數據一行
    % 過濾掉 NaN 值並存儲有效的 Dice 值
    validDiceScores = diceScore(~isnan(diceScore));
    if ~isempty(validDiceScores)
        diceScores = [diceScores; mean(validDiceScores)];
    end
    % predictedLabelIdx = uint8(predictedLabel);  % 轉換為數字標籤
    % trueLabelIdx = uint8(trueLabel);            % 轉換為數字標籤
    % % 獲取當前數據中的唯一標籤值
    % uniqueLabels = union(unique(trueLabelIdx), unique(predictedLabelIdx));
    % cm = confusionmat(trueLabelIdx(:), predictedLabelIdx(:), ...
    %     'Order', labelIDs);  % 使用預定義的類別順序
    % confusionMatrices{i} = cm;  % 保存混淆矩陣
end
disp([mean(allDiceScores)])
% 計算測試的平均 Dice 分數
if ~isempty(diceScores)
    meanDice = mean(diceScores);
    disp(['Mean Dice Score for Testing: ', num2str(meanDice)]);
else
    disp('所有的 Dice Score 都是 NaN，無法計算平均值。');
end

% 將結果轉為表格格式（可選）
% diceTable = array2table(allDiceScores, ...
%     'VariableNames', ["Background", "Pleura", "Reverberation", "Muscle"]);
% 
% % 保存混淆矩陣和 Dice 結果（可選）
% save('DiceScores.mat', 'diceTable', 'confusionMatrices');