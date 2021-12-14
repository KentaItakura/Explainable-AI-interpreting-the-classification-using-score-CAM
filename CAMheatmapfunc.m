function HeatMap = CAMheatmap(myNet,image,scores)

imageActivations = activations(myNet,image,...
    'inception_5b-output','OutputAs','channels');%ACTIVATIONデータを計算
% 対象クラスのWeightをWeightベクターに該当するLayerより取得
if scores(1) > scores(2)
    classIndex      = 1;
else
    classIndex      = 2;
end
weightVector    = myNet.Layers(142).Weights(classIndex,:);

% ACTIVATIONヒートマップ計算
weightVectorSize = size(weightVector);
weightVector = reshape(weightVector,[1 weightVectorSize]);
dotProduct = bsxfun(@times,imageActivations,weightVector);
classActivationMap = sum(dotProduct,3);
originalSize = size(image);
% ACTIVATIONヒートマップを画像の元サイズに復元
classActivationMap = imresize(classActivationMap,originalSize(1:2));

HeatMap = map2jpg(classActivationMap, [], 'jet');
HeatMap = uint8((im2double(image)*0.3+HeatMap*0.5)*255);

    function [img] = map2jpg(imgmap, range, colorMap)
        imgmap = double(imgmap);
        if(~exist('range', 'var') || isempty(range)),
            range = [min(imgmap(:)) max(imgmap(:))];
        end
        
        heatmap_gray = mat2gray(imgmap, range);
        heatmap_x = gray2ind(heatmap_gray, 256);
        heatmap_x(isnan(imgmap)) = 0;
        
        if(~exist('colorMap', 'var'))
            img = ind2rgb(heatmap_x, jet(256));
        else
            img = ind2rgb(heatmap_x, eval([colorMap '(256)']));
        end
    end
end

%% Copyright 2019 The MathWorks, Inc.