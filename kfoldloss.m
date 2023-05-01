function [error] = kfoldloss(partitionedModel)
  error=kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
% error=100-acc;
end