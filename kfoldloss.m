function [error] = kfoldloss(acc)
  %error=kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
   error=100-acc;
end