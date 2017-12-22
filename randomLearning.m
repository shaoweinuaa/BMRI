function [result,queryIndex,predict_b,unLabelIndexRandom]=randomLearning(labelIndexRandom,data,label,batchSize)
% % %   currentFolder = pwd;
% % %   addpath(genpath(currentFolder));
  num=1:size(data,1);
  unLabelIndexRandom=setdiff(num,labelIndexRandom);
  A=randperm(length(unLabelIndexRandom));
  queryIndex=unLabelIndexRandom(A(1:batchSize));
  labelIndexRandom=[labelIndexRandom,queryIndex];
  unLabelIndexRandom=setdiff(A,labelIndexRandom);
  
  randomTrainSample=data(labelIndexRandom',:);
  randomTrainLabel=label(labelIndexRandom');
  randomTestSample=data(unLabelIndexRandom,:);
  randomTestLabel=label(unLabelIndexRandom);
  options = ['-t 2 -c ' num2str(1) ' -g ' num2str(1)];
% % % %   options=[];
  modelRandom=svmtrain(double(randomTrainLabel),double(randomTrainSample),options);
  [predict_b, accuracy_b, dec_values_b] = svmpredict(double(randomTestLabel), double(randomTestSample), modelRandom);
  result=getClassificationResult(predict_b,randomTestLabel);
end