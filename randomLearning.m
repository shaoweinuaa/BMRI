function [result,queryIndex,predict_b,unLabelIndexRandom]=randomLearning(labelIndexRandom,data,label,batchSize,testData,testLabel)
  currentFolder = pwd;
  addpath(genpath(currentFolder));
  num=1:size(data,1);
  unLabelIndexRandom=setdiff(num,labelIndexRandom);
  A=randperm(length(unLabelIndexRandom));
  queryIndex=unLabelIndexRandom(A(1:batchSize));
  labelIndexRandom=[labelIndexRandom,queryIndex];
  unLabelIndexRandom=setdiff(num,labelIndexRandom);
  
  randomTrainSample=data(labelIndexRandom',:);
  randomTrainLabel=label(labelIndexRandom');
  modelRandom=svmtrain(double(randomTrainLabel),double(randomTrainSample));
  [predict_b, accuracy_b, dec_values_b] = svmpredict(double(testLabel), double(testData), modelRandom);
  result=getClassificationResult(predict_b,testLabel);
end