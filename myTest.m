clear all
clc
close all

load simulation.mat

for k=1:20
rho1=100;
rho2=0.01;
rho3=1;
sampleNumber=100;
trainLabel=[ones(100,1);-ones(100,1)];
testLabel=trainLabel;
maxIteration=10;
meanFeature=mean(trainData,1);
stdFeature=std(trainData,1);
trainData=(trainData-repmat(meanFeature,size(trainData,1),1))./repmat(stdFeature,size(trainData,1),1);
testData=(testData-repmat(meanFeature,size(testData,1),1))./repmat(stdFeature,size(testData,1),1);
batchSize=10;
labelIndex=[];

for i=1:20
  if i==1
     [result{i},queryIndex,predict_b,unLabelIndexRandom]=randomLearning(labelIndex,trainData,trainLabel,batchSize,testData,testLabel);     
     accuracyAL(i)=result{i}.accuracy;

  else
    [result{i},queryIndex,predict_b,unLabelIndexSS]=ActiveLearning(labelIndex,trainData,trainLabel,batchSize,maxIteration,rho1,rho2,rho3,testData,testLabel);     
     accuracyAL(i)=result{i}.accuracy; 
  end
     labelIndex=[labelIndex,queryIndex];
    
end

result=[];
labelInd=[];
queryInde=[];
labelIndex=[];
for i=1:20
       [result{i},queryIndex,predict_b,unLabelIndexRandom]=randomLearning(labelIndex,trainData,trainLabel,batchSize,testData,testLabel);     
       accuracy_random(i)=result{i}.accuracy; 
       labelIndex=[labelIndex,queryIndex];
    
end
AL(k,:)=accuracyAL;
end
mean(AL)