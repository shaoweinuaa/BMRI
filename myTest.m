clear all
clc
close all


load simulation.mat
rho1=0.1;
rho2=0.1;
rho3=100;
sampleNumber=100;
trainData=[r1Train;r2Train];
testData=[r1Test;r2Test]

trainLabel=[ones(size(r1Train,1),1);-ones(size(r1Train,1),1)];
testLabel=trainLabel;

% % % % label=trainLabel;

maxIteration=10;
meanFeature=mean(trainData,1);
stdFeature=std(trainData,1);
trainData=(trainData-repmat(meanFeature,size(trainData,1),1))./repmat(stdFeature,size(trainData,1),1);
testData=(testData-repmat(meanFeature,size(testData,1),1))./repmat(stdFeature,size(testData,1),1);

batchSize=sampleNumber*0.1;
labelIndex=[];

for i=1:19
  if i==1
     [result{i},queryIndex,predict_b,unLabelIndexRandom]=randomLearning(labelIndex,trainData,trainLabel,batchSize);     
     accuracyAL(i)=result{i}.accuracy;
  else
     [result{i},queryIndex,predict_b,unLabelIndexSS]=ActiveLearning(labelIndex,trainData,trainLabel,batchSize,maxIteration,rho1,rho2,rho3);     
     accuracyAL(i)=result{i}.accuracy; 
  end
     labelIndex=[labelIndex,queryIndex];
    
end


result=[];
labelInd=[];
queryInde=[];
labelIndex=[];
for i=1:19
     [result{i},queryIndex,predict_b,unLabelIndexRandom]=randomLearning(labelIndex,trainData,trainLabel,batchSize);     
     accuracy_random(i)=result{i}.accuracy; 
  
     labelIndex=[labelIndex,queryIndex];
    
end
accuracyAL
accuracy_random

