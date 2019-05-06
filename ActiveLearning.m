function [result,queryIndex,predict_b,unLabelIndexSS]=ActiveLearning(labelIndexSS,data,label,batchSize,maxIteration,rho1,rho2,rho3,testData,testLabel)
  currentFolder = pwd;
  addpath(genpath(currentFolder));
  L=getLaplacianMatrix(data,1);
  unLabelIndexSS=setdiff(1:size(data,1),labelIndexSS);
  XL=data(labelIndexSS,:);
  XU=data(unLabelIndexSS,:);
  YL=label(labelIndexSS);
  YU=label(unLabelIndexSS);
  alpha=zeros(1,length(unLabelIndexSS));
  m1=eye(length(unLabelIndexSS))*(1/(batchSize+length(labelIndexSS)));
  Z=mean(data)-ones(1,length(labelIndexSS))./((batchSize+length(labelIndexSS)))*XL;
  for i=1:maxIteration
    if i==1
       W(:,i)=(XL'*XL+data'*L*data+0.01*eye(size(data,2)))\XL'*YL;
    else
       W(:,i)=(XL'*XL+rho1*XU'*diag(alpha')*XU+rho2*data'*L*data+rho3*(Z-alpha'*m1*XU)'*(Z-alpha'*m1*XU)+0.01*eye(size(data,2)))\XL'*YL;    
    end
    k1=m1*XU*W(:,i)*W(:,i)'*XU'*m1';
    k=-2*rho2*m1*XU*W(:,i)*W(:,i)'*Z'+rho1*diag(XU*W(:,i)*W(:,i)'*XU');
    alpha=quadprog(k1,k,[],[],ones(1,length(unLabelIndexSS)),batchSize,zeros(length(unLabelIndexSS),1)); 
    if(i>1&&norm(W(:,i)-W(:,i-1),2)<1e-9)
      break;
    end    
  end
  [sA,index] = sort(alpha,'descend');
  queryIndex=unLabelIndexSS(index(1:batchSize));
  
  labelIndexSS=[labelIndexSS,queryIndex];
  unLabelIndexSS=setdiff(1:size(data,1),labelIndexSS);
  
  TrainSample=data(labelIndexSS',:);
  TrainLabel=label(labelIndexSS');
  model=svmtrain(double(TrainLabel),double(TrainSample));
  [predict_b, accuracy, dec] = svmpredict(double(testLabel), double(testData), model);
  result=getClassificationResult(predict_b,testLabel);
end