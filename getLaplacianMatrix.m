function L=getLaplacianMatrix(data,sigma)
S=calckernel('rbf',sigma,data);
D=diag(S*ones(size(data,1),1));
L=D-S;
end