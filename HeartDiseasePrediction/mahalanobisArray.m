%MA(i) is Mahalanobis distance between y and ith instance of X
function MA = mahalanobisArray(X,y)

M = zeros(1,length(X(:,1)));
d = length(X(:,1));
S = cov(X);

for i=1:d
    M(i) = (X(i,:)-y)*inv(S)*(X(i,:)-y)';
end

MA = M;

end