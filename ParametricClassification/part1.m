% HW1 Part1
function part1()
fileID = fopen('iris-dataset.txt');
C = textscan(fileID,'%f %f %f %f %s','Delimiter',',');
fclose(fileID);

%Xall is data matrix. Rows are instances, columns are features
Xall = zeros(150,4);
Xall(:,1) = C{1};
Xall(:,2) = C{2};
Xall(:,3) = C{3};
Xall(:,4) = C{4};

%Xtrain is data matrix for training. Rows are instances, columns are features
Xtrain = zeros(90,4);
for i=1:30
    Xtrain(i,:) = Xall(i,:);
    Xtrain(i+30,:) = Xall(i+50,:);
    Xtrain(i+60,:) = Xall(i+100,:);
end

%Xtest is data matrix for testing. Rows are instances, columns are features
Xtest = zeros(60,4);
for i=1:20
    Xtest(i,:) = Xall(i+30,:);
    Xtest(i+20,:) = Xall(i+80,:);
    Xtest(i+40,:) = Xall(i+130,:);
end

%Rall is output matrix
Rall = zeros(150,1);
for i=1:150
    if strcmp(C{5}{i},'Iris-setosa')
        Rall(i,1) = 1;
    end
    if strcmp(C{5}{i},'Iris-versicolor')
        Rall(i,1) = 2;
    end
    if strcmp(C{5}{i},'Iris-virginica')
        Rall(i,1) = 3;
    end
end

%Rtrain is output matrix for training data
Rtrain = zeros(90,1);
for i=1:30
    Rtrain(i) = Rall(i);
    Rtrain(i+30) = Rall(i+50);
    Rtrain(i+60) = Rall(i+100);
end

%Rtest is output matrix for test data
Rtest = zeros(60,1);
for i=1:20
    Rtest(i) = Rall(i+30);
    Rtest(i+20) = Rall(i+80);
    Rtest(i+40) = Rall(i+130);
end

%Class Priors
N = length(Rtrain);
j = 0;
k = 0;
l = 0;
for i=1:N
    if Rtrain(i) == 1
        j = j + 1;
    elseif Rtrain(i) == 2
        k = k + 1;
    else
        l = l + 1;
    end
end
pC1 = j/N;
pC2 = k/N;
pC3 = l/N;

%Sample Means
X1 = [0 0 0 0];
X2 = [0 0 0 0];
X3 = [0 0 0 0];
for i=1:N
    if Rtrain(i) == 1
        X1 = X1 + Xtrain(i,:);
    elseif Rtrain(i) == 2
        X2 = X2 + Xtrain(i,:);
    else
        X3 = X3 + Xtrain(i,:);
    end
end
m1 = (X1/j)';
m2 = (X2/k)';
m3 = (X3/l)';

%Sample Variances
S1 = zeros(4,4);
S2 = zeros(4,4);
S3 = zeros(4,4);
for i=1:N
    if Rtrain(i) == 1
        S1 = S1 + ((Xtrain(i,:))'-m1)*((Xtrain(i,:))'-m1)';
    elseif Rtrain(i) == 2
        S2 = S2 + ((Xtrain(i,:))'-m2)*((Xtrain(i,:))'-m2)';
    else
        S3 = S3 + ((Xtrain(i,:))'-m3)*((Xtrain(i,:))'-m3)';
    end
end
S1 = S1 / j;
S2 = S2 / k;
S3 = S3 / l;

%Discriminant Function and Output
function y = output(x,s1,s2,s3)
    g1 = -0.5*log(det(s1)) - 0.5*(x-m1)'*inv(s1)*(x-m1) + log(pC1);
    g2 = -0.5*log(det(s2)) - 0.5*(x-m2)'*inv(s2)*(x-m2) + log(pC2);
    g3 = -0.5*log(det(s3)) - 0.5*(x-m3)'*inv(s3)*(x-m3) + log(pC3);
    if g1>g2
        if g1>g3
            y = 1;
        else
            y = 3;
        end
    else
        if g2>g3
            y = 2;
        else
            y = 3;
        end
    end
end

%CASE 1
%Test with training data
outputTrain1 = zeros(90,1);
for i=1:90
    outputTrain1(i) = output(Xtrain(i,:)',S1,S2,S3);
end
outputTrain1
%Test with test data
outputTest1 = zeros(60,1);
for i=1:60
    outputTest1(i) = output(Xtest(i,:)',S1,S2,S3);
end
outputTest1

%CASE 2
%Common Covariance Matrix
Scommon = pC1*S1 + pC2*S2 +pC3*S3;
%Test with training data
outputTrain2 = zeros(90,1);
for i=1:90
    outputTrain2(i) = output(Xtrain(i,:)',Scommon,Scommon,Scommon);
end
outputTrain2
%Test with test data
outputTest2 = zeros(60,1);
for i=1:60
    outputTest2(i) = output(Xtest(i,:)',Scommon,Scommon,Scommon);
end
outputTest2

%CASE 3
%Common Covariance Matrix with 0 off-diagonals
for i=1:4
    for j=1:4
        if i ~= j
            Scommon(i,j) = 0;
        end
    end
end
%Test with training data
outputTrain3 = zeros(90,1);
for i=1:90
    outputTrain3(i) = output(Xtrain(i,:)',Scommon,Scommon,Scommon);
end
outputTrain3
%Test with test data
outputTest3 = zeros(60,1);
for i=1:60
    outputTest3(i) = output(Xtest(i,:)',Scommon,Scommon,Scommon);
end
outputTest3

%CASE 4
%Common Covariance Matrix with 0 off-diagonals and equal variances
tot = sum(Scommon(:));
for i=1:4
    Scommon(i,i) = tot/4;
end
%Test with training data
outputTrain4 = zeros(90,1);
for i=1:90
    outputTrain4(i) = output(Xtrain(i,:)',Scommon,Scommon,Scommon);
end
outputTrain4
%Test with test data
outputTest4 = zeros(60,1);
for i=1:60
    outputTest4(i) = output(Xtest(i,:)',Scommon,Scommon,Scommon);
end
outputTest4
end