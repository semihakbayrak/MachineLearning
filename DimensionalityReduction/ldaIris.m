% HW2 ldaIris
function ldaIris()
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
N = length(Xtrain(:,1));
X_1 = [0 0 0 0];
X_2 = [0 0 0 0];
X_3 = [0 0 0 0];
for i=1:N
    if Rtrain(i) == 1
        X_1 = X_1 + Xtrain(i,:);
    elseif Rtrain(i) == 2
        X_2 = X_2 + Xtrain(i,:);
    else
        X_3 = X_3 + Xtrain(i,:);
    end
end
mX1 = (X_1/j)';
mX2 = (X_2/k)';
mX3 = (X_3/l)';

%Scatter Matrices
Sw1 = zeros(4,4);
Sw2 = zeros(4,4);
Sw3 = zeros(4,4);
for i=1:N
    if Rtrain(i) == 1
        Sw1 = Sw1 + ((Xtrain(i,:))'-mX1)*((Xtrain(i,:))'-mX1)';
    elseif Rtrain(i) == 2
        Sw2 = Sw2 + ((Xtrain(i,:))'-mX2)*((Xtrain(i,:))'-mX2)';
    else
        Sw3 = Sw3 + ((Xtrain(i,:))'-mX3)*((Xtrain(i,:))'-mX3)';
    end
end
Sw = Sw1 + Sw2 + Sw3;
mall = (mX1 + mX2 + mX3)/3;
Sb = j*(mX1-mall)*(mX1-mall)' + k*(mX2-mall)*(mX2-mall)' + l*(mX3-mall)*(mX3-mall)';
STotal = inv(Sw)*Sb;

%Finding largest eigenvalues and corresponding eigenvectors
[EigVec,EigVal] = eig(STotal);
max1Val = EigVal(1,1);
max1Index = 1;
for i=2:4
    if EigVal(i,i) > max1Val
        max1Val = EigVal(i,i);
        max1Index = i;
    end
end
EigVal(max1Index,:) = [];
EigVal(:,max1Index) = [];
max2Val = EigVal(1,1);
max2Index = 1;
for i=1:3
    if EigVal(i,i) >= max2Val
        max2Val = EigVal(i,i);
        if i>=max1Index
            max2Index = i+1;
        else
            max2Index = i;
        end
    end
end
max1Vec = EigVec(:,max1Index);
max2Vec = EigVec(:,max2Index);

%Finding new coordinate values
ztrain = zeros(N,2);
for i=1:N
    ztrain(i,1) = max1Vec' * (Xtrain(i,:))';
    ztrain(i,2) = max2Vec' * (Xtrain(i,:))';
end

%Sample Means in new domain
X1 = [0 0];
X2 = [0 0];
X3 = [0 0];
for i=1:N
    if Rtrain(i) == 1
        X1 = X1 + ztrain(i,:);
    elseif Rtrain(i) == 2
        X2 = X2 + ztrain(i,:);
    else
        X3 = X3 + ztrain(i,:);
    end
end
m1 = (X1/j)';
m2 = (X2/k)';
m3 = (X3/l)';

%Sample Variances in new domain
S1 = zeros(2,2);
S2 = zeros(2,2);
S3 = zeros(2,2);
for i=1:N
    if Rtrain(i) == 1
        S1 = S1 + ((ztrain(i,:))'-m1)*((ztrain(i,:))'-m1)';
    elseif Rtrain(i) == 2
        S2 = S2 + ((ztrain(i,:))'-m2)*((ztrain(i,:))'-m2)';
    else
        S3 = S3 + ((ztrain(i,:))'-m3)*((ztrain(i,:))'-m3)';
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

%Finding new coordinate values for test data 
ztest = zeros(60,2);
for i=1:60
    ztest(i,1) = max1Vec' * (Xtest(i,:))';
    ztest(i,2) = max2Vec' * (Xtest(i,:))';
end

%CASE 1
%Test with training data
outputTrain1 = zeros(90,1);
for i=1:90
    outputTrain1(i) = output(ztrain(i,:)',S1,S2,S3);
end
outputTrain1
%Test with test data
outputTest1 = zeros(60,1);
for i=1:60
    outputTest1(i) = output(ztest(i,:)',S1,S2,S3);
end
outputTest1

%plotting with training data
figure
for i=1:30
    plot(ztrain(i,1),ztrain(i,2),'g+');
    hold on
end
for i=31:60
    plot(ztrain(i,1),ztrain(i,2),'ko');
    hold on
end
for i=61:90
    plot(ztrain(i,1),ztrain(i,2),'r*');
    hold on
end
%Plotting decision boundries
axisforz1 = -3:0.1:3;
axisforz2 = -3:0.1:3;
[ax1,ax2] = meshgrid(axisforz1,axisforz2);
F1 = mvnpdf([ax1(:) ax2(:)],m1',S1);
F1 = reshape(F1,length(axisforz1),length(axisforz2));
F2 = mvnpdf([ax1(:) ax2(:)],m2',S2);
F2 = reshape(F2,length(axisforz1),length(axisforz2));
F3 = mvnpdf([ax1(:) ax2(:)],m3',S3);
F3 = reshape(F3,length(axisforz1),length(axisforz2));
contour(axisforz1,axisforz2,F1);
hold on
contour(axisforz1,axisforz2,F2);
hold on
contour(axisforz1,axisforz2,F3);
hold on
bound1z1 = zeros(1,61);
bound1z2 = zeros(1,61);
bound1extraz1 = zeros(1,61);
bound1extraz2 = zeros(1,61);
bound2z1 = zeros(1,61);
bound2z2 = zeros(1,61);
bound2extraz1 = zeros(1,61);
bound2extraz2 = zeros(1,61);

for a=61:-1:1
    for b=1:61
        if posterior((a-31)/10,(b-31)/10,1,1,0)>posterior((a-31)/10,(b-31)/10,2,1,0)
            bound1z1(62-a) = (a-31)/10;
            bound1z2(62-a) = (b-31)/10;
            break
        end
    end
    for b=61:-1:1
        if posterior((a-31)/10,(b-31)/10,1,1,0)>posterior((a-31)/10,(b-31)/10,2,1,0)
            bound1extraz1(62-a) = (a-31)/10;
            bound1extraz2(62-a) = (b-31)/10;
            break
        end
    end
end
for a=61:-1:1
    for b=1:61
        if posterior((a-31)/10,(b-31)/10,2,1,0)>posterior((a-31)/10,(b-31)/10,3,1,0)
            bound2z1(62-a) = (a-31)/10;
            bound2z2(62-a) = (b-31)/10;
            break
        end
    end
    for b=61:-1:1
        if posterior((a-31)/10,(b-31)/10,2,1,0)>posterior((a-31)/10,(b-31)/10,3,1,0)
            bound2extraz1(62-a) = (a-31)/10;
            bound2extraz2(62-a) = (b-31)/10;
            break
        end
    end
end
bound2z1(bound2z1 == 0) = NaN;
bound2z2(bound2z2 == 0) = NaN;
bound2extraz1(bound2extraz1 == 0) = NaN;
bound2extraz2(bound2extraz2 == 0) = NaN;
plot(bound1z1,bound1z2,'black');
hold on
plot(bound1extraz1,bound1extraz2,'black');
hold on
plot(bound2z1,bound2z2,'red');
hold on
plot(bound2extraz1,bound2extraz2,'red');
xlabel('z1'); ylabel('z2');
title('Decision boundries for case 1')
%posterior function
function p = posterior(x1,x2,class,casenum,s)
    if casenum == 1
        mp1 = mvnpdf([x1 x2], m1', S1);
        mp2 = mvnpdf([x1 x2], m2', S2);
        mp3 = mvnpdf([x1 x2], m3', S3);
    else
        mp1 = mvnpdf([x1 x2], m1', s);
        mp2 = mvnpdf([x1 x2], m2', s);
        mp3 = mvnpdf([x1 x2], m3', s);
    end
    if class == 1
        p = mp1*pC1/(mp1*pC1+mp2*pC2+mp3*pC3);
    elseif class == 2
        p = mp2*pC2/(mp1*pC1+mp2*pC2+mp3*pC3);
    else
        p = mp3*pC3/(mp1*pC1+mp2*pC2+mp3*pC3);
    end
end

%CASE 2
%Common Covariance Matrix
Scommon = pC1*S1 + pC2*S2 +pC3*S3;
%Test with training data
outputTrain2 = zeros(90,1);
for i=1:90
    outputTrain2(i) = output(ztrain(i,:)',Scommon,Scommon,Scommon);
end
outputTrain2
%Test with test data
outputTest2 = zeros(60,1);
for i=1:60
    outputTest2(i) = output(ztest(i,:)',Scommon,Scommon,Scommon);
end
outputTest2

%plotting with training data
figure
for i=1:30
    plot(ztrain(i,1),ztrain(i,2),'g+');
    hold on
end
for i=31:60
    plot(ztrain(i,1),ztrain(i,2),'ko');
    hold on
end
for i=61:90
    plot(ztrain(i,1),ztrain(i,2),'r*');
    hold on
end
%Plotting decision boundries
axisforz1 = -3:0.1:3;
axisforz2 = -3:0.1:3;
[ax1,ax2] = meshgrid(axisforz1,axisforz2);
F1 = mvnpdf([ax1(:) ax2(:)],m1',Scommon);
F1 = reshape(F1,length(axisforz1),length(axisforz2));
F2 = mvnpdf([ax1(:) ax2(:)],m2',Scommon);
F2 = reshape(F2,length(axisforz1),length(axisforz2));
F3 = mvnpdf([ax1(:) ax2(:)],m3',Scommon);
F3 = reshape(F3,length(axisforz1),length(axisforz2));
contour(axisforz1,axisforz2,F1);
hold on
contour(axisforz1,axisforz2,F2);
hold on
contour(axisforz1,axisforz2,F3);
hold on
bound1z1 = zeros(1,61);
bound1z2 = zeros(1,61);
bound1extraz1 = zeros(1,61);
bound1extraz2 = zeros(1,61);
bound2z1 = zeros(1,61);
bound2z2 = zeros(1,61);
bound2extraz1 = zeros(1,61);
bound2extraz2 = zeros(1,61);

for a=61:-1:1
    for b=1:61
        if posterior((a-31)/10,(b-31)/10,1,2,Scommon)>posterior((a-31)/10,(b-31)/10,2,2,Scommon)
            bound1z1(62-a) = (a-31)/10;
            bound1z2(62-a) = (b-31)/10;
            break
        end
    end
    for b=61:-1:1
        if posterior((a-31)/10,(b-31)/10,1,2,Scommon)>posterior((a-31)/10,(b-31)/10,2,2,Scommon)
            bound1extraz1(62-a) = (a-31)/10;
            bound1extraz2(62-a) = (b-31)/10;
            break
        end
    end
end
for a=61:-1:1
    for b=1:61
        if posterior((a-31)/10,(b-31)/10,2,2,Scommon)>posterior((a-31)/10,(b-31)/10,3,2,Scommon)
            bound2z1(62-a) = (a-31)/10;
            bound2z2(62-a) = (b-31)/10;
            break
        end
    end
    for b=61:-1:1
        if posterior((a-31)/10,(b-31)/10,2,2,Scommon)>posterior((a-31)/10,(b-31)/10,3,2,Scommon)
            bound2extraz1(62-a) = (a-31)/10;
            bound2extraz2(62-a) = (b-31)/10;
            break
        end
    end
end
bound2z1(bound2z1 == 0) = NaN;
bound2z2(bound2z2 == 0) = NaN;
bound2extraz1(bound2extraz1 == 0) = NaN;
bound2extraz2(bound2extraz2 == 0) = NaN;
bound1z1(bound1z1 == 0) = NaN;
bound1z2(bound1z2 == 0) = NaN;
bound1extraz1(bound1extraz1 == 0) = NaN;
bound1extraz2(bound1extraz2 == 0) = NaN;
plot(bound1z1,bound1z2,'black');
hold on
plot(bound1extraz1,bound1extraz2,'black');
hold on
plot(bound2z1,bound2z2,'red');
hold on
plot(bound2extraz1,bound2extraz2,'red');
xlabel('z1'); ylabel('z2');
title('Decision boundries for case 2')

%CASE 3
%Common Covariance Matrix with 0 off-diagonals
for i=1:2
    for j=1:2
        if i ~= j
            Scommon(i,j) = 0;
        end
    end
end
%Test with training data
outputTrain3 = zeros(90,1);
for i=1:90
    outputTrain3(i) = output(ztrain(i,:)',Scommon,Scommon,Scommon);
end
outputTrain3
%Test with test data
outputTest3 = zeros(60,1);
for i=1:60
    outputTest3(i) = output(ztest(i,:)',Scommon,Scommon,Scommon);
end
outputTest3

%plotting with training data
figure
for i=1:30
    plot(ztrain(i,1),ztrain(i,2),'g+');
    hold on
end
for i=31:60
    plot(ztrain(i,1),ztrain(i,2),'ko');
    hold on
end
for i=61:90
    plot(ztrain(i,1),ztrain(i,2),'r*');
    hold on
end
%Plotting decision boundries
axisforz1 = -3:0.1:3;
axisforz2 = -3:0.1:3;
[ax1,ax2] = meshgrid(axisforz1,axisforz2);
F1 = mvnpdf([ax1(:) ax2(:)],m1',Scommon);
F1 = reshape(F1,length(axisforz1),length(axisforz2));
F2 = mvnpdf([ax1(:) ax2(:)],m2',Scommon);
F2 = reshape(F2,length(axisforz1),length(axisforz2));
F3 = mvnpdf([ax1(:) ax2(:)],m3',Scommon);
F3 = reshape(F3,length(axisforz1),length(axisforz2));
contour(axisforz1,axisforz2,F1);
hold on
contour(axisforz1,axisforz2,F2);
hold on
contour(axisforz1,axisforz2,F3);
hold on
bound1z1 = zeros(1,61);
bound1z2 = zeros(1,61);
bound1extraz1 = zeros(1,61);
bound1extraz2 = zeros(1,61);
bound2z1 = zeros(1,61);
bound2z2 = zeros(1,61);
bound2extraz1 = zeros(1,61);
bound2extraz2 = zeros(1,61);

for a=61:-1:1
    for b=1:61
        if posterior((a-31)/10,(b-31)/10,1,3,Scommon)>posterior((a-31)/10,(b-31)/10,2,3,Scommon)
            bound1z1(62-a) = (a-31)/10;
            bound1z2(62-a) = (b-31)/10;
            break
        end
    end
    for b=61:-1:1
        if posterior((a-31)/10,(b-31)/10,1,3,Scommon)>posterior((a-31)/10,(b-31)/10,2,3,Scommon)
            bound1extraz1(62-a) = (a-31)/10;
            bound1extraz2(62-a) = (b-31)/10;
            break
        end
    end
end
for a=61:-1:1
    for b=1:61
        if posterior((a-31)/10,(b-31)/10,2,3,Scommon)>posterior((a-31)/10,(b-31)/10,3,3,Scommon)
            bound2z1(62-a) = (a-31)/10;
            bound2z2(62-a) = (b-31)/10;
            break
        end
    end
    for b=61:-1:1
        if posterior((a-31)/10,(b-31)/10,2,3,Scommon)>posterior((a-31)/10,(b-31)/10,3,3,Scommon)
            bound2extraz1(62-a) = (a-31)/10;
            bound2extraz2(62-a) = (b-31)/10;
            break
        end
    end
end
bound2z1(bound2z1 == 0) = NaN;
bound2z2(bound2z2 == 0) = NaN;
bound2extraz1(bound2extraz1 == 0) = NaN;
bound2extraz2(bound2extraz2 == 0) = NaN;
bound1z1(bound1z1 == 0) = NaN;
bound1z2(bound1z2 == 0) = NaN;
bound1extraz1(bound1extraz1 == 0) = NaN;
bound1extraz2(bound1extraz2 == 0) = NaN;
plot(bound1z1,bound1z2,'black');
hold on
plot(bound1extraz1,bound1extraz2,'black');
hold on
plot(bound2z1,bound2z2,'red');
hold on
plot(bound2extraz1,bound2extraz2,'red');
xlabel('z1'); ylabel('z2');
title('Decision boundries for case 3')

%CASE 4
%Common Covariance Matrix with 0 off-diagonals and equal variances
tot = sum(Scommon(:));
for i=1:2
    Scommon(i,i) = tot/2;
end
%Test with training data
outputTrain4 = zeros(90,1);
for i=1:90
    outputTrain4(i) = output(ztrain(i,:)',Scommon,Scommon,Scommon);
end
outputTrain4
%Test with test data
outputTest4 = zeros(60,1);
for i=1:60
    outputTest4(i) = output(ztest(i,:)',Scommon,Scommon,Scommon);
end
outputTest4

%plotting with training data
figure
for i=1:30
    plot(ztrain(i,1),ztrain(i,2),'g+');
    hold on
end
for i=31:60
    plot(ztrain(i,1),ztrain(i,2),'ko');
    hold on
end
for i=61:90
    plot(ztrain(i,1),ztrain(i,2),'r*');
    hold on
end
%Plotting decision boundries
axisforz1 = -3:0.1:3;
axisforz2 = -3:0.1:3;
[ax1,ax2] = meshgrid(axisforz1,axisforz2);
F1 = mvnpdf([ax1(:) ax2(:)],m1',Scommon);
F1 = reshape(F1,length(axisforz1),length(axisforz2));
F2 = mvnpdf([ax1(:) ax2(:)],m2',Scommon);
F2 = reshape(F2,length(axisforz1),length(axisforz2));
F3 = mvnpdf([ax1(:) ax2(:)],m3',Scommon);
F3 = reshape(F3,length(axisforz1),length(axisforz2));
contour(axisforz1,axisforz2,F1);
hold on
contour(axisforz1,axisforz2,F2);
hold on
contour(axisforz1,axisforz2,F3);
hold on
bound1z1 = zeros(1,61);
bound1z2 = zeros(1,61);
bound1extraz1 = zeros(1,61);
bound1extraz2 = zeros(1,61);
bound2z1 = zeros(1,61);
bound2z2 = zeros(1,61);
bound2extraz1 = zeros(1,61);
bound2extraz2 = zeros(1,61);

for a=61:-1:1
    for b=1:61
        if posterior((a-31)/10,(b-31)/10,1,4,Scommon)>posterior((a-31)/10,(b-31)/10,2,4,Scommon)
            bound1z1(62-a) = (a-31)/10;
            bound1z2(62-a) = (b-31)/10;
            break
        end
    end
    for b=61:-1:1
        if posterior((a-31)/10,(b-31)/10,1,4,Scommon)>posterior((a-31)/10,(b-31)/10,2,4,Scommon)
            bound1extraz1(62-a) = (a-31)/10;
            bound1extraz2(62-a) = (b-31)/10;
            break
        end
    end
end
for a=61:-1:1
    for b=1:61
        if posterior((a-31)/10,(b-31)/10,2,4,Scommon)>posterior((a-31)/10,(b-31)/10,3,4,Scommon)
            bound2z1(62-a) = (a-31)/10;
            bound2z2(62-a) = (b-31)/10;
            break
        end
    end
    for b=61:-1:1
        if posterior((a-31)/10,(b-31)/10,2,4,Scommon)>posterior((a-31)/10,(b-31)/10,3,4,Scommon)
            bound2extraz1(62-a) = (a-31)/10;
            bound2extraz2(62-a) = (b-31)/10;
            break
        end
    end
end
bound2z1(bound2z1 == 0) = NaN;
bound2z2(bound2z2 == 0) = NaN;
bound2extraz1(bound2extraz1 == 0) = NaN;
bound2extraz2(bound2extraz2 == 0) = NaN;
bound1z1(bound1z1 == 0) = NaN;
bound1z2(bound1z2 == 0) = NaN;
bound1extraz1(bound1extraz1 == 0) = NaN;
bound1extraz2(bound1extraz2 == 0) = NaN;
plot(bound1z1,bound1z2,'black');
hold on
plot(bound1extraz1,bound1extraz2,'black');
hold on
plot(bound2z1,bound2z2,'red');
hold on
plot(bound2extraz1,bound2extraz2,'red');
xlabel('z1'); ylabel('z2');
title('Decision boundries for case 4')

end