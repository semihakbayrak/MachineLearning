% HW1 Part2
function part2()
fileID = fopen('iris-dataset.txt');
C = textscan(fileID,'%f %f %f %f %s','Delimiter',',');
fclose(fileID);

%Xall is data matrix. Rows are instances, columns are features
Xall = zeros(150,2);
Xall(:,1) = C{3};
Xall(:,2) = C{4};

%Xtrain is data matrix for training. Rows are instances, columns are features
Xtrain = zeros(90,2);
for i=1:30
    Xtrain(i,:) = Xall(i,:);
    Xtrain(i+30,:) = Xall(i+50,:);
    Xtrain(i+60,:) = Xall(i+100,:);
end
%Xtest is data matrix for testing. Rows are instances, columns are features
Xtest = zeros(60,2);
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
X1 = [0 0];
X2 = [0 0];
X3 = [0 0];
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
S1 = zeros(2,2);
S2 = zeros(2,2);
S3 = zeros(2,2);
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
%plotting likelihoods
figure
axisforx3 = 0:0.1:8;
axisforx4 = 0:0.1:8;
[ax3,ax4] = meshgrid(axisforx3,axisforx4);
F1 = mvnpdf([ax3(:) ax4(:)],m1',S1);
F1 = reshape(F1,length(axisforx3),length(axisforx4));
surf(axisforx3,axisforx4,F1);
xlabel('x3'); ylabel('x4'); zlabel('Likelihood density p(x|C1)');
title('Case 1 p(x|C1)')
figure
F2 = mvnpdf([ax3(:) ax4(:)],m2',S2);
F2 = reshape(F2,length(axisforx3),length(axisforx4));
surf(axisforx3,axisforx4,F2);
xlabel('x3'); ylabel('x4'); zlabel('Likelihood density p(x|C2)');
title('Case 1 p(x|C2)')
figure
F3 = mvnpdf([ax3(:) ax4(:)],m3',S3);
F3 = reshape(F3,length(axisforx3),length(axisforx4));
surf(axisforx3,axisforx4,F3);
xlabel('x3'); ylabel('x4'); zlabel('Likelihood density p(x|C3)');
title('Case 1 p(x|C3)')
%plotting posteriors
Post1 = F1*pC1./(F1*pC1+F2*pC2+F3*pC3);
Post2 = F2*pC2./(F1*pC1+F2*pC2+F3*pC3);
Post3 = F3*pC3./(F1*pC1+F2*pC2+F3*pC3);
figure
surf(axisforx3,axisforx4,Post1);
xlabel('x3'); ylabel('x4'); zlabel('Posterior density p(C1|x)');
title('Case 1 p(C1|x)')
figure
surf(axisforx3,axisforx4,Post2);
xlabel('x3'); ylabel('x4'); zlabel('Posterior density p(C2|x)');
title('Case 1 p(C2|x)')
figure
surf(axisforx3,axisforx4,Post3);
xlabel('x3'); ylabel('x4'); zlabel('Posterior density p(C3|x)');
title('Case 1 p(C3|x)')
%Plotting decision boundries
figure
contour(axisforx3,axisforx4,F1);
hold on
contour(axisforx3,axisforx4,F2);
hold on
contour(axisforx3,axisforx4,F3);
hold on
bound1x3 = zeros(1,81);
bound1x4 = zeros(1,81);
bound2x3 = zeros(1,81);
bound2x4 = zeros(1,81);
bound2extrax3 = zeros(1,81);
bound2extrax4 = zeros(1,81);
for a=1:81
    for b=1:81
        if posterior((a-1)/10,(b-1)/10,1,1,0)<posterior((a-1)/10,(b-1)/10,2,1,0)
            bound1x3(a) = (a-1)/10;
            bound1x4(a) = (b-1)/10;
            break
        end
    end
end
for a=81:-1:1
    for b=1:81
        if posterior((a-1)/10,(b-1)/10,2,1,0)>posterior((a-1)/10,(b-1)/10,3,1,0)
            bound2x3(82-a) = (a-1)/10;
            bound2x4(82-a) = (b-1)/10;
            break
        end
    end
    for b=81:-1:1
        if posterior((a-1)/10,(b-1)/10,2,1,0)>posterior((a-1)/10,(b-1)/10,3,1,0)
            bound2extrax3(82-a) = (a-1)/10;
            bound2extrax4(82-a) = (b-1)/10;
            break
        end
    end
end
bound2x3(bound2x3 == 0) = NaN;
bound2x4(bound2x4 == 0) = NaN;
bound2extrax3(bound2extrax3 == 0) = NaN;
bound2extrax4(bound2extrax4 == 0) = NaN;
plot(bound1x3,bound1x4,'black');
hold on
plot(bound2x3,bound2x4,'red');
hold on
plot(bound2extrax3,bound2extrax4,'red');
xlabel('x3'); ylabel('x4');
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
    outputTrain2(i) = output(Xtrain(i,:)',Scommon,Scommon,Scommon);
end
outputTrain2
%Test with test data
outputTest2 = zeros(60,1);
for i=1:60
    outputTest2(i) = output(Xtest(i,:)',Scommon,Scommon,Scommon);
end
outputTest2
%plotting likelihoods
figure
axisforx3 = 0:0.1:8;
axisforx4 = 0:0.1:8;
[ax3,ax4] = meshgrid(axisforx3,axisforx4);
F1 = mvnpdf([ax3(:) ax4(:)],m1',Scommon);
F1 = reshape(F1,length(axisforx3),length(axisforx4));
surf(axisforx3,axisforx4,F1);
xlabel('x3'); ylabel('x4'); zlabel('Likelihood density p(x|C1)');
title('Case 2 p(x|C1)')
figure
F2 = mvnpdf([ax3(:) ax4(:)],m2',Scommon);
F2 = reshape(F2,length(axisforx3),length(axisforx4));
surf(axisforx3,axisforx4,F2);
xlabel('x3'); ylabel('x4'); zlabel('Likelihood density p(x|C2)');
title('Case 2 p(x|C2)')
figure
F3 = mvnpdf([ax3(:) ax4(:)],m3',Scommon);
F3 = reshape(F3,length(axisforx3),length(axisforx4));
surf(axisforx3,axisforx4,F3);
xlabel('x3'); ylabel('x4'); zlabel('Likelihood density p(x|C3)');
title('Case 2 p(x|C3)')
%plotting posteriors
Post1 = F1*pC1./(F1*pC1+F2*pC2+F3*pC3);
Post2 = F2*pC2./(F1*pC1+F2*pC2+F3*pC3);
Post3 = F3*pC3./(F1*pC1+F2*pC2+F3*pC3);
figure
surf(axisforx3,axisforx4,Post1);
xlabel('x3'); ylabel('x4'); zlabel('Posterior density p(C1|x)');
title('Case 2 p(C1|x)')
figure
surf(axisforx3,axisforx4,Post2);
xlabel('x3'); ylabel('x4'); zlabel('Posterior density p(C2|x)');
title('Case 2 p(C2|x)')
figure
surf(axisforx3,axisforx4,Post3);
xlabel('x3'); ylabel('x4'); zlabel('Posterior density p(C3|x)');
title('Case 2 p(C2|x)')
%Plotting decision boundries
figure
contour(axisforx3,axisforx4,F1);
hold on
contour(axisforx3,axisforx4,F2);
hold on
contour(axisforx3,axisforx4,F3);
hold on
bound1x3 = zeros(1,81);
bound1x4 = zeros(1,81);
bound2x3 = zeros(1,81);
bound2x4 = zeros(1,81);
bound2extrax3 = zeros(1,81);
bound2extrax4 = zeros(1,81);
for a=1:81
    for b=1:81
        if posterior((a-1)/10,(b-1)/10,1,2,Scommon)<posterior((a-1)/10,(b-1)/10,2,2,Scommon)
            bound1x3(a) = (a-1)/10;
            bound1x4(a) = (b-1)/10;
            break
        end
    end
end
for a=81:-1:1
    for b=1:81
        if posterior((a-1)/10,(b-1)/10,2,2,Scommon)>posterior((a-1)/10,(b-1)/10,3,2,Scommon)
            bound2x3(82-a) = (a-1)/10;
            bound2x4(82-a) = (b-1)/10;
            break
        end
    end
    for b=81:-1:1
        if posterior((a-1)/10,(b-1)/10,2,2,Scommon)>posterior((a-1)/10,(b-1)/10,3,2,Scommon)
            bound2extrax3(82-a) = (a-1)/10;
            bound2extrax4(82-a) = (b-1)/10;
            break
        end
    end
end
bound1x3(bound1x3 == 0) = NaN;
bound1x4(bound1x4 == 0) = NaN;
bound2x3(bound2x3 == 0) = NaN;
bound2x4(bound2x4 == 0) = NaN;
bound2extrax3(bound2extrax3 == 0) = NaN;
bound2extrax4(bound2extrax4 == 0) = NaN;
plot(bound1x3,bound1x4,'black');
hold on
plot(bound2x3,bound2x4,'red');
hold on
plot(bound2extrax3,bound2extrax4,'red');
xlabel('x3'); ylabel('x4');
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
    outputTrain3(i) = output(Xtrain(i,:)',Scommon,Scommon,Scommon);
end
outputTrain3
%Test with test data
outputTest3 = zeros(60,1);
for i=1:60
    outputTest3(i) = output(Xtest(i,:)',Scommon,Scommon,Scommon);
end
outputTest3
%plotting likelihoods
figure
axisforx3 = 0:0.1:8;
axisforx4 = 0:0.1:8;
[ax3,ax4] = meshgrid(axisforx3,axisforx4);
F1 = mvnpdf([ax3(:) ax4(:)],m1',Scommon);
F1 = reshape(F1,length(axisforx3),length(axisforx4));
surf(axisforx3,axisforx4,F1);
xlabel('x3'); ylabel('x4'); zlabel('Likelihood density p(x|C1)');
title('Case 3 p(x|C1)')
figure
F2 = mvnpdf([ax3(:) ax4(:)],m2',Scommon);
F2 = reshape(F2,length(axisforx3),length(axisforx4));
surf(axisforx3,axisforx4,F2);
xlabel('x3'); ylabel('x4'); zlabel('Likelihood density p(x|C2)');
title('Case 3 p(x|C2)')
figure
F3 = mvnpdf([ax3(:) ax4(:)],m3',Scommon);
F3 = reshape(F3,length(axisforx3),length(axisforx4));
surf(axisforx3,axisforx4,F3);
xlabel('x3'); ylabel('x4'); zlabel('Likelihood density p(x|C3)');
title('Case 3 p(x|C3)')
%plotting posteriors
Post1 = F1*pC1./(F1*pC1+F2*pC2+F3*pC3);
Post2 = F2*pC2./(F1*pC1+F2*pC2+F3*pC3);
Post3 = F3*pC3./(F1*pC1+F2*pC2+F3*pC3);
figure
surf(axisforx3,axisforx4,Post1);
xlabel('x3'); ylabel('x4'); zlabel('Posterior density p(C1|x)');
title('Case 3 p(C1|x)')
figure
surf(axisforx3,axisforx4,Post2);
xlabel('x3'); ylabel('x4'); zlabel('Posterior density p(C2|x)');
title('Case 3 p(C2|x)')
figure
surf(axisforx3,axisforx4,Post3);
xlabel('x3'); ylabel('x4'); zlabel('Posterior density p(C3|x)');
title('Case 3 p(C3|x)')
%Plotting decision boundries
figure
contour(axisforx3,axisforx4,F1);
hold on
contour(axisforx3,axisforx4,F2);
hold on
contour(axisforx3,axisforx4,F3);
hold on
bound1x3 = zeros(1,81);
bound1x4 = zeros(1,81);
bound2x3 = zeros(1,81);
bound2x4 = zeros(1,81);
bound2extrax3 = zeros(1,81);
bound2extrax4 = zeros(1,81);
for a=1:81
    for b=1:81
        if posterior((a-1)/10,(b-1)/10,1,3,Scommon)<posterior((a-1)/10,(b-1)/10,2,3,Scommon)
            bound1x3(a) = (a-1)/10;
            bound1x4(a) = (b-1)/10;
            break
        end
    end
end
for a=81:-1:1
    for b=1:81
        if posterior((a-1)/10,(b-1)/10,2,3,Scommon)>posterior((a-1)/10,(b-1)/10,3,3,Scommon)
            bound2x3(82-a) = (a-1)/10;
            bound2x4(82-a) = (b-1)/10;
            break
        end
    end
    for b=81:-1:1
        if posterior((a-1)/10,(b-1)/10,2,3,Scommon)>posterior((a-1)/10,(b-1)/10,3,3,Scommon)
            bound2extrax3(82-a) = (a-1)/10;
            bound2extrax4(82-a) = (b-1)/10;
            break
        end
    end
end
bound1x3(bound1x3 == 0) = NaN;
bound1x4(bound1x4 == 0) = NaN;
bound2x3(bound2x3 == 0) = NaN;
bound2x4(bound2x4 == 0) = NaN;
bound2extrax3(bound2extrax3 == 0) = NaN;
bound2extrax4(bound2extrax4 == 0) = NaN;
plot(bound1x3,bound1x4,'black');
hold on
plot(bound2x3,bound2x4,'red');
hold on
plot(bound2extrax3,bound2extrax4,'red');
xlabel('x3'); ylabel('x4');
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
    outputTrain4(i) = output(Xtrain(i,:)',Scommon,Scommon,Scommon);
end
outputTrain4
%Test with test data
outputTest4 = zeros(60,1);
for i=1:60
    outputTest4(i) = output(Xtest(i,:)',Scommon,Scommon,Scommon);
end
outputTest4
%plotting likelihoods
figure
axisforx3 = 0:0.1:8;
axisforx4 = 0:0.1:8;
[ax3,ax4] = meshgrid(axisforx3,axisforx4);
F1 = mvnpdf([ax3(:) ax4(:)],m1',Scommon);
F1 = reshape(F1,length(axisforx3),length(axisforx4));
surf(axisforx3,axisforx4,F1);
xlabel('x3'); ylabel('x4'); zlabel('Likelihood density p(x|C1)');
title('Case 4 p(x|C1)')
figure
F2 = mvnpdf([ax3(:) ax4(:)],m2',Scommon);
F2 = reshape(F2,length(axisforx3),length(axisforx4));
surf(axisforx3,axisforx4,F2);
xlabel('x3'); ylabel('x4'); zlabel('Likelihood density p(x|C2)');
title('Case 4 p(x|C2)')
figure
F3 = mvnpdf([ax3(:) ax4(:)],m3',Scommon);
F3 = reshape(F3,length(axisforx3),length(axisforx4));
surf(axisforx3,axisforx4,F3);
xlabel('x3'); ylabel('x4'); zlabel('Likelihood density p(x|C3)');
title('Case 4 p(x|C3)')
%plotting posteriors
Post1 = F1*pC1./(F1*pC1+F2*pC2+F3*pC3);
Post2 = F2*pC2./(F1*pC1+F2*pC2+F3*pC3);
Post3 = F3*pC3./(F1*pC1+F2*pC2+F3*pC3);
figure
surf(axisforx3,axisforx4,Post1);
xlabel('x3'); ylabel('x4'); zlabel('Posterior density p(C1|x)');
title('Case 4 p(C1|x)')
figure
surf(axisforx3,axisforx4,Post2);
xlabel('x3'); ylabel('x4'); zlabel('Posterior density p(C2|x)');
title('Case 4 p(C2|x)')
figure
surf(axisforx3,axisforx4,Post3);
xlabel('x3'); ylabel('x4'); zlabel('Posterior density p(C3|x)');
title('Case 4 p(C3|x)')
%Plotting decision boundries
figure
contour(axisforx3,axisforx4,F1);
hold on
contour(axisforx3,axisforx4,F2);
hold on
contour(axisforx3,axisforx4,F3);
hold on
bound1x3 = zeros(1,81);
bound1x4 = zeros(1,81);
bound2x3 = zeros(1,81);
bound2x4 = zeros(1,81);
bound2extrax3 = zeros(1,81);
bound2extrax4 = zeros(1,81);
for a=1:81
    for b=1:81
        if posterior((a-1)/10,(b-1)/10,1,3,Scommon)<posterior((a-1)/10,(b-1)/10,2,3,Scommon)
            bound1x3(a) = (a-1)/10;
            bound1x4(a) = (b-1)/10;
            break
        end
    end
end
for a=81:-1:1
    for b=1:81
        if posterior((a-1)/10,(b-1)/10,2,3,Scommon)>posterior((a-1)/10,(b-1)/10,3,3,Scommon)
            bound2x3(82-a) = (a-1)/10;
            bound2x4(82-a) = (b-1)/10;
            break
        end
    end
    for b=81:-1:1
        if posterior((a-1)/10,(b-1)/10,2,3,Scommon)>posterior((a-1)/10,(b-1)/10,3,3,Scommon)
            bound2extrax3(82-a) = (a-1)/10;
            bound2extrax4(82-a) = (b-1)/10;
            break
        end
    end
end
bound1x3(bound1x3 == 0) = NaN;
bound1x4(bound1x4 == 0) = NaN;
bound2x3(bound2x3 == 0) = NaN;
bound2x4(bound2x4 == 0) = NaN;
bound2extrax3(bound2extrax3 == 0) = NaN;
bound2extrax4(bound2extrax4 == 0) = NaN;
plot(bound1x3,bound1x4,'black');
hold on
plot(bound2x3,bound2x4,'red');
hold on
plot(bound2extrax3,bound2extrax4,'red');
xlabel('x3'); ylabel('x4');
title('Decision boundries for case 4')

end