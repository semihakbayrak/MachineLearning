function kNN_after_feature_selection()

fileID = fopen('data.txt');
C = textscan(fileID,'%f %f %f %f %f %f %f %f %f %f %f %f %f %f','Delimiter',',');
fclose(fileID);

%Xall is feature selection processed data matrix. Rows are instances, 
%columns are features
Xall = zeros(297,7);
Xall(:,1) = C{13};
Xall(:,2) = C{3};
Xall(:,3) = C{12};
Xall(:,4) = C{10};
Xall(:,5) = C{9};
Xall(:,6) = C{7};
Xall(:,7) = C{2};
%Rall is output matrix
Rall = C{14};

%2 class case, heart disease is exist or not
v = Rall;
v(v>0) = 1;
Rall = v;

Xtrain = zeros(100,7);
Xval = zeros(100,7);
Xtest = zeros(97,7);
Rtrain = zeros(100,1);
Rval = zeros(100,1);
Rtest = zeros(97,1);
for i=1:297
    if i<=100
        Xtrain(i,:) = Xall(i,:);
        Rtrain(i) = Rall(i);
    elseif i<=200
        Xval(i-100,:) = Xall(i,:);
        Rval(i-100) = Rall(i);
    else
        Xtest(i-200,:) = Xall(i,:);
        Rtest(i-200) = Rall(i);
    end
end

%SMD_val(i,j) is index of the jth nearest training set instance neighbor of 
%ith validation set instance
SMD_val = zeros(100,100);
for i=1:100
    MA = mahalanobisArray(Xtrain,Xval(i,:));
    [Msorted,is] = sort(MA,'ascend');
    SMD_val(i,:) = is;
end

%SMD_train(i,j) is index of the jth nearest training set instance neighbor of 
%ith training set instance
SMD_train = zeros(100,100);
for i=1:100
    MA = mahalanobisArray(Xtrain,Xtrain(i,:));
    [Msorted,is] = sort(MA,'ascend');
    SMD_train(i,:) = is;
end

%SMD_test(i,j) is index of the jth nearest training set instance neighbor of 
%ith test set instance
SMD_test = zeros(97,100);
for i=1:97
    MA = mahalanobisArray(Xtrain,Xtest(i,:));
    [Msorted,is] = sort(MA,'ascend');
    SMD_test(i,:) = is;
end

Etrain = zeros(50,2);
Eval = zeros(50,2);
for k=1:2:99
    true_train = 0;
    true_val = 0;
    for i=1:100
        t_0 = 0;
        t_1 = 0;
        v_0 = 0;
        v_1 = 0;
        for j=1:k
            if Rtrain(SMD_train(i,j)) == 0
                t_0 = t_0 + 1;
            else
                t_1 = t_1 + 1;
            end
            if Rtrain(SMD_val(i,j)) == 0
                v_0 = v_0 + 1;
            else
                v_1 = v_1 + 1;
            end
        end
        if t_0>t_1
            if Rtrain(i) == 0
                true_train = true_train+1;
            end
        else
            if Rtrain(i) == 1
                true_train = true_train+1;
            end
        end
        if v_0>v_1
            if Rval(i) == 0
                true_val = true_val+1;
            end
        else
            if Rval(i) == 1
                true_val = true_val+1;
            end
        end
    end
    Etrain((k+1)/2,1) = k;
    Etrain((k+1)/2,2) = true_train;
    Eval((k+1)/2,1) = k;
    Eval((k+1)/2,2) = true_val;
end
Etrain
Eval

%In validation, k=9 gives highest accuracy
for k=9
    true_test = 0;
    predicted = [0 0];
    actual = [0 0];
    for i=1:97
        t_0 = 0;
        t_1 = 0;
        for j=1:k
            if Rtrain(SMD_test(i,j)) == 0
                t_0 = t_0 + 1;
            else
                t_1 = t_1 + 1;
            end
        end
        if t_0>t_1
            predicted = predicted + [1 0];
            if Rtest(i) == 0
                true_test = true_test+1;
                actual = actual + [1 0];
            else
                actual = actual + [0 1]; 
            end
        else
            predicted = predicted + [0 1];
            if Rtest(i) == 1
                true_test = true_test+1;
                actual = actual + [0 1];
            else
                actual = actual + [1 0];
            end
        end
    end
end
actual
predicted
true_test

end