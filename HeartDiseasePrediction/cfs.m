function cfs()
fileID = fopen('data.txt');
C = textscan(fileID,'%f %f %f %f %f %f %f %f %f %f %f %f %f %f','Delimiter',',');
fclose(fileID);

%Xall is data matrix. Rows are instances, columns are features.
Xall = zeros(297,13);
Xall(:,1) = C{1};
Xall(:,2) = C{2};
Xall(:,3) = C{3};
Xall(:,4) = C{4};
Xall(:,5) = C{5};
Xall(:,6) = C{6};
Xall(:,7) = C{7};
Xall(:,8) = C{8};
Xall(:,9) = C{9};
Xall(:,10) = C{10};
Xall(:,11) = C{11};
Xall(:,12) = C{12};
Xall(:,13) = C{13};
%Rall is output vector
Rall = C{14};

%2 class case, heart disease is exist or not
v = Rall;
v(v>0) = 1;
Rall = v;

Xtrain = zeros(100,13);
Xval = zeros(100,13);
Xtest = zeros(97,13);
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

%Feature-Class Correlation Matrix
m_y = sum(Rtrain)/100;
R_cf = zeros(1,13);
for i=1:13
    m_x = sum(Xtrain(:,i))/100;
    totx = 0;
    toty = 0;
    totxy = 0;
    for j=1:100
        totxy = totxy + (Xtrain(j,i)-m_x)*(Rtrain(j)-m_y);
        totx = totx + (Xtrain(j,i)-m_x)^2;
        toty = toty + (Rtrain(j)-m_y)^2;
    end
    R_cf(i) = totxy/sqrt(totx*toty);
end

%Feature-Feature Correlation Matrix
R_ff = zeros(1,13);
for i=1:13
    m_x = sum(Xtrain(:,i))/100;
    for k=1:13
        m_y = sum(Xtrain(:,k))/100;
        totx = 0;
        toty = 0;
        totxy = 0;
        for j=1:100
            totxy = totxy + (Xtrain(j,i)-m_x)*(Xtrain(j,k)-m_y);
            totx = totx + (Xtrain(j,i)-m_x)^2;
            toty = toty + (Xtrain(j,k)-m_y)^2;
        end
        R_ff(i,k) = totxy/sqrt(totx*toty);
    end
end

%Finding best feature set
used = zeros(1,13);
r_cf = 0;
r_ff = 0;
MeritAll = zeros(13,2);
for k=1:13
    count = 0;
    for i=1:13
        if ismember(i,used) ~= 1
            count = count + 1;
            temp_rcf = ((k-1)*r_cf + abs(R_cf(i)))/k;
            if (k-1)>=2
                temp_rff =  nchoosek(k-1,2)*r_ff;
            else
                temp_rff = r_ff;
            end
            for j=1:13
                if used(j) == 0
                    break
                else
                    temp_rff = temp_rff + abs(R_ff(used(j),i));
                end
            end
            if k>=2
                temp_rff = temp_rff/nchoosek(k,2);
            end
            if count == 1
                maxMerit = Merit(k,temp_rcf,temp_rff);
                maxInd = i;
                max_rcf = temp_rcf;
                max_rff = temp_rff;
            else
                if Merit(k,temp_rcf,temp_rff)>maxMerit
                    maxMerit = Merit(k,temp_rcf,temp_rff);
                    maxInd = i;
                    max_rcf = temp_rcf;
                    max_rff = temp_rff;
                end
            end
        end
    end
    r_cf = max_rcf;
    r_ff = max_rff;
    used(k) = maxInd
    maxMerit
    MeritAll(k,1) = k;
    MeritAll(k,2) = maxMerit;
end

    function m = Merit(k,r_cf,r_ff)
        m = k*r_cf/sqrt(k+k*(k-1)*r_ff);
    end

figure
plot(MeritAll(:,1),MeritAll(:,2));
xlabel('number of features')
ylabel('Merit')

end