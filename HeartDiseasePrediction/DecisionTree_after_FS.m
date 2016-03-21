function DecisionTree_after_FS()

fileID = fopen('data.txt');
C = textscan(fileID,'%f %f %f %f %f %f %f %f %f %f %f %f %f %f','Delimiter',',');
fclose(fileID);

%Xall is feature selection processed data matrix. Rows are instances, 
%columns are features
Xall = zeros(297,7);
Xall(:,1) = C{13}; %Discrete
Xall(:,2) = C{3}; %Discrete
Xall(:,3) = C{12}; %Discrete
Xall(:,4) = C{10}; %Numeric
Xall(:,5) = C{9}; %Discrete
Xall(:,6) = C{7}; %Discrete
Xall(:,7) = C{2}; %Discrete
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

function Tree(X,R,threshold,FeatureSet,FeatureVal,FeatureSit)
    IMALL = zeros(1,7);
    for iT=1:7
        if iT ~= 4
            unique_vals = unique(X(:,iT));
            Im_sp_mj = zeros(1,length(unique_vals));
            p0_mj = zeros(1,length(unique_vals));
            p1_mj = zeros(1,length(unique_vals));
            for jT=1:length(unique_vals)
                for kT=1:length(R)
                    if R(kT)==0 && X(kT,iT)==unique_vals(jT)
                        p0_mj(jT) = p0_mj(jT) + 1;
                    end
                    if R(kT)==1 && X(kT,iT)==unique_vals(jT)
                        p1_mj(jT) = p1_mj(jT) + 1;
                    end
                end
            end
            N_mj = p0_mj + p1_mj;
            ratio_N_mj = N_mj/length(R);
            p0_mj = p0_mj./N_mj;
            p1_mj = p1_mj./N_mj;
            Im_sp_mj = ratio_N_mj.*(p0_mj.*log2(p0_mj)+p1_mj.*log2(p1_mj));
            Im_sp_mj(isnan(Im_sp_mj)) = 0;
            Im_split = -1*sum(Im_sp_mj);
            IMALL(iT) = Im_split;
        else
            unique_vals = unique(X(:,iT));
            if length(unique_vals)>1
                Im_sp_in = zeros(1,length(unique_vals)-1);
                for kT=1:(length(unique_vals)-1)
                    teta = (unique_vals(kT)+unique_vals(kT+1))/2;
                    Im_sp_mj = zeros(1,2);
                    p0_mj = zeros(1,2);
                    p1_mj = zeros(1,2);
                    for mT=1:length(R)
                        if R(mT)==0 && X(mT,iT)<teta
                            p0_mj(1) = p0_mj(1) + 1;
                        end
                        if R(mT)==0 && X(mT,iT)>teta
                            p0_mj(2) = p0_mj(2) + 1;
                        end
                        if R(mT)==1 && X(mT,iT)<teta
                            p1_mj(1) = p1_mj(1) + 1;
                        end
                        if R(mT)==1 && X(mT,iT)>teta
                            p1_mj(2) = p1_mj(2) + 1;
                        end
                    end
                    N_mj = p0_mj + p1_mj;
                    ratio_N_mj = N_mj/length(R);
                    p0_mj = p0_mj./N_mj;
                    p1_mj = p1_mj./N_mj;
                    Im_sp_mj = ratio_N_mj.*(p0_mj.*log2(p0_mj)+p1_mj.*log2(p1_mj));
                    Im_sp_mj(isnan(Im_sp_mj)) = 0;
                    Im_split = -1*sum(Im_sp_mj);
                    Im_sp_in(kT) = Im_split;
                end
                [m_im, m_ind] = min(Im_sp_in);
                teta = (unique_vals(m_ind)+unique_vals(m_ind+1))/2;
                IMALL(iT) = m_im;
            else
                IMALL(iT) = 1;
            end
        end
    end
    [m_Im, m_Ind] = min(IMALL);
    if m_Ind ~= 4
        unique_vals = unique(X(:,m_Ind));
        CX = cell(1,length(unique_vals));
        CR = cell(1,length(unique_vals));
        for iT=1:length(R)
            CX{find(unique_vals==X(iT,m_Ind))} = [CX{find(unique_vals==X(iT,m_Ind))};X(iT,:)];
            CR{find(unique_vals==X(iT,m_Ind))} = [CR{find(unique_vals==X(iT,m_Ind))};R(iT)];
        end
        for iT=1:length(unique_vals)
            pCRes = purityCheck(CR{iT},threshold);
            if pCRes(1) == 0
                FSet = [FeatureSet m_Ind];
                FVal = [FeatureVal unique_vals(iT)];
                FSit = [FeatureSit 0];
                Tree(CX{iT},CR{iT},threshold,FSet,FVal,FSit);
            else
                FSet = [FeatureSet m_Ind];
                FVal = [FeatureVal unique_vals(iT)];
                FSit = [FeatureSit 0];
                CellSet{end+1} = FSet;
                CellVal{end+1} = FVal;
                CellSit{end+1} = FSit;
                CellLabel{end+1} = pCRes(2);
            end
        end
    else
        CX = cell(1,2);
        CR = cell(1,2);
        for iT=1:length(R)
            if X(iT,m_Ind)<teta
                CX{1} = [CX{1};X(iT,:)];
                CR{1} = [CR{1};R(iT)];
            else
                CX{2} = [CX{2};X(iT,:)];
                CR{2} = [CR{2};R(iT)];
            end
        end
        for iT=1:2
            pCRes = purityCheck(CR{iT},threshold);
            if pCRes(1) == 0
                FSet = [FeatureSet m_Ind];
                FVal = [FeatureVal teta];
                if iT==1
                    FSit = [FeatureSit -1];
                else
                    FSit = [FeatureSit 1];
                end
                Tree(CX{iT},CR{iT},threshold,FSet,FVal,FSit);
            else
                FSet = [FeatureSet m_Ind];
                FVal = [FeatureVal teta];
                if iT==1
                    FSit = [FeatureSit -1];
                else
                    FSit = [FeatureSit 1];
                end
                CellSet{end+1} = FSet;
                CellVal{end+1} = FVal;
                CellSit{end+1} = FSit;
                CellLabel{end+1} = pCRes(2);
            end
        end
    end
end

function p = purityCheck(R,threshold)
    count0 = 0;
    count1 = 0;
    for iP=1:length(R)
        if R(iP)==0
            count0 = count0 + 1;
        else
            count1 = count1 + 1;
        end
    end
    p0 = count0/(count0+count1);
    p1 = count1/(count0+count1);
    if p0 == 0
        L0 = 0;
    else
        L0 = log2(p0);
    end
    if p1 == 0
        L1 = 0;
    else
        L1 = log2(p1);
    end
    if p0>p1
        label = 0;
    else
        label = 1;
    end
    imp = -(p0*L0+p1*L1);
    if imp<=threshold
        p=[1 label];
    else
        p=[0 label];
    end
end

%Cross-Validation
EVal = zeros(8,2);
for i=0.3:0.1:1
    CellSet = {};
    CellVal = {};
    CellSit = {};
    CellLabel = {};
    Fset = [];
    Fval = [];
    Fsit = [];
    Tree(Xtrain,Rtrain,i,Fset,Fval,Fsit)

    sz = size(CellLabel);
    cellsize = sz(2);
    PredictedVal = zeros(100,1);
    for t=1:100
        for j=1:cellsize
            rulesize = length(CellSit{j});
            rulecount = 0;
            for k=1:rulesize
                if CellSit{j}(k)==1
                    if Xval(t,CellSet{j}(k))>CellVal{j}(k)
                        rulecount = rulecount + 1;
                    else
                        break
                    end
                elseif CellSit{j}(k)==-1
                    if Xval(t,CellSet{j}(k))<CellVal{j}(k)
                        rulecount = rulecount + 1;
                    else
                        break
                    end
                else
                    if Xval(t,CellSet{j}(k))==CellVal{j}(k)
                        rulecount = rulecount + 1;
                    else
                        break
                    end
                end
            end
            if rulecount==rulesize
                PredictedVal(t) = CellLabel{j};
                break
            end
        end
    end
    truecount = 0;
    for t=1:100
        if PredictedVal(t) == Rval(t)
            truecount = truecount + 1;
        end
    end
    EVal(round((i-0.2)*10),1) = i;
    EVal(round((i-0.2)*10),2) = truecount;
end
EVal;

%This is for drawing the tree for threshold=0.5
CellSet = {};
CellVal = {};
CellSit = {};
CellLabel = {};
Fset = [];
Fval = [];
Fsit = [];
Tree(Xtrain,Rtrain,0.5,Fset,Fval,Fsit)
sz = size(CellLabel);
cellsize = sz(2);
for i=1:cellsize
    CellSet{i}
    CellVal{i}
    CellSit{i}
    CellLabel{i}
end

%Validation gives best result for threshold=0.5
for i=0.5
    CellSet = {};
    CellVal = {};
    CellSit = {};
    CellLabel = {};
    Fset = [];
    Fval = [];
    Fsit = [];
    Tree(Xtrain,Rtrain,i,Fset,Fval,Fsit)

    sz = size(CellLabel);
    cellsize = sz(2);
    PredictedTest = zeros(97,1);
    for t=1:97
        for j=1:cellsize
            rulesize = length(CellSit{j});
            rulecount = 0;
            for k=1:rulesize
                if CellSit{j}(k)==1
                    if Xtest(t,CellSet{j}(k))>CellVal{j}(k)
                        rulecount = rulecount + 1;
                    else
                        break
                    end
                elseif CellSit{j}(k)==-1
                    if Xtest(t,CellSet{j}(k))<CellVal{j}(k)
                        rulecount = rulecount + 1;
                    else
                        break
                    end
                else
                    if Xtest(t,CellSet{j}(k))==CellVal{j}(k)
                        rulecount = rulecount + 1;
                    else
                        break
                    end
                end
            end
            if rulecount==rulesize
                PredictedTest(t) = CellLabel{j};
                break
            end
        end
    end
    truecount = 0;
    for t=1:97
        if PredictedTest(t) == Rtest(t)
            truecount = truecount + 1;
        end
    end
end
truecount

end