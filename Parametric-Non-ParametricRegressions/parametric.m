%Parametric Regression
function parametric()

fileID1 = fopen('training.txt');
C1 = textscan(fileID1,'%f %f','Delimiter',' ');
fclose(fileID1);

fileID2 = fopen('validation.txt');
C2 = textscan(fileID2,'%f %f','Delimiter',' ');
fclose(fileID2);

%X_train is training data set. 1st column is x values, 
%2nd column is corresponding r values
X_train = zeros(25,2);
X_train(:,1) = C1{1};
X_train(:,2) = C1{2};

%Sorting to plot example polynomial fits easily
s = X_train(:,1);
r = X_train(:,2);
[s_sorted,sorted_index] = sort(s);
X_train(:,1) = s_sorted;
for i=1:25
    X_train(i,2) = r(sorted_index(i));
end

%X_val is validation data set. 1st column is x values, 
%2nd column is corresponding r values
X_val = zeros(25,2);
X_val(:,1) = C2{1};
X_val(:,2) = C2{2};

E_train = zeros(11,2);
E_val = zeros(11,2);
G_train = zeros(25,11);

for i=1:11
    D = zeros(25,i);
    for j=1:i
        D(:,j) = X_train(:,1).^(j-1);
    end
    w = inv(D'*D)*D'*X_train(:,2);
    tot_Error_train = 0;
    tot_Error_val = 0;
    for t=1:25
        g_train = 0;
        g_val = 0;
        for k=1:length(w)
            g_train = g_train + w(k)*(X_train(t,1)^(k-1));
            g_val = g_val + w(k)*(X_val(t,1)^(k-1));
        end
        G_train(t,i) = g_train;
        tot_Error_train = tot_Error_train + (X_train(t,2)-g_train)^2;
        tot_Error_val = tot_Error_val + (X_val(t,2)-g_val)^2;
    end
    Error_val = tot_Error_val/25; 
    E_val(i,1) = i-1;
    E_val(i,2) = Error_val;
    Error_train = tot_Error_train/25; 
    E_train(i,1) = i-1;
    E_train(i,2) = Error_train;
end

figure
plot(E_train(:,1),E_train(:,2),'LineWidth',2)
hold on
plot(E_val(:,1),E_val(:,2),'red','LineWidth',2)
title('Mean Square Error')
xlabel('polynomial degree')
ylabel('error')
legend('training error','validation error')

figure
plot(X_train(:,1),G_train(:,3),'green','LineWidth',1.2)
hold on
plot(X_train(:,1),G_train(:,7),'black','LineWidth',1.2)
hold on
plot(X_train(:,1),G_train(:,11),'red','LineWidth',1.2)
legend('underfit','goodfit','overfit')
hold on
for i=1:25
    plot(X_train(i,1),X_train(i,2),'o','MarkerSize',8)
    hold on
end

end