%Non-Parametric Regression
function non_parametric()

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

    function K = KernelG(u)
        K = exp(-(u^2)/2)/sqrt(2*pi);
    end

    function g = g_func(x,h)
        K_sum = 0;
        K_r_sum = 0;
        for t=1:25
            K_r_sum = K_r_sum + KernelG((x-X_train(t,1))/h) * X_train(t,2);
            K_sum = K_sum + KernelG((x-X_train(t,1))/h);
        end
        g = K_r_sum/K_sum;
    end

%h=0.01:0.01:0.5;
E_train = zeros(50,2);
E_val = zeros(50,2);
G_train = zeros(25,50);
count = 0;
for i=0.01:0.01:0.5
    count = count+1;
    tot_Error_train = 0;
    tot_Error_val = 0;
    for j=1:25
        tot_Error_train = tot_Error_train + (X_train(j,2)-g_func(X_train(j,1),i))^2;
        tot_Error_val = tot_Error_val + (X_val(j,2)-g_func(X_val(j,1),i))^2;
        G_train(j,count) = g_func(X_train(j,1),i);
    end
    Error_val = tot_Error_val/25; 
    E_val(count,1) = i;
    E_val(count,2) = Error_val;
    Error_train = tot_Error_train/25; 
    E_train(count,1) = i;
    E_train(count,2) = Error_train;
end

figure
plot(E_train(:,1),E_train(:,2),'LineWidth',2)
hold on
plot(E_val(:,1),E_val(:,2),'red','LineWidth',2)
title('Mean Square Error')
xlabel('h')
ylabel('error')
legend('training error','validation error')

figure
plot(X_train(:,1),G_train(:,1),'green','LineWidth',1.2)
hold on
plot(X_train(:,1),G_train(:,3),'black','LineWidth',1.2)
hold on
plot(X_train(:,1),G_train(:,10),'red','LineWidth',1.2)
legend('overfit','goodfit','underfit')
hold on
for i=1:25
    plot(X_train(i,1),X_train(i,2),'o','MarkerSize',8)
    hold on
end

end