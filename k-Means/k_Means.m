% HW3 k_Means
function k_Means()

A = textread('digits2.txt', '%s');
B = cat(1,A{:})-'0';
%600 digit each of them represented by 1*256 dimensional vectors
Dig = zeros(600,256);
for i=1:600
    for j=((i-1)*16+1):((i-1)*16+16)
        t = j-(i-1)*16;
        for k=((t-1)*16+1):((t-1)*16+16)
            p = k - (t-1)*16;
            Dig(i,k) = B(j,p);
        end
    end
end

m = sum(Dig,1)/ 600;
R_coef = (0.6*(rand(10,256)-0.5));
M_init = zeros(10,256);
for i=1:10
    M_init(i,:) = m+R_coef(i,:).*m;
end

M = M_init;
count = 0;
while count<80
    count = count+1;
    %Assigning each data to clusters
    b = zeros(600,1);
    for i=1:600
        dist = zeros(10,1);
        for j=1:10
            dist(j) = sqrt(sum((Dig(i,:) - M(j,:)) .^ 2));
        end
        [val,ind] = min(dist);
        b(i) = ind;
    end
    %Finding new mean values of clusters
    M = zeros(10,256);
    for i=1:10
        num_elements = 0;
        total = zeros(1,256);
        for j=1:600
            if b(j)==i
                total = total + Dig(j,:);
                num_elements = num_elements + 1;
            end
        end
        M(i,:) = total/num_elements;
    end
end
 
C1 = zeros(16,16);
C2 = zeros(16,16);
C3 = zeros(16,16);
C4 = zeros(16,16);
C5 = zeros(16,16);
C6 = zeros(16,16);
C7 = zeros(16,16);
C8 = zeros(16,16);
C9 = zeros(16,16);
C10 = zeros(16,16);
for i=1:16
    for j=((i-1)*16+1):((i-1)*16+16)
        t = j-(i-1)*16;
        C1(i,t) = M(1,j);
        C2(i,t) = M(2,j);
        C3(i,t) = M(3,j);
        C4(i,t) = M(4,j);
        C5(i,t) = M(5,j);
        C6(i,t) = M(6,j);
        C7(i,t) = M(7,j);
        C8(i,t) = M(8,j);
        C9(i,t) = M(9,j);
        C10(i,t) = M(10,j);
    end
end
        
figure
imagesc(C1);
colormap(flipud(cool));
figure
imagesc(C2);
colormap(flipud(cool));
figure
imagesc(C3);
colormap(flipud(cool));
figure
imagesc(C4);
colormap(flipud(cool));
figure
imagesc(C5);
colormap(flipud(cool));
figure
imagesc(C6);
colormap(flipud(cool));
figure
imagesc(C7);
colormap(flipud(cool));
figure
imagesc(C8);
colormap(flipud(cool));
figure
imagesc(C9);
colormap(flipud(cool));
figure
imagesc(C10);
colormap(flipud(cool));