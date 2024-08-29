close all;
clear;
set(groot,'defaultLineLineWidth',3)
set(0,'DefaultaxesLineWidth', 1.5)
set(0,'DefaultaxesFontSize', 14)
set(0,'DefaultaxesFontWeight', 'bold')
set(0,'DefaultTextFontSize', 18)
set(0,'DefaultaxesFontName', 'Times new Roman')
set(0,'DefaultlegendFontName', 'Times new Roman')
set(0,'defaultAxesXGrid','on')
set(0,'defaultAxesYGrid','on')
%%
% no control
%load no_control_high_noise.mat
% load no_control_ERR_sparse_150_1_test_NewNoise.mat
load no_control_ERR_sparseloop_150_1_test_HighNoise.mat
N = 20;
m = 5;
T = length(EST);
p_error_no_control = zeros(T,m);
th_error_no_control = zeros(T,m);
for t = 1:T
    for i = 1:N
        for j = 1:m
            p_error_no_control(t,j) = p_error_no_control(t,j) + sum((EST(i,j,1:2,t) - TRUTH(i,j,1:2,t)).^2);%10X5X3X1500
            th_error_no_control(t,j) = th_error_no_control(t,j) + sum((EST(i,j,3,t) - TRUTH(i,j,3,t)).^2);%10X5X3X1500
        end
    end
end
colorcode ='rbgck';
figure;
for i = 1:m
    plot(reshape(TRUTH(1,i,1,:),[],1), reshape(TRUTH(1,i,2,:),[],1),colorcode(i))
    Leg{i} = ['vehicle ' num2str(i)];
    hold on;
    %plot(posf(1, i), posf(2,i), ['s' colorcode(i)],'MarkerSize',10)
end
grid on
xlabel('x (meters)')
ylabel('y (meters)')
title('basic guidance')
%ylim([-500 2500])
axis equal
for i = 1:m
    plot(posf(1, i), posf(2,i), ['d' colorcode(i)],'MarkerSize',10)
    Leg{m+i} = ['vehicle ' num2str(i) ': goal'];
end
legend(Leg,'location','best');

Ts = length(ERR);
p_ERR_no_control = zeros(Ts,m);
th_ERR_no_control = zeros(Ts,m);
for t = 1:Ts
    for i = 1:N
        for j = 1:m
            p_ERR_no_control(t,j) = p_ERR_no_control(t,j) + sum(ERR(i,j,1:2,t).^2);%10X5X3X1500
            th_ERR_no_control(t,j) = th_ERR_no_control(t,j) + sum(ERR(i,j,3,t).^2);%10X5X3X1500
        end
    end
end



%% full control
clear EST TRUTH ERR;
%load control_high_noise.mat %_ERR_150.mat
% load control_ERR_sparse_150_1_test_NewNoise_obsv.mat
load control_ERR_sparseloop_150_1_test_HighNoise_obsv.mat
N = 20;
m = 5;
T = length(EST);
p_error_control = zeros(T,m);
th_error_control = zeros(T,m);
for t = 1:T
    for i = 1:N
        for j = 1:m
            p_error_control(t,j) = p_error_control(t,j) + sum((EST(i,j,1:2,t) - TRUTH(i,j,1:2,t)).^2);%10X5X3X1500
            th_error_control(t,j) = th_error_control(t,j) + sum((EST(i,j,3,t) - TRUTH(i,j,3,t)).^2);%10X5X3X1500
        end
    end
end
figure;
for i = 1:m
    plot(reshape(TRUTH(1,i,1,:),[],1), reshape(TRUTH(1,i,2,:),[],1), colorcode(i))
    Leg{i} = ['vehicle ' num2str(i)];
    hold on;
    %plot(posf(1, i), posf(2,i), ['s' colorcode(i)],'MarkerSize',10)
end
grid on
xlabel('x (meters)')
ylabel('y (meters)')
title('active guidance (obsv)')
%ylim([-500 2500])
axis equal
for i = 1:m
    plot(posf(1, i), posf(2,i), ['d' colorcode(i)],'MarkerSize',10)
    Leg{m+i} = ['vehicle ' num2str(i) ': goal'];
end
legend(Leg,'location','best');


Ts = length(ERR);
p_ERR_control = zeros(Ts,m);
th_ERR_control = zeros(Ts,m);
for t = 1:Ts
    for i = 1:N
        for j = 1:m
            p_ERR_control(t,j) = p_ERR_control(t,j) + sum(ERR(i,j,1:2,t).^2);%10X5X3X1500
            th_ERR_control(t,j) = th_ERR_control(t,j) + sum(ERR(i,j,3,t).^2);%10X5X3X1500
        end
    end
end
% figure;
% plot(sqrt(sum(p_ERR_no_control,2)/N),'-')
% hold on
% plot(sqrt(sum(p_ERR_control,2)/N),'-.')
% legend('no control','control')
%%


% % full control with obsv
% load control_obsv.mat
% N = 10;
% m = 5;
% T = 1000;
% p_error_control_obsv = zeros(T,m);
% th_error_control_obsv = zeros(T,m);
% for t = 1:T
%     for i = 1:N
%         for j = 1:m
%             p_error_control_obsv(t,j) = p_error_control_obsv(t,j) + sum((EST(i,j,1:2,t) - TRUTH(i,j,1:2,t)).^2);%10X5X3X1500
%             th_error_control_obsv(t,j) = th_error_control_obsv(t,j) + sum((EST(i,j,3,t) - TRUTH(i,j,3,t)).^2);%10X5X3X1500
%         end
%     end
% end
% figure;
% for i = 1:m
%     plot(reshape(TRUTH(i,1,1,:),[],1), reshape(TRUTH(i,1,2,:),[],1))
%     Leg{i} = ['vehicle ' num2str(i)];
%     hold on;
% end
% legend(Leg);
% grid on
% xlabel('x (meters)')
% ylabel('y (meters)')

% figure;
% plot(sqrt(sum(p_error_no_control,2)/N),'-')
% hold on
% plot(sqrt(sum(p_error_control,2)/N),'-.')
% %plot(sqrt(sum(p_error_control_obsv,2)/N),'-.')
% legend('no active guidance', 'with active guidance (SAM)','location','best')%, 'with optimization (obsv)')
% grid on
% xlabel('time step (x 0.1 sec = time)')
% ylabel('position RMSE (meters)')
% figure;
% plot(sqrt(sum(th_error_no_control,2)/N),'.','MarkerSize',10)
% hold on
% plot(sqrt(sum(th_error_control,2)/N),'.','MarkerSize',10)
% %plot(sqrt(sum(th_error_control_obsv,2)/N),'-')
%
% legend('no active guidance', 'with active guidance (SAM)','location','best')%, 'with optimization (obsv)')
% grid on
% xlabel('time step (x 0.1 sec = time)')
% ylabel('heading RMSE (rad)')




% %%
% % no control with complete graph
% clear EST TRUTH;
% %load no_control_high_noise_full.mat%ERR_full_150_1.mat
%
% load no_control_ERR_150_1_test_NewNoise.mat
% N = 10;
% m = 5;
% T = length(EST);
% p_error_no_control_full = zeros(T,m);
% th_error_no_control_full = zeros(T,m);
% for t = 1:T
%     for i = 1:N
%         for j = 1:m
%             p_error_no_control_full(t,j) = p_error_no_control_full(t,j) + sum((EST(i,j,1:2,t) - TRUTH(i,j,1:2,t)).^2);%10X5X3X1500
%             th_error_no_control_full(t,j) = th_error_no_control_full(t,j) + sum((EST(i,j,3,t) - TRUTH(i,j,3,t)).^2);%10X5X3X1500
%         end
%     end
% end
% figure;
% for i = 1:m
%     plot(reshape(TRUTH(1,i,1,:),[],1), reshape(TRUTH(1,i,2,:),[],1), [colorcode(i)])
%     Leg{i} = ['vehicle ' num2str(i)];
%     hold on;
% end
% legend(Leg,'location','best');
% grid on
% xlabel('x (meters)')
% ylabel('y (meters)')
% title('without active guidance')
% %ylim([-500 2500])
% axis square
% for i = 1:m
%     plot(posf(1, i), posf(2,i), ['d' colorcode(i)],'MarkerSize',10)
% end
% Ts = length(ERR);
% p_ERR_no_control_full = zeros(Ts,m);
% th_ERR_no_control_full = zeros(Ts,m);
% for t = 1:Ts
%     for i = 1:N
%         for j = 1:m
%             p_ERR_no_control_full(t,j) = p_ERR_no_control_full(t,j) + sum(ERR(i,j,1:2,t).^2);%10X5X3X1500
%             th_ERR_no_control_full(t,j) = th_ERR_no_control_full(t,j) + sum(ERR(i,j,3,t).^2);%10X5X3X1500
%         end
%     end
% end
%%
clear EST TRUTH;
%load control_high_noise_full.mat%ERR_full_150.mat
% load control_ERR_sparse_150_1_test_NewNoise_SAM.mat
load control_ERR_sparseloop_150_1_test_HighNoise_SAM.mat
N = 20;
m = 5;
T = length(EST);
p_error_control_SAM = zeros(T,m);
th_error_control_SAM = zeros(T,m);
for t = 1:T
    for i = 1:N
        for j = 1:m
            p_error_control_SAM(t,j) = p_error_control_SAM(t,j) + sum((EST(i,j,1:2,t) - TRUTH(i,j,1:2,t)).^2);%10X5X3X1500
            th_error_control_SAM(t,j) = th_error_control_SAM(t,j) + sum((EST(i,j,3,t) - TRUTH(i,j,3,t)).^2);%10X5X3X1500
        end
    end
end
figure;
for i = 1:m
    plot(reshape(TRUTH(2,i,1,:),[],1), reshape(TRUTH(2,i,2,:),[],1), [colorcode(i)],'MarkerSize',10)
    Leg{i} = ['vehicle ' num2str(i)];
    hold on;
end
grid on
xlabel('x (meters)')
ylabel('y (meters)')
title('active guidance (SAM)')
%ylim([-500 2500])
axis equal
for i = 1:m
    plot(posf(1, i), posf(2,i), ['d' colorcode(i)],'MarkerSize',10)
    Leg{m+i} = ['vehicle ' num2str(i) ': goal'];
end
legend(Leg,'location','best');

Ts = length(ERR);

p_ERR_control_SAM = zeros(Ts,m);
th_ERR_control_SAM = zeros(Ts,m);
for t = 1:Ts
    for i = 1:N
        for j = 1:m
            p_ERR_control_SAM(t,j) = p_ERR_control_SAM(t,j) + sum(ERR(i,j,1:2,t).^2);%10X5X3X1500
            th_ERR_control_SAM(t,j) = th_ERR_control_SAM(t,j) + sum(ERR(i,j,3,t).^2);%10X5X3X1500
        end
    end
end
% %%
% clear EST TRUTH;
% %load control_high_noise_full.mat%ERR_full_150.mat
% load control_ERR_150_1_test_NewNoise_min_eig_SAM.mat
% N = 20;
% m = 5;
% T = length(EST);
% p_error_control_mineigSAM = zeros(T,m);
% th_error_control_mineigSAM = zeros(T,m);
% for t = 1:T
%     for i = 1:N
%         for j = 1:m
%             p_error_control_mineigSAM(t,j) = p_error_control_mineigSAM(t,j) + sum((EST(i,j,1:2,t) - TRUTH(i,j,1:2,t)).^2);%10X5X3X1500
%             th_error_control_mineigSAM(t,j) = th_error_control_mineigSAM(t,j) + sum((EST(i,j,3,t) - TRUTH(i,j,3,t)).^2);%10X5X3X1500
%         end
%     end
% end
% figure;
% for i = 1:m
%     plot(reshape(TRUTH(1,i,1,:),[],1), reshape(TRUTH(1,i,2,:),[],1), [colorcode(i)],'MarkerSize',10)
%     Leg{i} = ['vehicle ' num2str(i)];
%     hold on;
% end
% grid on
% xlabel('x (meters)')
% ylabel('y (meters)')
% title('active guidance (min eigen value SAM)')
% %ylim([-500 2500])
% axis equal
% for i = 1:m
%     plot(posf(1, i), posf(2,i), ['d' colorcode(i)],'MarkerSize',10)
%     Leg{m+i} = ['vehicle ' num2str(i) ': goal'];
% end
% legend(Leg,'location','best');
% 
% Ts = length(ERR);
% 
% p_ERR_control_mineigSAM = zeros(Ts,m);
% th_ERR_control_mineigSAM = zeros(Ts,m);
% for t = 1:Ts
%     for i = 1:N
%         for j = 1:m
%             p_ERR_control_mineigSAM(t,j) = p_ERR_control_mineigSAM(t,j) + sum(ERR(i,j,1:2,t).^2);%10X5X3X1500
%             th_ERR_control_mineigSAM(t,j) = th_ERR_control_mineigSAM(t,j) + sum(ERR(i,j,3,t).^2);%10X5X3X1500
%         end
%     end
% end
%%
clear EST TRUTH;
% load control_ERR_sparse_150_1_test_NewNoise_inv_cov.mat
load control_ERR_sparseloop_150_1_test_HighNoise_inv_cov.mat
% load control_ERR_sparseloop_150_1_test_HighNoise_powell_obsv.mat
N = 20;
m = 5;
T = length(EST);
p_error_control_invcov = zeros(T,m);
th_error_control_invcov = zeros(T,m);
for t = 1:T
    for i = 1:N
        for j = 1:m
            p_error_control_invcov(t,j) = p_error_control_invcov(t,j) + sum((EST(i,j,1:2,t) - TRUTH(i,j,1:2,t)).^2);%10X5X3X1500
            th_error_control_invcov(t,j) = th_error_control_invcov(t,j) + sum((EST(i,j,3,t) - TRUTH(i,j,3,t)).^2);%10X5X3X1500
        end
    end
end
figure;
for i = 1:m
    plot(reshape(TRUTH(1,i,1,:),[],1), reshape(TRUTH(1,i,2,:),[],1), [colorcode(i)],'MarkerSize',10)
    Leg{i} = ['vehicle ' num2str(i)];
    hold on;
end
grid on
xlabel('x (meters)')
ylabel('y (meters)')
title('active guidance (EKF inv cov)')
%ylim([-500 2500])
axis equal
for i = 1:m
    plot(posf(1, i), posf(2,i), ['d' colorcode(i)],'MarkerSize',10)
    Leg{m+i} = ['vehicle ' num2str(i) ': goal'];
end
legend(Leg,'location','best');

Ts = length(ERR);

p_ERR_control_invcov = zeros(Ts,m);
th_ERR_control_invcov = zeros(Ts,m);
for t = 1:Ts
    for i = 1:N
        for j = 1:m
            p_ERR_control_invcov(t,j) = p_ERR_control_invcov(t,j) + sum(ERR(i,j,1:2,t).^2);%10X5X3X1500
            th_ERR_control_invcov(t,j) = th_ERR_control_invcov(t,j) + sum(ERR(i,j,3,t).^2);%10X5X3X1500
        end
    end
end


%%
clear EST TRUTH;
% load control_ERR_sparse_150_1_test_NewNoise_det_inv_cov.mat
% load control_ERR_sparseloop_150_1_test_HighNoise_det_inv_cov.mat
load control_ERR_sparseloop_150_1_test_HighNoise_powell_SAM.mat
N = 20;
m = 5;
T = length(EST);
p_error_control_detinvcov = zeros(T,m);
th_error_control_detinvcov = zeros(T,m);
for t = 1:T
    for i = 1:N
        for j = 1:m
            p_error_control_detinvcov(t,j) = p_error_control_detinvcov(t,j) + sum((EST(i,j,1:2,t) - TRUTH(i,j,1:2,t)).^2);%10X5X3X1500
            th_error_control_detinvcov(t,j) = th_error_control_detinvcov(t,j) + sum((EST(i,j,3,t) - TRUTH(i,j,3,t)).^2);%10X5X3X1500
        end
    end
end
figure;
for i = 1:m
    plot(reshape(TRUTH(1,i,1,:),[],1), reshape(TRUTH(1,i,2,:),[],1), [colorcode(i)],'MarkerSize',10)
    Leg{i} = ['vehicle ' num2str(i)];
    hold on;
end
grid on
xlabel('x (meters)')
ylabel('y (meters)')
title('active guidance (EKF det inv cov)')
%ylim([-500 2500])
axis equal
for i = 1:m
    plot(posf(1, i), posf(2,i), ['d' colorcode(i)],'MarkerSize',10)
    Leg{m+i} = ['vehicle ' num2str(i) ': goal'];
end
legend(Leg,'location','best');

Ts = length(ERR);

p_ERR_control_detinvcov = zeros(Ts,m);
th_ERR_control_detinvcov = zeros(Ts,m);
for t = 1:Ts
    for i = 1:N
        for j = 1:m
            p_ERR_control_detinvcov(t,j) = p_ERR_control_detinvcov(t,j) + sum(ERR(i,j,1:2,t).^2);%10X5X3X1500
            th_ERR_control_detinvcov(t,j) = th_ERR_control_detinvcov(t,j) + sum(ERR(i,j,3,t).^2);%10X5X3X1500
        end
    end
end


%%
% figure;
% plot(sqrt(sum(p_ERR_no_control,2)/N),'-')
% hold on
% plot(sqrt(sum(p_ERR_control,2)/N),'-.')
% %plot(sqrt(sum(p_ERR_no_control_full,2)/N),'-.')
% plot(sqrt(sum(p_ERR_control_SAM,2)/N),'-.')
% plot(sqrt(sum(p_ERR_control_invcov,2)/N),'-.')
% legend('basic guidance','control (obsv)' , 'control (complete)', 'control (invcov)')
%%
figure;
plot(sqrt(sum(p_error_no_control,2)/N),'-')
hold on
plot(sqrt(sum(p_error_control,2)/N),'-.')
%plot(sqrt(sum(p_error_no_control_full,2)/N),'-')
hold on;
plot(sqrt(sum(p_error_control_SAM,2)/N),'-.')
hold on;
% plot(sqrt(sum(p_error_control_mineigSAM,2)/N),'-.')
hold on
plot(sqrt(sum(p_error_control_invcov,2)/N),'-.')
hold on
plot(sqrt(sum(p_error_control_detinvcov,2)/N),'-.')
% legend('basic guidance ','active guidance (min $(det($obsv$))^{-1}$)',...
%     'active guidance (min $(det($SAM$))^{-1}$)',  ...
%     'active guidance (max $\lambda_{min}(P^{-1}_{EKF})$)', 'active guidance (min $(det(P^{-1}_{EKF}))^{-1}$)','location','best','interpreter','latex')
legend('basic guidance ','obsv',...
    'SAM',  ...
    'inv_cov', 'powell SAM','location','best','interpreter','latex')
grid on
xlabel('time step (x 0.1 sec = time)')
ylabel('position RMSE (meters)')

figure;
plot(sqrt(sum(th_error_no_control,2)/N),'.','MarkerSize',10)
hold on
plot(sqrt(sum(th_error_control,2)/N),'.')
%plot(sqrt(sum(th_error_no_control_full,2)/N),'.','MarkerSize',10)
hold on
plot(sqrt(sum(th_error_control_SAM,2)/N),'.')
hold on
% plot(sqrt(sum(th_error_control_mineigSAM,2)/N),'.')
hold on
plot(sqrt(sum(th_error_control_invcov,2)/N),'.')
hold on
plot(sqrt(sum(th_error_control_detinvcov,2)/N),'.')
% legend('basic guidance ','active guidance (min $(det($obsv$))^{-1}$)',...
%     'active guidance (min $(det($SAM$))^{-1}$)',  ...
%     'active guidance (max $\lambda_{min}(P^{-1}_{EKF})$)', 'active guidance (min $(det(P^{-1}_{EKF}))^{-1}$)','location','best','interpreter','latex')
legend('basic guidance ','obsv',...
    'SAM',  ...
    'powell obsv', 'powell SAM','location','best','interpreter','latex')
grid on
xlabel('time step (x 0.1 sec = time)')
ylabel('heading RMSE (rad)')


