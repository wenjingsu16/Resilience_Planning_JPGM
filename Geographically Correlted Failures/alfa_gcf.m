
data=readmatrix('cost_gcf.csv')
% data=readmatrix('usenergy_gcf.csv')
Z_full = data([1:100],[2:30])

samp_n=15

Z = Z_full;
[h,L] = size(Z);
z_bar = mean(Z,2);
A = Z-z_bar;
%A=Z
lf_n = min(samp_n-1,L-1);
%lf_n = max(lf_n,2);


p = ones(1,h)./h;

zeta = mean(p*Z_full);


[U,S,V] = svd(A,'econ');

% plot(diag(S));
% saveas(gcf,'PC_gcf_cost10.png')
% saveas(gcf,'PC_gcf_usenergy10.png');

Uk = U(:,1:lf_n);
Sk = S(1:lf_n,1:lf_n);

W = Uk*Sk;

samp_scenarios = candexch(W, samp_n, 'tries', 100, 'display', 'off');
lf_nn = lf_n;
count = 1;
while (length(samp_scenarios) > length(unique(samp_scenarios))) && count<10
    lf_nn = lf_nn + 1;
    Unn = U(:,1:lf_nn);
    Snn = S(1:lf_nn,1:lf_nn);
    Wnn = Unn*Snn;
    samp_scenarios = candexch(Wnn, samp_n, 'tries', 100, 'display', 'off');
    count = count+1;
end


W_samp = W(samp_scenarios,:);


alpha = mean((Uk*Sk)*(((W_samp'*W_samp)^(-1))*W_samp'));

% otherwise use q*H
H = U(:,1:lf_n)*S(1:lf_n,1:lf_n)*((W_samp'*W_samp)\(W_samp'));    
alpha_check = p*H;

alfa_data.g_mean = zeta;
alfa_data.hr_mean = z_bar;
alfa_data.w = alpha_check;
alfa_data.samp_scenarios = samp_scenarios;

%% Make Guaranteed Positive Weights
A_apx_samp = A(samp_scenarios,:)';
target_cost = (p*A)';
fun = @(x)sum((A_apx_samp*x-target_cost).^2);
w_pos = fmincon(fun, abs(alpha*2)',[],[],[],[],.01.*p(samp_scenarios)',[]);
alpha_pos = w_pos';
alfa_data.w_pos = w_pos;

%% Make full alpha approx
%apx_pos = (Z_full(:,samp_hrs) - z_bar(samp_hrs)')*alpha_pos'+ zeta;
apx_pos = (alpha_pos*(Z_full(samp_scenarios,:) - z_bar(samp_scenarios)))'+ zeta;
apx_nat = (alpha_check*(Z_full(samp_scenarios,:) - z_bar(samp_scenarios)))'+ zeta;
full_cost = mean(Z_full);
expected_cost = p*Z_full;

plot(full_cost)
hold on
%plot(expected_cost)
%plot(apx_nat)
plot(apx_pos)
apx_out = apx_pos;

%legend('full cost','expected cost','apx nat','apx pos')
legend('full cost','apx pos');
% saveas(gcf,'ALFA Approximate GCF Usenergy 10.png');
% saveas(gcf,'ALFA Approximate GCF Cost 10.png');

%% output the sample scenarios
%writematrix(samp_scenarios,'gcf_usenergy10.txt');
%writematrix(alpha_pos,'gcf_usenergy_weight10.csv');

writematrix(samp_scenarios,'gcf_cost15.txt');
writematrix(alpha_pos,'gcf_cost_weight15.csv');

transpose(samp_scenarios);

