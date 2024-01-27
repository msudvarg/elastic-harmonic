%% Import enumerated data
import_chain_counts;

%% Organize data
n = 2:30;
k = 1:100;
[N,K] = meshgrid(n,k);
chaincounts.n = double(chaincounts.n);
chaincounts.k = double(chaincounts.k);
chaincounts.count = double(chaincounts.count);
Counts = griddata(chaincounts.n, chaincounts.k, chaincounts.count, N, K);

%Compute upper bound per Theorem 4
chaincounts.upperbound = chaincounts.k .* (chaincounts.n - 1) .^ floor(log2(chaincounts.k));

%% Formatting
labelsize = 32;
zlabelsize = 18;
fontsize = 18;

%% Fig. 4(a): Calculated number of PHIs
surf(N,K,Counts);
xlabel("n", 'FontSize', labelsize);
ylabel("k", 'FontSize', labelsize);
zlabel("Number of Chains", 'FontSize', zlabelsize);
set(gca,'zscale','log')
set(gca,'ColorScale','log')
set(gca,'FontSize',fontsize)
zticks([1,1E3,1E6]);
grid minor
%shading interp


%% Optional: Ratio between upper bound and calculated number
figure
Ratio = griddata(chaincounts.n, chaincounts.k, chaincounts.upperbound ./ chaincounts.count, N, K);
surf(N,K,Ratio);
xlabel("n");
ylabel("k");
zlabel("Upper Bound Pessimism")
set(gca,'zscale','log')
set(gca,'ColorScale','log')
%shading interp

%% Fig. 4(b): Upper bound on PHIs
figure
UpperBound = griddata(chaincounts.n, chaincounts.k, chaincounts.upperbound, N, K);
surf(N,K,UpperBound);
xlabel("n", 'FontSize', labelsize);
ylabel("k", 'FontSize', labelsize);
zlabel("Number of Chains", 'FontSize', zlabelsize);
set(gca,'zscale','log')
set(gca,'ColorScale','log')
set(gca,'FontSize',fontsize)
zticks([1,1E5,1E10]);
grid minor
%shading interp

%% Print information to console
fprintf("For n=30 and k=100:\n");
fprintf("Upper bound from Thm. 4: %.1E\n", UpperBound(100,29));
fprintf("Enumerated count: %.1E\n", Counts(100,29));
fprintf("Ratio: %.0f\n", Ratio(100,29));