h1 = subplot(1,2,1);

data = Y_errs_2d_bp;

%heatmap(x_axis, y_axis, Y_errs_2d_lp);
%heatmap(x_axis, y_axis, Y_errs_2d_bp);
%// Define integer grid of coordinates for the above data
[H,V] = meshgrid(1:size(data,2), 1:size(data,1));

%// Define a finer grid of points
[H2,V2] = meshgrid(1:0.01:size(data,2), 1:0.01:size(data,1));

%// Interpolate the data and show the output
outData = interp2(H, V, data, H2, V2, 'linear');
imagesc(outData);

%// Cosmetic changes for the axes
set(gca, 'XTick', linspace(1,size(H2,2),size(H,2))); 
set(gca, 'YTick', linspace(1,size(H2,1),size(H,1)));
set(gca, 'XTickLabel', x_axis);
set(gca, 'YTickLabel', y_axis);

%// Add colour bar

xlabel('Sample Size', 'Interpreter', 'latex');
ylabel('Dimension $d$', 'Interpreter', 'latex');
title('SGD');

colorbar;
c1 = caxis;
colorbar off;

originalSize1 = get(gca, 'Position');

h2 = subplot(1,2,2);

data = Y_errs_2d_lp;

%heatmap(x_axis, y_axis, Y_errs_2d_lp);
%heatmap(x_axis, y_axis, Y_errs_2d_bp);
%// Define integer grid of coordinates for the above data
[H,V] = meshgrid(1:size(data,2), 1:size(data,1));

%// Define a finer grid of points
[H2,V2] = meshgrid(1:0.01:size(data,2), 1:0.01:size(data,1));

%// Interpolate the data and show the output
outData = interp2(H, V, data, H2, V2, 'linear');
imagesc(outData);

%// Cosmetic changes for the axes
set(gca, 'XTick', linspace(1,size(H2,2),size(H,2))); 
set(gca, 'YTick', linspace(1,size(H2,1),size(H,1)));
set(gca, 'XTickLabel', x_axis);
set(gca, 'YTickLabel', y_axis);

%// Add colour bar

xlabel('Sample Size', 'Interpreter', 'latex');
ylabel('Dimension $d$', 'Interpreter', 'latex');
title('Ours');

colorbar;
c2 = caxis;
colorbar off;

originalSize2 = get(gca, 'Position');

c3 = [min([c1 c2]), max([c1 c2])];
caxis(c3);
colormap('jet');
colorbar;

originalSize1 = [0.08, 0.16, 0.35, 0.75];
originalSize2 = [0.54, 0.16, 0.35, 0.75];
set(h1, 'Position', originalSize1);
set(h2, 'Position', originalSize2);