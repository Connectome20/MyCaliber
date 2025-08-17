function h = plotstd(x_mu, y_mu, y_sd, rgb, alpha, option)
%   Code author: Hong-Hsi Lee (orcid: 0000-0002-3663-6559)
%   Massachusetts General Hospital, Harvard Medical School

x_mu = x_mu(:).';
y_mu = y_mu(:).';
y_sd = y_sd(:).';

y_upper = y_mu + y_sd;
y_lower = y_mu - y_sd;

x_fill = [x_mu, fliplr(x_mu)];
y_fill = [y_upper, fliplr(y_lower)];
if strcmpi(option,'area')
    h = fill(x_fill, y_fill, rgb, 'EdgeColor', 'none', 'FaceAlpha', alpha);
else
    h = plot(x_mu,y_upper,x_mu,y_lower,'Color',[rgb, alpha]);
end

end

