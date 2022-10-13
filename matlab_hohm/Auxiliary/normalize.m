function zn = normalize(z)

z  = z - min(z(:));
zn = z./max(z(:));

