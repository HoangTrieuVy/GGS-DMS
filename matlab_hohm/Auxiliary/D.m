function y = D(x)
  
y = zeros(size(x,1),size(x,2),2,size(x,3));

y(:,:,1,:) = [x(:,2:end,:) - x(:,1:end-1,:) , zeros(size(x,1),1,size(x,3))] ./ 2.;  % horizontal differences
y(:,:,2,:) = [x(2:end,:,:) - x(1:end-1,:,:) ; zeros(1,size(x,2),size(x,3))] ./ 2.;  % vertical differences

y = squeeze(y); % remove singleton dimensions
  
end