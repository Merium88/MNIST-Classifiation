function [Y_test] = KNN_classifier (xtest,xtrain,ytrain,K)
%Calculate euclidean distance/hamming distance based on bin_flag for all testing data
[m,n] = size(xtest);
%for i=1:1:length(xtest(:,1))
for k=1:1:m
  distance = pdist2(xtest(k,:),xtrain,'euclidean');
 %Sum of selected indices from ytrain will give Kc
 %Sort the rows of matrix
[~,sorted_index] = sort(distance,2,'ascend'); %1536x3065 matrix (each column represents neighbours based on distance)
 
 for j=1:1:10
    class_score(j,:) = length(find(ytrain(sorted_index(1:K))==(j-1)));
 end  
    [~,ind] = max(class_score);
    Y_test(k) = ind-1;
%     disp(k);
 end
end