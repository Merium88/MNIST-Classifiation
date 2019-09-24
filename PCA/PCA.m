function [E,SCORES,SCORES1] = PCA(train_data,test_data,train_labels,test_labels,K)
%PCA Perform Principal Component Analysis on testing and training data
%   INPUT:
%   train_data = training data (features in columns, samples in rows)
%   test_data = test data (features in columns, samples in rows)
%   train_labels = training labels (column vector)
%   test_labels = test labels (column vector) 
%   K = number of dimensions to reduce the data
%   OUTPUT:
%   E = misclassification error 
%   SCORES = transformed training data
%   SCORES1 = transformed testing data

% Size of training data
[n,m] = size(train_data);
[n1,m1] = size(test_data);
% Subtract mean from data
train = bsxfun(@minus,train_data,mean(train_data,1));
test = bsxfun(@minus,test_data,mean(train_data,1));
% Perform SVD on data
[U,S,V] = svd(train,'econ');
%Eigen Vectors
COEFF = V;
%Sign convention for Eigen Vectors from training data
[~,maxind] = max(abs(COEFF),[],1); %indices of max value in each eigen vector
d = size(COEFF,2);
colsign = sign(COEFF(maxind+(0:m:(d-1)*m)));%get sign of each max value
COEFF = bsxfun(@times,COEFF,colsign);%ensure all max elements in each eigen vector are positive
%Eigen Values
LATENT = diag(S);
%Projection of training data on principal components
SCORES = bsxfun(@times,U,LATENT');
% Correct sign of each score
SCORES = bsxfun(@times,SCORES,colsign);
avg_train = repmat(mean(train_data,1),[length(test_data(:,1)),1]);
%Projection of testing data on principal components
SCORES1 = avg_train + test*COEFF;
%Eigen Values 
LATENT = LATENT./sqrt(n-1);
LATENT = LATENT.^2;
%Select Projected data on first K components
Train_new = SCORES(:,1:K); %Projections of data on Principal Axis
Test_new = SCORES1(:,1:K); %Projections of data on Principal Axis
%% If Dimensionality is reduced to 2
if(K==2)
   %Plot data after projection on to first two principal components 
   for t=1:1:10
   index = find(train_labels==t-1);
   scatter(Train_new(index,1),Train_new(index,2),'*');
   hold on
   avg = mean(Train_new(index,:),1);
   plot(avg(1),avg(2),'co','MarkerSize',8,'MarkerEdgeColor','c',...
       'Color','m','LineWidth',2,'MarkerFaceColor','m');
   hold on
   title('Data Dimensionality Reduction to 2D')
   xlabel('X-axis' ) % x-axis label
   ylabel('Y-axis') % y-axis label
   end
   
  %Visualize eigen vectors
   row = (length(COEFF(:,1)))^0.5;
   j=1;
   for i = 1:1:row
       eigen_1(i,:) = COEFF(j:j+row-1,1);
       eigen_2(i,:) = COEFF(j:j+row-1,2);
       j = j+row-1;
   end
   figure
   imshow(eigen_1);
   title('Visualization of First Eigen Vector')
  
   figure
   imshow(eigen_2);
   title('Visualization of Second Eigen Vector')
   
   E = 0;
   
elseif(K==3)
   for t=1:1:10
   index = find(train_labels==t-1);
   scatter3(Train_new(index,1),Train_new(index,2),Train_new(index,3),'*');
   hold on
   avg = mean(Train_new(index,:),1);
   title('Data Dimensionality Reduction to 2D')
   xlabel('X-axis' ) % x-axis label
   ylabel('Y-axis') % y-axis label
   zlabel('Z-axis') % y-axis label
   end
   
  %Visualize eigen vectors
   row = (length(COEFF(:,1)))^0.5;
   j=1;
   for i = 1:1:row
       eigen_3(i,:) = COEFF(j:j+row-1,3);
       j = j+row-1;
   end
   figure
   imshow(eigen_3);
   title('Visualization of Third Eigen Vector')

   E = 0;
end 
    Class = KNN_classifier(SCORES1(:,1:K),SCORES(:,1:K),train_labels,1);
    error = (Class'-test_labels);
    ind = find(error~=0);
    error(ind) = 1;
    E = sum(error)/length(error);
    CP = confusionmat(test_labels,single(Class));
    for l=1:1:10
    TP(l) = CP(l,l)/sum(CP(:,l));
    end

%% Determine value of d which satisfies 0.95 threshold criteria 
Total_Sum = sum(LATENT);
for i=1:1:length(LATENT)
    Part_Sum = sum(LATENT(1:i));
    Crit = Part_Sum/Total_Sum;
    if(Crit>0.95)
        disp(i);
        break;
    end
end


end