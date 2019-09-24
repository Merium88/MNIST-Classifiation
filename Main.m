%% Main file for running codes used in project CA2 Pattern Recognition 
clear all
clc
%% Load training and tetsing data : 'train-images.idx3-ubyte', 'train-labels.idx1-ubyte'
[trainingdata, traingnd] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
trainingdata = double(reshape(trainingdata, size(trainingdata,1)*size(trainingdata,2), []).');
traingnd = single(traingnd);

[testdata, testgnd] = mnist_parse('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');
testdata = double(reshape(testdata, size(testdata,1)*size(testdata,2), []).');
testgnd = single(testgnd);

%% PCA Implementation 
% Input: training data, test data, training labels, test labels, K-dimensions
% Output: Mis-classification Error, Reduced training data, Reduced tetsing data 

[Error, Train_new,Test_new] = PCA(trainingdata,testdata,traingnd,testgnd,2);%154

%% LDA Implementation
% Input: training data, test data, 
% Output:  Transformed and reduced training data, W vector 

[Y, W] = LDA(trainingdata,traingnd);
% Reduce testing data dimensionality
Y_test = testdata*W;

K=9;
%Plot data after projection on to first two principal components 
if(K==2)
   for t=1:1:10
   index = find(traingnd==t-1);
   plot(Y(index,1),Y(index,2),'*');
   hold on
   avg = mean(Y(index,1:K),1);
   plot(avg(1),avg(2),'co','MarkerSize',8,'MarkerEdgeColor','c',...
       'Color','m','LineWidth',2,'MarkerFaceColor','m');
   hold on
   title('Data Dimensionality Reduction to 2D')
   xlabel('X-axis' ) % x-axis label
   ylabel('Y-axis') % y-axis label
   end
   
   %Visualize eigen vectors
   row = (length(W(:,1)))^0.5;
   j=1;
   for i = 1:1:row
       eigen_1(i,:) = W(j:j+row-1,1);
       eigen_2(i,:) = W(j:j+row-1,2);
       j = j+row-1;
   end
   figure
   imshow(eigen_1);
   title('Visualization of First Eigen Vector')
  
   figure
   imshow(eigen_2);
   title('Visualization of Second Eigen Vector')
elseif(K==3)
for t=1:1:10
   index = find(traingnd==t-1);
   scatter3(Y(index,1),Y(index,2),Y(index,3),'*');
   hold on
   avg = mean(Y(index,1:K),1);
   plot3(avg(1),avg(2),avg(3),'co','MarkerSize',8,'MarkerEdgeColor','c',...
       'Color','m','LineWidth',2,'MarkerFaceColor','m');
   hold on 
   title('Data Dimensionality Reduction to 3D')
   xlabel('X-axis' ) % x-axis label
   ylabel('Y-axis') % y-axis label
   zlabel('Z-axis') % z-axis label
end
 
  %Visualize eigen vectors
   row = (length(W(:,1)))^0.5;
   j=1;
   for i = 1:1:row
       eigen_3(i,:) = W(j:j+row-1,3);
       j = j+row-1;
   end
   figure
   imshow(eigen_3);
   title('Visualization of Third Eigen Vector')
end
% Perform Classification
Class = KNN_classifier(Y_test(:,1:K),Y(:,1:K),traingnd,1);
error = (Class'-testgnd);
ind = find(error~=0);
error(ind) = 1;
E = sum(error)/length(error);
CP = confusionmat(testgnd,single(Class'));
for l=1:1:10
TP(l) = CP(l,l)/sum(CP(:,l));
end

%% SVM Implementation
[predict_label, accuracy] = svm(trainingdata,testdata,traingnd,testgnd);

%% CNN Implementation
% loading the dataset MNIST as an IMDB structure
imdb_mnist = load('data/imdb_mnist.mat');

% we start training! partial results will go to results/lenet_experiment_1 folder
[net_lenet, info] = cnn_train(imdb_mnist, 'results/lenet_experiment_1');

% load a previously trained (LeNet) net
lenet1 = load('Nets/lenet.mat');

% make it a Dag object
lenet1 = dagnn.DagNN.loadobj(lenet1.net);
% again, keep the meta info about the classes (useful for inference)
lenet1.meta.classes.description = imdb_mnist.meta.classes;

% first we select validation images (those images belonging to set == 3)
val_images = find(imdb_mnist.images.set == 3);
% Let's pick a random image from validation set
for i=1:1:10000
[class(i), score] = inference(imdb_mnist.images.data(:,:,:,val_images(i)), lenet1, false);
end
labels = imdb_mnist.images.labels(val_images);
%Estimate error for the predicted labels
error = ((class)-labels);
ind = find(error~=0);
error(ind) = 1;
E = sum(error)/length(error);
