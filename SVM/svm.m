function [predict_label, accuracy] = svm(train_data,test_data,train_labels,test_labels)
%SVM Support Vector Machine on testing and training data
%   INPUT:
%   train_data = training data (features in columns, samples in rows)
%   test_data = test data (features in columns, samples in rows)
%   train_labels = training labels (column vector)
%   test_labels = test labels (column vector) 
%   OUTPUT:
%   predict_label = predicted label on testing data 
%   accuracy = misclassification error 
%   SCORES = transformed training data
%   SCORES1 = transformed testing data
[n,m] = size(train_data);
[nt,mt] = size(test_data);
%% convert labels to binary
train_labels = train_labels + 1;
test_labels = test_labels + 1;
%% scale data
 train_scaled = train_data/255;
 test_scaled = test_data/255;
%% train svm where svmtrain is part of LibSVM
model = svmtrain(train_labels, train_scaled, '-s 0 -t 2 -c 10 -g 0.1 -h 0');
[predict_label, accuracy, dec_values] = svmpredict(test_labels, test_scaled, model);
end