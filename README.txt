Execute the Main file 'Main.m' to execute all Questions 1-4 and plot their results

Input the MNIST data in the following form for PCA, LDA, SVM:
   t10k-images.idx3-ubyte
   train-images.idx3-ubyte
   t10k-labels.idx1-ubyte
   train-labels.idx1-ubyte

Input for CNN:
  'data/imdb_mnist.mat'
Previously trained data:
   'Nets/lenet.mat'


The following is done in the main file 'Main.m':

   a)Load data by calling mnist_parse.m
   b)Implement PCA, call PCA.m
   c)Implement LDA, call LDA.m
   d)Implement SVM, call svm.m
   e)Implement CNN, call cnn_train.m, inference.m

Function files:

   a)mnist_parse.m = Load MNIST data and convert to matrices of training, testing data and labels
   b)PCA.m :  Implements PCA, calls KNN_classifier.m for nearest neighbor classification
   c)LDA.m :  Implements LDA 
   d)svm.m :  Implements SVM using LibSVM (will not work without libsvm)
   e)cnn_train.m : Creates the architecture for CNN
   f)inference.m : Checks the testing data using CNN trained model
   g)KNN_classifier.m : Implements nearest neighbour algorithm

NOTE: 1) If exectution of 1 function is required, please comment all subsequent questions to decrease running time. 
      2) For execution of each function, go inside the respective folder
      3) Current value of K for nearest neighbor implementation is kept 1 for 
         fast implementation. Actual value used is K = 5 (for the report).