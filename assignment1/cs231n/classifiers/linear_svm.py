import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  Loss = np.zeros([num_train, num_classes])
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        Loss[i][j] = margin
        loss += margin
        dW[:,j] += X[i]
        dW[:,y[i]] -= X[i] 
    pass
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW   /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  
  #Adding the gradient of the regularization
  dW += reg*W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  return loss, dW, Loss


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_classes = W.shape[1]
  num_train = X.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  #Calculating the score
  Scores = X.dot(W)
  Correct_class_scores = np.array( [ [ Scores[i][y[i]] ]*num_classes for i in range(num_train) ] )
  
  #Creating the delta matrix, with zeros at the correct class
  matrix1 = np.ones(Scores.shape)
  for i in range(num_train):
      matrix1[i][y[i]] = 0
  #calculating the margin
  Margin = Scores - Correct_class_scores + matrix1
  
  Margin_ge_zero = np.maximum(Margin, np.zeros(Margin.shape))
  
  loss += np.sum(Margin_ge_zero)/num_train
  loss += 0.5 * reg * np.sum(W * W)
  
  X_with_margin_count = np.multiply(X.T , ( Margin_ge_zero !=0).sum(1) ).T
  for j in range(num_classes):
      Margin_j  = np.where( Margin_ge_zero[:,j] > 0 )
      #Margin_j = range(num_train)

      dW[:,j]  += sum( X[Margin_j] )   
      #data_of_class_j = range(num_train)
      data_of_class_j = np.where(y==j)
      dW[:,j] -= sum( X_with_margin_count[data_of_class_j] )
  dW   /= num_train
  dW += reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW, Margin_ge_zero
