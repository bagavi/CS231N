import numpy as np
import math
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
#   
#   print "W shape", W.shape
#   print "X shape", X.shape
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
      
      scores = X[i].dot(W)
      exp_scores = pow(math.e, scores)
      exp_scores_sum = sum(exp_scores)
      #Loss update
      loss += np.log( exp_scores[y[i]]/exp_scores_sum )
      #Gradient update
      dW[:,y[i]] += X[i]
      for j in xrange(num_classes):
        dW[:,j] -= X[i]*exp_scores[j]/exp_scores_sum
  
  loss /= num_train
  dW /= num_train    
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
#   print y.shape
#   print X.shape
#   Label_data = np.dstack((y,X))
#   print Label_data
#   Label_data.sort()
#   y = Label_data[:,0]
#   x = Label_data[:,1:]

  #Sorting the input based on the class
  num_classes = W.shape[1]
  num_train = X.shape[0]
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  Scores = X.dot(W)
  exp_Scores = pow(math.e,Scores)
  #normarlizing the score
  exp_Scores = (exp_Scores.T/np.sum(exp_Scores, axis = 1)).T
  for i in range(num_train):
      loss += np.log( exp_Scores[i][y[i]] )
  
  for j in range(num_classes):
      dW[:,j] -= np.sum( X.T*exp_Scores[:,j] , axis = 1 )
      dW[:,j] += np.sum(X[np.where(y==j)].T, axis = 1)
      
  
  loss /= num_train
  dW /= num_train
  loss +=  0.5 * reg * np.sum(W * W)  
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

