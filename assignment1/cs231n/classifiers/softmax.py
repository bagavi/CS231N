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
      loss -= np.log( exp_scores[y[i]]/exp_scores_sum )
      #Gradient update
      dW[:,y[i]] -= X[i]
      for j in xrange(num_classes):
        dW[:,j] += X[i]*exp_scores[j]/exp_scores_sum
  
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

  X1 = X.dot(W)
  scores = pow(math.e,X1)
  #normarlizing the score
  scores = (scores.T/np.sum(scores, axis = 1)).T
  
  Y = np.zeros([num_train, num_classes])
  for i,row in enumerate(Y):
    row[y[i]] = 1
  
  loss_ma = np.ma.log(scores*Y)
  loss = np.sum( - loss_ma.filled(0))/num_train + 0.5 * reg * np.sum(W * W)  
  
  dX1 = scores - Y
  dW = ((dX1.T).dot(X)).T/num_train + reg*W
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

