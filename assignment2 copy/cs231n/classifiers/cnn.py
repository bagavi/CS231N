import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *
from cs231n.classifiers.fc_net import *

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               bn_params = {'mode': 'train'}, dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    #Conv Layer
    self.params['W1'] = np.random.normal(0,weight_scale,[num_filters, input_dim[0], filter_size, filter_size])
    self.params['b1'] = np.zeros(num_filters)
    self.params['y1'] = np.random.normal( 1 , 1e-3, num_filters)
    self.params['beta1'] = np.zeros( num_filters )
    
    #Hidden affine layer
    # Dividing by 4 because of the maxpool layer
    self.params['W2'] = np.random.normal(0,weight_scale,[num_filters*input_dim[1]*input_dim[2]/4, hidden_dim])
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['y2'] = np.random.normal( 1 , 1e-3, hidden_dim)
    self.params['beta2'] = np.zeros( hidden_dim )
    
    #Output affine layer
    self.params['W3'] = np.random.normal(0,weight_scale,[hidden_dim, num_classes])
    self.params['b3'] = np.zeros(num_classes)
    
    self.bn_params = bn_params
    self.bn_params2 = dict(bn_params)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1, y1, beta1 = self.params['W1'], self.params['b1'], self.params['y1'], self.params['beta1']
    W2, b2, y2, beta2 = self.params['W2'], self.params['b2'], self.params['y2'], self.params['beta2']
    W3, b3            = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    conv_relu_out, conv_relu_cache = conv_batchnorm_relu_forward(X, W1, b1, conv_param, y1, beta1, self.bn_params)
    maxpool_out, maxpool_cache     = max_pool_forward_fast(conv_relu_out, pool_param)
    aff_relu_out, aff_relu_cache   = affine_batchnorm_relu_forward(maxpool_out, W2, b2, y2, beta2, self.bn_params2)
    aff2_out, aff2_cache           = affine_forward(aff_relu_out, W3, b3)
    scores = aff2_out
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    sftm_loss, sftm_grad = softmax_loss(aff2_out, y)
    loss += sftm_loss + .5*self.reg*np.sum(W1*W1) + .5*self.reg*np.sum(W2*W2) + .5*self.reg*np.sum(W3*W3)
    
    
    dx_3, grads['W3'], grads['b3'] = affine_backward( sftm_grad, aff2_cache )
    dx_2, grads['W2'], grads['b2'], grads['y2'], grads['beta2'] = affine_batchnorm_relu_backward( dx_3, aff_relu_cache )
    dx_2_prime                     = max_pool_backward_fast( dx_2, maxpool_cache )
    dx_1, grads['W1'], grads['b1'], grads['y1'], grads['beta1'] = conv_batchnorm_relu_backward( dx_2_prime, conv_relu_cache )
    
    
    grads['W1'] += self.reg*W1
    grads['W2'] += self.reg*W2
    grads['W3'] += self.reg*W3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads


class FullyConnectedConvNet(object):
  """
    [conv-relu-pool]XN - [affine]XM - [softmax or SVM]  
    
    Max 2x2 pool with stride = 2
  """

  def __init__(self, pool_params, conv_params, affine_hidden_dims, input_dim=(3,32,32), num_classes=10,
               dropout=0,  reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - conv_hidden_dims: A list of integers giving the size of each convolutional layer.
    - affine_hidden_dims: A list of integers giving the size of each affine layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_dropout = dropout > 0
    self.reg = reg
    self.dtype = dtype
    self.params = {}
    self.conv_params = conv_params[ 'conv_params']
    self.num_conv_layers = len(self.conv_params)

    self.pool_params = pool_params
    self.num_affine_layers = len(affine_hidden_dims)
    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    # First layer
    conv_filter_number = conv_params[ 'filter_number']
    conv_filter_size   = conv_params[ 'filter_size']

    for layer in range( 0,self.num_conv_layers ):
      index = layer + 1 + self.num_affine_layers
      Wi = 'W'+str(index)
      bi = 'b'+str(index)
      yi = 'y'+str(index)
      betai = 'beta'+str(index)
      if layer == 0:
          self.params[Wi] = np.random.normal( 0, weight_scale, 
                            [ 
                             conv_filter_number[layer], #Number of filters in this layer
                             input_dim[0],              #Number of filters in previous layer
                             conv_filter_size[layer],   #Filter dimension
                             conv_filter_size[layer]    #Squaare filter
                            ] 
                                            )
      else:
          self.params[Wi] = np.random.normal( 0, weight_scale, 
                            [ 
                             conv_filter_number[layer], #Number of filters in this layer
                             conv_filter_number[layer - 1], #Number of filters in previous layer
                             conv_filter_size[layer],  #Filter dimension
                             conv_filter_size[layer]    #Squaare filter
                            ] 
                                            )
      
      self.params[bi] = np.zeros( conv_filter_number[layer] )
      self.params[yi] = np.random.normal( 1 , 1e-3,conv_filter_number[layer])
      self.params[betai] = np.zeros( conv_filter_number[layer] )
    
    affine_input_dim = conv_params[ 'filter_number'][-1]*input_dim[1]*input_dim[2]
    affine_input_dim /= math.pow(2, 2*len(conv_params[ 'filter_number']) )
    self.FullyConnectedNet = FullyConnectedNet( affine_hidden_dims, input_dim = affine_input_dim, 
                                 weight_scale=weight_scale, dropout=dropout, use_batchnorm=True, reg = 1e-4 )
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    self.bn_params = [{'mode': 'train'} for i in xrange( self.num_conv_layers )]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
    
  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing. 
    for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    input_data = X
    output = {}
    for layer in range(self.num_conv_layers):
      index = layer + 1 + self.num_affine_layers
      Wi = 'W'+str(index)
      bi = 'b'+str(index)
      yi = 'y'+str(index)
      betai = 'beta'+str(index)
      out, cache = conv_batchnorm_relu_pool_forward( 
                                                     input_data, self.params[Wi], self.params[bi],
                                                     self.conv_params[layer],
                                                     self.params[yi], self.params[betai], self.bn_params[layer],
                                                     self.pool_params[layer]
                                                   )
      input_data = out
      output[index] =  { 'out_data':out, 'cache': cache }
    
    #Reshaping input data
    input_data_shape = input_data.shape
    input_data = input_data.reshape(input_data_shape[0], np.prod(input_data_shape)/input_data_shape[0])
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      scores = self.FullyConnectedNet.loss( input_data, y)
      return scores
    loss, grads = self.FullyConnectedNet.loss( input_data, y)
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    #Calculating loss

    #Adding regularizatoin loss
    for layer in range(self.num_conv_layers):
      index = layer + 1 + self.num_affine_layers
      Wi = 'W'+str(index)
      loss += .5*self.reg*np.sum(self.params[Wi]*self.params[Wi])

    
    #Gradient from FNC
    dx = grads.pop('dx').reshape(input_data_shape)

    for layer in range(self.num_conv_layers)[::-1]:
      index = layer + 1 + self.num_affine_layers
      Wi = 'W'+str(index)
      bi = 'b'+str(index)
      yi = 'y'+str(index)
      betai = 'beta'+str(index)
      
      # Storing the gradients
      dx, grads[Wi], grads[bi], grads[yi], grads[betai] = conv_batchnorm_relu_pool_backward( dx, output[index]['cache'] )  
            # Loss due to the regularazitation parameter.
      grads[Wi] += self.reg*self.params[Wi]

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return loss, grads

