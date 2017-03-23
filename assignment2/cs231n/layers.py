import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  dimx=x.shape
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  x=np.reshape(x,(dimx[0],np.prod(dimx[1:])))
  out=x.dot(w)+b 
  x=np.reshape(x,dimx)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dimx=x.shape
  x=np.reshape(x,(dimx[0],np.prod(dimx[1:])))
  dx, dw, db = None, None, None
  N=dimx[0]
  
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  db=np.sum(dout,axis=0)
  #db/=N
  dx=dout.dot(w.T)
  dx=np.reshape(dx,dimx)
  dw=x.T.dot(dout)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out=np.copy(x)
  out[out<0]=0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  tmp=np.zeros_like(x)
  tmp[x>0]=1
  dx=tmp*dout
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    sample_mean=np.mean(x,axis=0)
    sample_var=np.var(x,axis=0)
    
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    xhat=(x-sample_mean)/(np.sqrt(sample_var+eps))
    out=xhat*gamma+beta
    cache=(x,xhat,sample_mean,sample_var,gamma,beta,eps)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    xhat=(x-running_mean)/(np.sqrt(running_var+eps))
    out=xhat*gamma+beta
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  x,xhat,sample_mean,sample_var,gamma,beta,eps=cache
  m=x.shape[0]
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  dldxhat=dout*gamma

  dldvar=np.sum(dldxhat*(x-sample_mean)*-0.5*((sample_var+eps)**-1.5),axis=0)
    
  dldmean=np.sum(dldxhat*-1*((sample_var+eps)**-0.5),axis=0)+dldvar*(-2.0/m)*np.sum((x-sample_mean),axis=0)
  
  dldx=dldxhat*((sample_var+eps)**-0.5)+dldvar*(2.0/m)*(x-sample_mean)+(dldmean/m)

  dldgamma=np.sum(dout*xhat,axis=0)
  dldbeta=np.sum(dout,axis=0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  dx=dldx
  dgamma=dldgamma
  dbeta=dldbeta
  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    mask = (np.random.rand(*x.shape) < p) / p # first dropout mask. Notice /p!
    out = x*mask # drop!
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out=x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    dx=dout*mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """

  cache=None
  N, C, rows, cols=x.shape
  F, _, HH, WW=w.shape
  stride=conv_param['stride']
  pad=conv_param['pad']
  H_ = 1 + (rows + 2 * pad - HH) / stride
  W_ = 1 + (cols + 2 * pad - WW) / stride 
  out = np.zeros((N, F, H_, W_))
  szpad=(rows+2*pad,cols+2*pad)
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  npad = ((0,0), (pad,pad), (pad,pad))
  #############################################################################
  """ for i in xrange(N):
        for j in xrange(F):
            cntr=0
            cntc=0
            for dr in xrange(0,szpad[0]-HH+1,stride):
                for dc in xrange(0,szpad[1]-WW+1,stride):
                    
                    img=x[i]                    
                    img=np.pad(img, pad_width=npad, mode='constant', constant_values=0)
                    img=img[:,dr:dr+HH,dc:dc+WW]
                    out[i,j,cntr,cntc]=np.sum(img*w[j])+b[j]
                    cntc+=1
                cntc=0
                cntr+=1
  """
  
  for i in xrange(N):
        for j in xrange(F):
              for dr in xrange(H_):
                for dc in xrange(W_):
                    
                    img=x[i]                    
                    img=np.pad(img, pad_width=npad, mode='constant', constant_values=0)
                    ini_r=stride*dr
                    ini_c=stride*dc
                    fin_r=ini_r+HH
                    fin_c=ini_c+WW
                    img=img[:,ini_r:fin_r,ini_c:fin_c]
                    out[i,j,dr,dc]=np.sum(img*w[j])+b[j]
                       
                    
                
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  (x, w, b, conv_param)=cache
  N, C, rows, cols=x.shape
  F, _, HH, WW=w.shape
  stride=conv_param['stride']
  pad=conv_param['pad']
  H_ = 1 + (rows + 2 * pad - HH) / stride
  W_ = 1 + (cols + 2 * pad - WW) / stride
  #out = np.zeros((N, F, H_, W_))
  wfliped=np.zeros_like(w)#inverted filter
  dx=np.zeros_like(x)
  for i in range(F):
    for j in range (C):
         wfliped[i,j]=np.rot90(w[i,j],2)
            
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################


    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    for i in xrange(N):      # for every example
      input = x[i, :, :, :]
      input = np.pad(input, ((0,0), (1,1), (1,1)), 'constant', constant_values=0)
      dinput = np.zeros_like(input)

      for j in xrange(w.shape[0]):    # for every filter
        filter = w[j, :, :, :]

        for hp in xrange(H_):     # slide height
          for wp in xrange(W_):   # slide width
            hstart, hend = hp * stride, hp * stride + HH
            wstart, wend = wp * stride, wp * stride + WW

            db[j] +=dout[i,j,hp,wp] ### funny part ###

            subinput = input[:, hstart:hend, wstart:wend]
            dw[j, :, :, :] +=dout[i,j,hp,wp]*subinput ### funny part ###

            dinput[:, hstart:hend, wstart:wend] +=dout[i,j,hp,wp]*filter ### funny part ###
      dx[i, :, :, :] =dinput[:,pad:rows+pad,pad:cols+pad] ### funny part ###

    
  """for i in range(F):
  #  tmp=dout[:,i]
    a=np.zeros_like(x)    
    for j in range (C):
        a[:,j]=tmp
        res,_=conv_forward_naive(a,np.expand_dims(wfliped[i], axis=0),np.array([0]),conv_param)
        print res
        dx[:,j]+=np.squeeze(res)

  #dw,_=conv_forward_naive(x, dout, np.zeros_like(b), conv_param)"""
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  N, C, rows, cols=x.shape
  HH=pool_param['pool_height']
  WW=pool_param['pool_width']
  stride=pool_param['stride']
    
  H_ = 1 + (rows - HH) / stride
  W_ = 1 + (cols  - WW) / stride
  out=np.zeros((N,C,H_,W_))
  x_c=np.zeros_like(x)
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  for i in xrange(N):#img
        for j in xrange(C):#channel
            for dr in xrange(H_):
                for dc in xrange(W_):
                    img=x[i]                    
                    ini_r=stride*dr
                    ini_c=stride*dc
                    fin_r=ini_r+HH
                    fin_c=ini_c+WW

                    tmp=np.zeros((HH,WW))
                    subimg=img[j,ini_r:fin_r,ini_c:fin_c]
                    
                    (idr,idc)=np.unravel_index(np.argmax(subimg),tmp.shape)
                    out[i,j,dr,dc]=subimg[idr,idc]


                    tmp[idr,idc]=1
                    x_c[i,j,ini_r:fin_r,ini_c:fin_c]=tmp
                
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x_c, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  
  x_c, pool_param=cache
  
  dx = np.zeros_like(x_c)
  HH=pool_param['pool_height']
  WW=pool_param['pool_width']
  stride=pool_param['stride']
  N,C,H_,W_=dout.shape
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  for i in xrange(N):
    for j in xrange(C):
         for dr in xrange(H_):
            for dc in xrange(W_):
                dimg=dout[i,j]                    
                ini_r=stride*dr
                ini_c=stride*dc
                fin_r=ini_r+HH
                fin_c=ini_c+WW
                dx[i,j,ini_r:fin_r,ini_c:fin_c]+=dimg[dr,dc]*x_c[i,j,ini_r:fin_r,ini_c:fin_c]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None
  (N, C, H, W)=x.shape
  mchannels=np.zeros((N*H*W,C))
  out=np.zeros_like(x)
  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  for i in xrange(C):
    mchannels[:,i]=np.ravel(x[:,i])
  
  outtmp, cache=batchnorm_forward(mchannels, gamma, beta, bn_param)
  for i in xrange(C):
    out[:,i]=np.reshape(outtmp[:,i],(N,H,W))

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  
  (N, C, H, W)=dout.shape  
  dx=np.zeros((N, C, H, W))
  dgamma=np.zeros(C)
  dbeta=np.zeros(C)
  
  dchannels=np.zeros((N*H*W,C))

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  for i in xrange(C):
    dchannels[:,i]=np.ravel(dout[:,i])
  
  dx1, dgamma, dbeta =batchnorm_backward(dchannels,cache) 
    
  for i in xrange(C):
    dx[:,i]=np.reshape(dx1[:,i],(N,H,W))
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
