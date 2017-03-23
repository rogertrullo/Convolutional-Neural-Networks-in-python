import numpy as np
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
    # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    scores=np.exp(scores)/np.sum(np.exp(scores))
    for j in xrange(num_classes):
        if j==y[i]:
            dW[:,j]+=X[i]*(scores[j]-1)            
        else:
            dW[:,j]+=X[i]*(scores[j])
        
    loss+=-1*np.log(scores[y[i]])
    

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /=num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW+=reg*W
  
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  scores -= np.reshape(np.max(scores,axis=1),(X.shape[0],-1))
  scores=np.exp(scores)/(np.reshape(np.sum(np.exp(scores),axis=1),(X.shape[0],-1)))
  correct_class_score = scores[np.arange(scores.shape[0]),y]
  l_vec=-1*np.log(correct_class_score)  
  loss=np.mean(l_vec)  
  r_i=correct_class_score-1
  R=scores 
  R[np.arange(R.shape[0]),y]=r_i
  dW=X.T.dot(R)
  dW /=X.shape[0]
  dW+=reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

