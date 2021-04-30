import math
import tensorflow as tf
import tensorflow.python.user_ops as ntm_rotate_module
import ntm
ntm_rotate_module = tf.load_op_library(
  "tensorflow/bazel-bin/tensorflow/core/user_ops/rotate.so",
)


def conv(v, k):
  """Computes circular convolution.
  Args:
      v: a 1-D `Tensor` (vector)
      k: a 1-D `Tensor` (kernel)
  """
  size = int(v.get_shape()[0])
  kernel_size = int(k.get_shape()[0])
  kernel_shift = int(math.floor(kernel_size/2.0))

  def loop(idx):
      if idx < 0: return size + idx
      if idx >= size : return idx - size
      else: return idx

  kernels = []
  for i in xrange(size):
      indices = [loop(i+j) for j in xrange(kernel_shift, -kernel_shift-1, -1)]
      v_ = tf.gather(v, indices)
      kernels.append(tf.reduce_sum(v_ * k, 0))

  return tf.pack(kernels)

def rotate(v, k):
  nbatches = int(v.get_shape()[0])
  res = []
  for i in xrange(nbatches):
    res.append(conv(
      tf.squeeze(tf.slice(v, [i, 0], [1, -1])),
      tf.squeeze(tf.slice(k, [i, 0], [1, -1])),
    ))
  return tf.pack(res)

weights = tf.constant([
  [1., 2., 3., 4., 5., 6., 7.],
  [8., 9., 10., 11., 12., 13., 14.],
])
shifts = tf.constant([
  [0., 0., 1.], 
  [1., 0., 0.],
])
rotated = ntm_rotate_module.ntm_rotate(weights, shifts)

with tf.Session() as sess:
  print sess.run(rotate(weights, shifts))
  print sess.run(rotated)

  err = tf.test.compute_gradient_error(
    [weights, shifts],
    [[2, 7], [2, 3]],
    rotated,
    [2, 7],
  )

  print "Gradient error: ", err
