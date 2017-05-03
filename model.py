import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers.python.layers import utils

from layers import *

class Model(object):
  def __init__(self, config, 
               inputs=None, labels=None, enc_seq_length=None, dec_seq_length=None,
               reuse=False, is_critic=False):
    self.task = config.task
    self.debug = config.debug
    self.config = config

    self.input_dim = config.input_dim
    self.hidden_dim = config.hidden_dim
    self.num_layers = config.num_layers

    #MS added
    self.num_classes = config.output_dim

    self.max_enc_length = config.max_enc_length

    self.init_min_val = config.init_min_val
    self.init_max_val = config.init_max_val
    self.initializer = \
        tf.random_uniform_initializer(self.init_min_val, self.init_max_val)


    self.lr_start = config.lr_start
    self.lr_decay_step = config.lr_decay_step
    self.lr_decay_rate = config.lr_decay_rate
    self.max_grad_norm = config.max_grad_norm

    self.layer_dict = {}

    ##############
    # inputs
    ##############

    self.is_training = tf.placeholder_with_default(
        tf.constant(False, dtype=tf.bool),
        shape=(), name='is_training'
    )

    if inputs != None:
      self.enc_inputs, self.dec_targets, self.enc_seq_length, self.dec_seq_length = \
          utils.smart_cond(
              self.is_training,
              lambda: (inputs['train'], labels['train'], enc_seq_length['train'],
                       dec_seq_length['train']),
              lambda: (inputs['test'], labels['test'], enc_seq_length['test'],
                       dec_seq_length['test'])
          )

      self.enc_seq_length = tf.reshape(self.enc_seq_length,[-1])

    else:
      self.enc_inputs = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_enc_length, self.input_dim],
                                       name='input') #<-batch size, 135? max seq length
      self.enc_seq_length = tf.placeholder(tf.int32, shape=[self.batch_size], name='seq_len')
      self.dec_seq_length = tf.placeholder(tf.int32, shape=[self.batch_size], name='dec_len')
      self.dec_targets = tf.placeholder(tf.int32, shape=[self.batch_size,self.num_classes], name='dec_targets')
    
    
    self._build_model()
    self._build_steps()

  

    self.train_summary = tf.summary.merge([
        tf.summary.scalar("train/total_loss", self.total_loss),
        tf.summary.scalar("train/lr", self.lr),
    ])

    self.test_summary = tf.summary.merge([
        tf.summary.scalar("test/total_loss", self.total_loss),
    ])

  def _build_steps(self):
    def run(sess, fetch, feed_dict, summary_writer, summary):
      fetch['step'] = self.global_step
      if summary is not None:
        fetch['summary'] = summary

      result = sess.run(fetch, feed_dict)
      if summary_writer is not None:
        summary_writer.add_summary(result['summary'], result['step'])
        summary_writer.flush()
      return result

    def train(sess, fetch, summary_writer):
      return run(sess, fetch, feed_dict={},
                 summary_writer=summary_writer, summary=self.train_summary)

    def test(sess, fetch, summary_writer=None):
      return run(sess, fetch, feed_dict={self.is_training: False},
                 summary_writer=summary_writer, summary=self.test_summary)

    def predict(sess, fetch, feed_dict, summary_writer = None):
      return run(sess, fetch, feed_dict=feed_dict,
                 summary_writer=summary_writer, summary=self.test_summary)

    self.train = train
    self.test = test
    self.predict = predict

  def _build_model(self):
    tf.logging.info("Create a model..")
    self.global_step = tf.Variable(0, trainable=False)

    input_embed = tf.get_variable(
        "input_embed", [1, self.input_dim, self.hidden_dim],
        initializer=self.initializer)

    with tf.variable_scope("encoder"):
      self.embeded_enc_inputs = tf.nn.conv1d(
          self.enc_inputs, input_embed, 1, "VALID")

    batch_size = tf.shape(self.enc_inputs)[0]
    with tf.variable_scope("encoder"):
      self.enc_cell = LSTMCell(
          self.hidden_dim,
          initializer=self.initializer)


      self.enc_cell = rnn.MultiRNNCell([self.enc_cell] * self.num_layers, state_is_tuple=True)
      self.enc_init_state = trainable_initial_state(
          batch_size, self.enc_cell.state_size)

      self.enc_outputs, self.enc_final_states = tf.nn.dynamic_rnn(
          self.enc_cell, self.embeded_enc_inputs,
          self.enc_seq_length, self.enc_init_state)

    with tf.variable_scope("output_projection"):
      W = tf.get_variable(
          "W",
          [self.hidden_dim, self.num_classes],
          initializer=tf.truncated_normal_initializer(stddev=0.1))
      b = tf.get_variable(
          "b",
          [self.num_classes],
          initializer=tf.constant_initializer(0.1))
      #we use the cell memory state for information on sentence embedding
      self.scores = tf.nn.xw_plus_b(self.enc_final_states[-1][0], W, b)
      self.y = tf.nn.softmax(self.scores)
      self.predictions = tf.argmax(self.scores, 1)

    with tf.variable_scope("loss"):
      #TODO, should not be hard coded:
      self.class_weight = tf.constant([1, 1-.021, 1-.059, 1-.156, 1-.01, 1.0, 1-.754])
      self.weighted_logits = tf.multiply(self.scores, self.class_weight)
      self.losses = tf.nn.weighted_cross_entropy_with_logits(tf.cast(self.dec_targets, tf.float32), 
          self.weighted_logits, self.class_weight, name=None)

      # self.losses = tf.nn.softmax_cross_entropy_with_logits(
      #     logits=self.scores, labels=self.dec_targets, name="ce_losses")
      self.total_loss = tf.reduce_sum(self.losses)
      self.mean_loss = tf.reduce_mean(self.losses)

    with tf.variable_scope("accuracy"):
      self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.dec_targets, 1))
      self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

    params = tf.trainable_variables()
    
    self.lr = tf.train.exponential_decay(
    self.lr_start, self.global_step, self.lr_decay_step,
    self.lr_decay_rate, staircase=True, name="learning_rate")

    # if self.is_training:
    with tf.name_scope("train") as scope:
      opt = tf.train.AdamOptimizer(self.lr)
    gradients = tf.gradients(self.losses, params)
    clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_grad_norm)
    with tf.name_scope("grad_norms") as scope:
      grad_summ = tf.summary.scalar("grad_norms", norm)
    self.optim = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
    loss_summ = tf.summary.scalar("{0}_loss".format("train"), self.mean_loss)
    acc_summ = tf.summary.scalar("{0}_accuracy".format("train"), self.accuracy)
    self.merged = tf.summary.merge([loss_summ, acc_summ])
    self.saver = tf.train.Saver(tf.global_variables())

