from __future__ import division
import os
import numpy as np
from tqdm import trange
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope

from model import Model
from utils import show_all_variables
from data_loader import S2LDataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

class Trainer(object):
  def __init__(self, config, rng):
    np.set_printoptions(formatter={'float':lambda x: '%.3f'%x})
    self.config = config
    self.rng = rng

    self.task = config.task
    self.model_dir = config.model_dir
    self.gpu_memory_fraction = config.gpu_memory_fraction

    self.log_step = config.log_step
    self.max_step = config.max_step
    self.num_log_samples = config.num_log_samples
    self.checkpoint_secs = config.checkpoint_secs

    # MS added
    self.print_logits = True

    if config.task.lower().startswith('s2l'):
      self.data_loader = S2LDataLoader(config, rng=self.rng)
    else:
      raise Exception("[!] Unknown task: {}".format(config.task))

    self.model = Model(
        config,
        inputs=self.data_loader.x,
        labels=self.data_loader.y,
        enc_seq_length=self.data_loader.seq_length,
        dec_seq_length=self.data_loader.seq_length)

    self.build_session()
    show_all_variables()

  def build_session(self):
    self.saver = tf.train.Saver()
    print self.model_dir
    self.summary_writer = tf.summary.FileWriter(self.model_dir)
    self.train_writer = tf.summary.FileWriter(self.model_dir+'/train')
    self.test_writer = tf.summary.FileWriter(self.model_dir+'/test')

    sv = tf.train.Supervisor(logdir=self.model_dir,
                             is_chief=True,
                             saver=self.saver,
                             summary_op=None,
                             summary_writer=self.summary_writer,
                             save_summaries_secs=300,
                             save_model_secs=self.checkpoint_secs,
                             global_step=self.model.global_step)

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=self.gpu_memory_fraction,
        allow_growth=True) # seems to be not working
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 gpu_options=gpu_options)

    self.sess = sv.prepare_or_wait_for_session(config=sess_config)

  def train(self):
    tf.logging.info("Training starts...")
    self.data_loader.run_input_queue(self.sess)

    summary_writer = None
    for k in trange(self.max_step, desc="train"):
      fetch = {
          'optim': self.model.optim,
      }
      result = self.model.train(self.sess, fetch, summary_writer)

      if result['step'] % self.log_step == 0:
        self._test(self.test_writer)
          # self.summary_writer)

      summary_writer = self._get_summary_writer(result)

    self.data_loader.stop_input_queue()

  def test(self):
    tf.logging.info("Testing starts...")
    self.data_loader.run_input_queue(self.sess)

    test_predictions = []
    test_labels = []
    for idx in range(10):
      _predictions, _labels = self._test(None, True)
      test_predictions.extend(_predictions)
      test_labels.extend(_labels)

    print confusion_matrix(y_true = test_labels, y_pred = test_predictions)




    self.data_loader.stop_input_queue()

  def calc_accuracy(self, predictions, labels):
    correct = predictions == labels
    correct_byclass = [np.count_nonzero(correct & (labels == i)) for 
      i in range(self.model.num_classes)]

    precision = [str(correct_byclass[i])+'/'+str(np.count_nonzero(predictions==i)) for 
      i in range(self.model.num_classes)]
      #number correct by class / total predicted in class 

    recall = [str(correct_byclass[i])+'/'+str(np.count_nonzero(labels==i)) for 
      i in range(self.model.num_classes)]
      # number correct by class / total (true) class membership

    accuracy = float(np.count_nonzero(correct))/len(labels)
    print np.count_nonzero(correct)
    print 'precision per class', precision
    print 'recall per class', recall

    return accuracy
    
     


  def _test(self, summary_writer, plot = False):
    if plot:
      fetch = {
          'loss': self.model.total_loss,
          'logits': self.model.y,
          'inputs': self.model.enc_inputs,
          'input_len': self.model.enc_seq_length,
          'true': self.model.dec_targets,
      }
    else:
      fetch = {
          'loss': self.model.total_loss,
          'logits': self.model.y,
          'true': self.model.dec_targets,
      }
    result = self.model.test(self.sess, fetch, summary_writer)
    predictions = np.argmax(result['logits'],1)
    labels = np.argmax(result['true'],1)
    accuracy = self.calc_accuracy(predictions,labels)

    tf.logging.info("")
    tf.logging.info("test loss: {}".format(result['loss']))
    tf.logging.info("test accuracy: {}".format(accuracy))


    for idx in range(self.num_log_samples):
      logits, true = result['logits'][idx], result['true'][idx]
      pred = np.array([int(i) for i in np.binary_repr(2**np.argmax(logits)).zfill(7)[::-1]])

      if self.print_logits:
        tf.logging.info("test logits: {}".format(logits))
      
      else:
        tf.logging.info("test logits: {}".format(pred))
      tf.logging.info("test true: {} ({})".format(true, np.array_equal(pred, true)))

    if summary_writer:
      summary_writer.add_summary(result['summary'], result['step'])
              #car,  walk, train,   bus,    subwas,  tram,   cable car
    colors = ['blue','red','green','black','orange','purple','grey']
    if plot:
      for i in range(len(predictions)):
        x = result["inputs"][i][0:result["input_len"][i]]
        # color = colors[6-predictions[i]]
        if labels[i] != predictions[i]:
          plt.plot(x[:,1], x[:,0], color=colors[6-predictions[i]], linewidth=2.0)
          plt.plot(x[:,1], x[:,0], 'o', color=colors[6-labels[i]])
        else:
          plt.plot(x[:,1], x[:,0], color=colors[6-predictions[i]])
        # print x, labels[i], color, predictions[i]
      plt.show()
    return predictions, labels

  def _get_summary_writer(self, result):
    if result['step'] % self.log_step == 0:
      return self.train_writer
      #self.summary_writer
    else:
      return None
