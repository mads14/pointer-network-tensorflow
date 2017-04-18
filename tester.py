from __future__ import division
import os
import numpy as np
from tqdm import trange
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope

from model import Model
from utils import show_all_variables
# from data_loader import S2LDataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt

class Tester(object):
  def __init__(self, config):
    np.set_printoptions(formatter={'float':lambda x: '%.3f'%x})
    self.config = config

    self.task = config.task
    self.model_dir = config.model_dir
    self.gpu_memory_fraction = config.gpu_memory_fraction

    self.log_step = config.log_step
    self.max_step = config.max_step
    self.num_log_samples = config.num_log_samples
    self.checkpoint_secs = config.checkpoint_secs

    # # MS added
    # self.print_logits = True

    # if config.task.lower().startswith('s2l'):
    #   #UPDATE for spark
    #   self.data_loader = S2LDataLoader(config, rng=self.rng)
    # else:
    #   raise Exception("[!] Unknown task: {}".format(config.task))

    self.model = Model(config)

    self.build_session()
    show_all_variables()

  def build_session(self):
    self.saver = tf.train.Saver()
    self.summary_writer = tf.summary.FileWriter(self.model_dir)

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


  def predict(self, data):
    tf.logging.info("Predicting...")

    
    _predictions= self._predict(None, data)
    return _predictions

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

  def _predict(self, summary_writer, data):
    fetch = {'logits': self.model.y}

    # format data
    # pad to make len(data)%128==0
    n_samples = len(data)

    data = list(data)
    data.extend(np.zeros((128-len(data)%128,1,3)))
    #TODO this should come from config (as well as batch size and shape above)
    max_seq_length = 135
    # if more than 128 samples break into multiple batches:
    seq_lengths = []
    for i in range(len(data)):
      seq = data[i]
      seq_lengths.append(len(seq))
      if len(seq) < max_seq_length:
        n = max_seq_length-len(seq)
        seq = np.vstack((seq, np.zeros((n,seq.shape[1]))))
      else:
        seq = seq[0:max_seq_length]
      data[i] = seq
    data = np.array(data)
    seq_lengths = np.array(seq_lengths)

    predictions = np.array([])
    for i in range(int(len(data)/128)):
      data_batch = data[i*128:(i+1)*128]
      batch_seq_lengths = seq_lengths[i*128:(i+1)*128]

      input_feed = {self.model.enc_inputs.name: data_batch,
                    self.model.enc_seq_length.name: batch_seq_lengths,
                    self.model.dec_seq_length.name: np.ones(128)*7,
                    self.model.dec_targets.name: np.zeros((128,7)),
                    self.model.is_training.name: False
      }

      result = self.model.predict(self.sess, fetch, 
                                  feed_dict=input_feed, 
                                  summary_writer=summary_writer)
      print result['logits']
      predictions = np.append(predictions, np.argmax(result['logits'],1)).astype(int)
    
    return(predictions[0:n_samples])

    # if plot:
    #             #car,  walk, train,   bus,    subwas,  tram,   cable car
    #   colors = ['blue','red','green','black','orange','purple','grey']
    #   for i in range(len(predictions)):
    #     x = result["inputs"][i][0:result["input_len"][i]]
    #     # color = colors[6-predictions[i]]
    #     if labels[i] != predictions[i]:
    #       plt.plot(x[:,1], x[:,0], color=colors[6-predictions[i]], linewidth=2.0)
    #       plt.plot(x[:,1], x[:,0], 'o', color=colors[6-labels[i]])
    #     else:
    #       plt.plot(x[:,1], x[:,0], color=colors[6-predictions[i]])
    #     # print x, labels[i], color, predictions[i]
    #   plt.show()
    # return predictions, labels

  def _get_summary_writer(self, result):
    if result['step'] % self.log_step == 0:
      return self.summary_writer
    else:
      return None