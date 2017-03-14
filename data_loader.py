# Most of the codes are from 
# https://github.com/vshallc/PtrNets/blob/master/pointer/misc/tsp.py
import os
import re
import zipfile
import itertools
import threading
import numpy as np
from tqdm import trange, tqdm
from collections import namedtuple

import tensorflow as tf
from download import download_file_from_google_drive


TSP = namedtuple('TSP', ['x', 'y', 'name'])

def length(x, y):
  return np.linalg.norm(np.asarray(x) - np.asarray(y))


def read_paper_dataset(paths, max_length):
  i=0
  x, y = [], []
  for path in paths:
    tf.logging.info("Read dataset {} which is used in the paper..".format(path))
    length = max(re.findall('\d+', path))
    with open(path) as f:
      for l in tqdm(f):
        i+=1
        inputs, outputs = l.split(' output ')
        x.append(np.array(inputs.split(), dtype=np.float32).reshape([-1, 3]))
        y.append(np.array(outputs.split(), dtype=np.int32))
  return x, y

#todo change name from tsp data loader
class TSPDataLoader(object):
  def __init__(self, config, rng=None):
    self.config = config
    self.rng = rng

    self.task = config.task.lower()
    self.batch_size = config.batch_size
    self.min_length = config.min_data_length
    self.max_length = config.max_data_length

    self.is_train = config.is_train
    self.use_terminal_symbol = config.use_terminal_symbol
    self.random_seed = {'train': config.random_seed, 'test':config.random_seed+372}

    self.data_num = {}
    self.data_num['train'] = config.train_num
    self.data_num['test'] = config.test_num

    self.data_dir = config.data_dir
    self.task_name = "{}_({},{})".format(
        self.task, self.min_length, self.max_length)

    self.data = None
    self.coord = None
    self.threads = None
    self.input_ops, self.target_ops = None, None
    self.queue_ops, self.enqueue_ops = None, None
    self.x, self.y, self.seq_length, self.mask = None, None, None, None

    paths = {'train':'data2/lstm/all_trips_normed.csv', 
             'test':'data2/lstm/all_trips_normed.csv'}

    if len(paths) != 0:
      #todo change to be check that train and test data exist
      self._maybe_generate_and_save(except_list=paths.keys())
      for name, path in paths.items():
        self.read_zip_and_update_data(path, name)
    else:
      raise Exception("Must specify train and test dataset paths!")
    self._create_input_queue()

  def _create_input_queue(self, queue_capacity_factor=16):
    self.input_ops, self.target_ops = {}, {}
    self.queue_ops, self.enqueue_ops = {}, {}
    self.x, self.y, self.seq_length, self.mask = {}, {}, {}, {}

    for name in self.data_num.keys():
      self.input_ops[name] = tf.placeholder(tf.float32, shape=[None, None])
      self.target_ops[name] = tf.placeholder(tf.int32, shape=[None])

      min_after_dequeue = 1000
      capacity = min_after_dequeue + 3 * self.batch_size

      self.queue_ops[name] = tf.RandomShuffleQueue(
          capacity=capacity,
          min_after_dequeue=min_after_dequeue,
          dtypes=[tf.float32, tf.int32],
          # MS
          shapes=[[self.max_length, 3,], [4]],
          # original
          # shapes=[[self.max_length, 2,], [self.max_length]],
          seed=self.random_seed[name],
          name="random_queue_{}".format(name))
      self.enqueue_ops[name] = \
          self.queue_ops[name].enqueue([self.input_ops[name], self.target_ops[name]])

      inputs, labels = self.queue_ops[name].dequeue()
      

      # original
      # seq_length = tf.shape(inputs)[0]
      # MS
      seq_length = 4

      #TODO check on mask size?
      if self.use_terminal_symbol:
        mask = tf.ones([seq_length + 1], dtype=tf.float32) # terminal symbol
      else:
        mask = tf.ones([seq_length], dtype=tf.float32)

      self.x[name], self.y[name], self.seq_length[name], self.mask[name] = \
          tf.train.batch(
              [inputs, labels, seq_length, mask],
              batch_size=self.batch_size,
              capacity=capacity,
              dynamic_pad=True,
              name="batch_and_pad")
      print 'self.seq_len', self.seq_length[name]

  def run_input_queue(self, sess):
    self.threads = []
    self.coord = tf.train.Coordinator()

    for name in self.data_num.keys():
      print name, 'x', len(self.data[name].x), 'y', len(self.data[name].y)
      def load_and_enqueue(sess, name, input_ops, target_ops, enqueue_ops, coord):
        idx = 0
        while not coord.should_stop():
          feed_dict = {
              input_ops[name]: self.data[name].x[idx],
              target_ops[name]: self.data[name].y[idx],
          }
          sess.run(self.enqueue_ops[name], feed_dict=feed_dict)
          idx = idx+1 if idx+1 <= len(self.data[name].x) - 1 else 0


      args = (sess, name, self.input_ops, self.target_ops, self.enqueue_ops, self.coord)
      t = threading.Thread(target=load_and_enqueue, args=args)
      t.start()
      self.threads.append(t)
      tf.logging.info("Thread for [{}] start".format(name))

  def stop_input_queue(self):
    self.coord.request_stop()
    self.coord.join(self.threads)
    tf.logging.info("All threads stopped")

  def _maybe_generate_and_save(self, except_list=[]):
    self.data = {}

    for name, num in self.data_num.items():
      if name in except_list:
        tf.logging.info("Skip creating {} because of given except_list {}".format(name, except_list))
        continue
      else:
        raise Exception("{} dataset not specified").format(name)

    tf.logging.info("Can't found dataset from the paper!")
    return paths

  def read_zip_and_update_data(self, path, name):
    if path.endswith('zip'):
      filenames = zipfile.ZipFile(path).namelist()
      paths = [os.path.join(self.data_dir, filename) for filename in filenames]
    else:
      paths = [path]

    x_list, y_list = read_paper_dataset(paths, self.max_length)

    x = np.zeros([len(x_list), self.max_length, 3], dtype=np.float32)
    y = np.zeros([len(y_list), 4], dtype=np.int32)

    for idx, (nodes, res) in enumerate(tqdm(zip(x_list, y_list))):
      x[idx,:min(self.max_length,len(nodes))] = nodes[:min(self.max_length,len(nodes))]
      y[idx,:len(res)] = res

    if self.data is None:
      self.data = {}

    tf.logging.info("Update [{}] data with {} used in the paper".format(name, path))
    self.data[name] = TSP(x=x, y=y, name=name)
