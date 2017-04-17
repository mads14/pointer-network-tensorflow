from __future__ import division
from datetime import datetime, date
from ast import literal_eval
import numpy as np
import os
import json
import pytz
tz = pytz.timezone('US/Pacific')


#lstm imports
import tensorflow as tf
import sys
# sys.path.insert(0, '/Users/madeleinesheehan/GitHub/pointer-network-tensorflow')
# sys.path.insert(0, '/home/msheehan/maddie_att/s2l_lstm')
from tester import Tester
from config import get_config
from utils import prepare_dirs_and_logger, save_config
    


# should filter trips to get trips of interest prior to running this.

def normalize(x, min_lat=36.5, max_lat=39, min_lon=-123.5, max_lon=-121, max_dur=25000):
     #normalize
    x[:,0]=(x[:,0]-min_lat)/(max_lat-min_lat)
    x[:,1]=(x[:,1]-(min_lon))/((max_lon)-(min_lon))
    x[:,2]=(x[:,2]-x[:,2][0])/max_dur
    return x

def rawSeqToLSTM(raw_sequence):

    raw_sequence = np.array(raw_sequence, dtype=float)
    seq_in = raw_sequence[:,np.array([1,2,0])]
    seq_in = normalize(seq_in)

    return seq_in 


def genLSTMin(trips):
    # filter down to raw sequence
    lstm_user_trips = []
    for date in trips:
        for trip_id in trips[date]:
            trip = trips[date][trip_id]

            print trip['great_circle_dist'], trip['origin_latlon'], trip['dest_latlon'], trip['origin_taz'], trip['dest_taz'], trip['duration']
            raw_sequence = trip['raw_sequence']
            lstm_one_trip = rawSeqToLSTM(raw_sequence)
            lstm_user_trips.append(lstm_one_trip)

    return lstm_user_trips

def main(_):
    prepare_dirs_and_logger(config)
    tester = Tester(config)
    # after getting trips of interest use tester.predict(data)

    directory = '/home/mogeng/attResearch/sample/CLF/MultipleCities/SF/single_month/20150601-20150628/trip_data/commuter/all/'
    for file in os.listdir(directory)[0:100]:
        if file.endswith(".json"):
            filename = os.path.join(directory, file)
            with open(filename) as data_file:    
                data = json.load(data_file)

                trips = data['trips']

                trips =  genLSTMin(trips)
                # print trips
                print tester.predict(trips)

if __name__ == "__main__":
    
    directory = '/home/mogeng/attResearch/sample/CLF/MultipleCities/SF/single_month/20150601-20150628/trip_data/commuter/all/'
    

    # tensorflow LSTM 
    config, unparsed = get_config()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    


    print datetime.now()
    # trip_profile = sc.textFile(spark_dir).map(literal_eval)
    

    # lstm_in = trip_profile.map(lambda (k,v): (k, genLSTMin(v)))
    # predictions = lstm_in.map(lambda(k,v): (k,tester.predict(v)))
    
    # lstm_in.saveAsTextFile("hdfs:/user/msheehan/CLF/KNN/June/lstm_in/test")
