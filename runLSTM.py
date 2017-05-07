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
    trip_ids = []

    for date in trips:
        for trip_id in trips[date]:
            trip = trips[date][trip_id]

            if (trip['great_circle_dist']>5.0) & (trip['origin_taz']!=None) & (trip['dest_taz']!=None):
              raw_sequence = trip['raw_sequence']
              if len(raw_sequence)>10: 
                # print trip['great_circle_dist'], trip['origin_latlon'], trip['dest_latlon'], trip['origin_taz'], trip['dest_taz'], trip['duration']
                lstm_one_trip = rawSeqToLSTM(raw_sequence)
                lstm_user_trips.append(lstm_one_trip)
                trip_ids.append([date,trip_id])

    return lstm_user_trips, trip_ids

def main(_):
    np.set_printoptions(precision = 2, suppress=True)
    prepare_dirs_and_logger(config)
    tester = Tester(config)
    # after getting trips of interest use tester.predict(data)

    directory = '/home/mogeng/attResearch/sample/CLF/MultipleCities/SF/single_month/20150601-20150628/trip_data/commuter/all/'
    for file in os.listdir(directory)[1::]:
        if file.endswith(".json"):
            filename = os.path.join(directory, file)
            print filename
            with open(filename) as data_file:    
                print data_file
                data = json.load(data_file)

            trips = data['trips']
            trips, trip_ids =  genLSTMin(trips)
  
            labels, probs =  tester.predict(trips, return_probs=True)
            
            for i in range(len(trip_ids)):
                data['trips'][trip_ids[i][0]][trip_ids[i][1]]['travel_mode_label'] = labels[i]
                data['trips'][trip_ids[i][0]][trip_ids[i][1]]['travel_mode_probs'] = list(probs[i].astype(float))

            #print labels, probs
            with open(filename,'w') as fp:
                json.dump(data,fp)


if __name__ == "__main__":
    
    directory = '/home/mogeng/attResearch/sample/CLF/MultipleCities/SF/single_month/20150601-20150628/trip_data/commuter/all/'
    

    # tensorflow LSTM 
    config, unparsed = get_config()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    

    print datetime.now()
    
    
