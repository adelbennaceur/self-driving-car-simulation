# -*- coding: utf-8 -*-
"""
file name : main.py
author : adel bennaceur
created : 14 march 2019
"""

import argparse
from sklearn.model_selection import train_test_split
from utils import *
from model import sfd_model


parser = argparse.ArgumentParser(description='command line for diffrent parameters')
parser.add_argument('-dir', type = str ,required = True,
                    help='data directory for the csv and the images')
parser.add_argument('-optimizer', type = str ,required = False , default = 'adam',
                    help='possible arguments : sgd , adagrad , adam , rmsprop. default: Adam')
parser.add_argument('-lr', type = float ,required = False , default = 0.0001,
                    help='learning rate for the optimizer ex : 0.01, 0,001 . default : 0.001')
parser.add_argument('-batch_size', type = int ,required = False , default = 32 ,
                    help='how many samples per batch. default : 32')
parser.add_argument('-epochs', type = int ,required = False , default = 25 ,
                    help='how many samples per batch. default : 32')

args = parser.parse_args()
data_dir =  args.dir

#hyperparameters
optimizer = args.optimizer
lr = args.lr
batch_size = args.batch_size
nb_epchs = args.epochs


#load the driving log data
data = load_data(data_dir, path_del)

#balancing the ground truth data distribution to steering = 0
new_data = delete_useless_data(data)

#loading images along with the correspending steering angles
img_paths , steerings = load_img_steering(data_dir + '/IMG' , new_data)

#training and validation split
X_train , X_val , y_train  , y_val = train_test_split(img_paths , steerings , test_size=0.2 , random_state = 6)
print('Training samples : {}\nValidation samples : {}'.format(len(X_train) , len(X_val)) )

#create batch generated data for efficiency.
X_train_gen , y_train_gen = next(batch_generator(X_train, y_train, 1 ,  batch_size = batch_size))
X_valid_gen , y_val_gen = next(batch_generator(X_val, y_val,0 ,batch_size = batch_size))

model = sfd_model(optimizer = optimizer  , learning_rate = args.lr)

#train the model on data generated batch by batch
history = model.fit_generator(batch_generator(X_train , y_train ,1 , batch_size = batch_size) ,
                              steps_per_epoch = 300,
                              epochs = nb_epchs ,
                              validation_data = batch_generator(X_val , y_val ,  100, 0) ,
                              validation_steps = 200,
                              verbose = 1 ,
                              shuffle = 1)


#Save the model
model.save('model_b.h5')
