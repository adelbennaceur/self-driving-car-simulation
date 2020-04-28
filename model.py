import keras
from keras.models import Sequential
from keras.layers import Conv2D , MaxPooling2D, Dropout, Flatten , Dense
from keras.optimizers import Adam ,SGD , RMSprop , Adagrad


def sfd_model(optimizer , learning_rate):
    '''
    nvidia self driving car inspired architecture.
    '''

    if optimizer == 'adagrad':
        optimizer = Adagrad(lr = learning_rate)
    elif optimizer == 'sgd':
        optimizer = SGD(lr = learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(lr = learning_rate)
    else:
        optimizer = Adam(lr = learning_rate)

    model = Sequential()
    model.add(Conv2D(24, 5, 5, subsample=(2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='elu'))
    model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    #model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation = 'elu'))
    #model.add(Dropout(0.5))
    model.add(Dense(64, activation = 'elu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation = 'elu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=optimizer)
    
    return model
