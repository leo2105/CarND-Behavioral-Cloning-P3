import pandas as pd
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Lambda, Dropout, Dense, Flatten
from keras.layers.convolutional import Convolution2D, Cropping2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

lr = 1e-4
epochs = 10
batch_size = 50
keep_prob = 0.5

def load():
    data = pd.read_csv('./driving_log.csv', names=['center','left','right','steering','throttle','reverse','speed'])

    X = data[['center','left','right']].values
    y = data['steering'].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)

    return X_train, X_val, y_train, y_val

def model():

    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5), input_shape=(160, 320, 3))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Dropout(keep_prob))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.summary()

    return model

def train(model, X_train, X_val, y_train, y_val):
    checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1)
    model.compile(loss='MSE', optimizer=Adam(lr=lr))
    history_object = model.fit_generator(
            batch_generator(X_train, y_train, './data/data', batch_size, True),
            samples_per_epoch=20000,
            validation_data = batch_generator(X_val, y_val, './data/data', 40, False),
            nb_val_samples = len(X_val),
            nb_epoch = 10,
            verbose = 1)
    print(history_object.history.keys())

    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model MSE loss')
    plt.ylabel('MSE loss')
    plt.xlabel('epochs')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

def main():
    print('Behavioral Cloning Training Program')

    data = load()
    modelo = model()
    train(modelo, *data)

if __name__ == '__main__':
    main()
