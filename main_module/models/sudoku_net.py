from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,Activation


class SudokuNet:
    @staticmethod

    def build(width,height,depth,classes):
        input_shape = (height,width,depth)

        model = Sequential()
        model.add(Conv2D(32,(5,5),padding='same',input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(32,(3,3),padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Flatten())

        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model

