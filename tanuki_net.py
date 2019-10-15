from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

def tanuki_net(input_shape, pool_size):
    inputs = Input(input_shape) 
    batch = BatchNormalization()(inputs) #0 
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(batch)
    drop1 = Dropout(0.2)(conv1)#2
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(drop1)
    drop1 = Dropout(0.2)(conv1)#4
    pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)

    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool1)
    drop2 = Dropout(0.2)(conv2)#7
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(drop2)
    drop2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(drop2)
    drop2 = Dropout(0.2)(conv2)#11
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)

    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool2)
    drop3 = Dropout(0.2)(conv3)#14
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(drop3)
    drop3 = Dropout(0.2)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

    up1 = UpSampling2D(size=pool_size)(pool3)
    deconv1 = Conv2DTranspose(128, 3, strides=(1, 1), padding='valid', activation = 'relu')(up1)
    drop4 = Dropout(0.2)(deconv1)
    deconv1 = Conv2DTranspose(128, 3, strides=(1, 1), padding='valid', activation = 'relu')(drop4)
    drop4 = Dropout(0.2)(deconv1)

    up2 = UpSampling2D(size=pool_size)(drop4)
    deconv2 = Conv2DTranspose(64, 3, strides=(1, 1), padding='valid', activation = 'relu')(up2)
    drop5 = Dropout(0.2)(deconv2)
    deconv2 = Conv2DTranspose(64, 3, strides=(1, 1), padding='valid', activation = 'relu')(drop5)
    drop5 = Dropout(0.2)(deconv2)
    deconv2 = Conv2DTranspose(64, 3, strides=(1, 1), padding='valid', activation = 'relu')(drop5)
    drop5 = Dropout(0.2)(deconv2)

    up3 = UpSampling2D(size=pool_size)(drop5)
    deconv3 = Conv2DTranspose(32, 3, strides=(1, 1), padding='valid', activation = 'relu')(up3)
    drop6 = Dropout(0.2)(deconv3)
    deconv3 = Conv2DTranspose(32, 3, strides=(1, 1), padding='valid', activation = 'relu')(drop6)
    deconv4 = Conv2DTranspose(1, 3, strides=(1, 1), padding='valid', activation = 'relu')(drop6)

    model = Model(input = inputs, output = deconv4)
    model.compile(optimizer = Adam(lr = 1e-4), loss='mean_square_error', metrics = ['accuracy'])

    return model