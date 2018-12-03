from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Concatenate, Dropout
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from keras import backend as K
import time
import numpy as np
from keras.utils import plot_model
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

K.set_image_dim_ordering('th')

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet(img_height, img_width):
    inputs = Input((1, img_height, img_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization(axis=1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization(axis=1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization(axis=1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization(axis=1)(conv5)

    up6 = Concatenate(axis=1)([UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization(axis=1)(conv6)

    up7 = Concatenate(axis=1)([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization(axis=1)(conv7)

    up8 = Concatenate(axis=1)([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization(axis=1)(conv8)

    up9 = Concatenate(axis=1)([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization(axis=1)(conv9)
    # ---- test for dropout -----
    conv9 = Dropout(0.5)(conv9)
    # ---------------------------

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr=1.0e-3), loss=dice_coef_loss, metrics=[dice_coef])
    return model

# ---------------------------------

def save_conv2d(layer, f):
    shape = layer.weights[0].get_shape()
    weights = layer.get_weights()
    count = 0
    for k in range(0, shape[3]):
        for c in range(0, shape[2]):
                for h in range(0, shape[0]):
                    for w in range(0, shape[1]):
                        print(weights[0][h][w][c][k], file=f)
                        count = count + 1
    print(count)

def save_batchnorm(layer, f):
    weights = layer.get_weights()
    count = 0;
    for i in range(0, 4):
        for j in range(0, layer.weights[i].get_shape()[0]):
            print(weights[i][j], file=f)
            count = count + 1
    print(count)

def run():
    in_batch = 1
    in_size = 512

    model = get_unet(in_size, in_size)
    # plot_model(model, to_file='model.png', show_shapes=True)

    # weights_file = "test-data.h5"
    # if (os.path.isfile(weights_file)):
    #     model.load_weights(weights_file)
    #     print("Weights loaded\n");
    # else:
    #     model.save_weights(weights_file)
    #     print("Weights Saved\n");
    
    print("Exporting Weights")
    file = 'unet2d-weights.txt'
    with open(file, "w") as f:
        for layer in model.layers:
            print(layer.weights)
            if (type(layer) == type(model.layers[1])):
                print("conv2d")
                save_conv2d(layer, f)
            if (type(layer) == type(model.layers[3])):
                print("batchnorm")
                save_batchnorm(layer, f)

    img = np.random.randint(0, 255, in_batch * in_size * in_size).reshape(in_batch, 1, in_size, in_size)
    # img = np.ones((1, in_size, in_size, in_size, 1), dtype=float)

    print("Exporting Input")
    in_file = "unet2d-in.txt"
    with open(in_file, "w") as f:
        for n in range(in_batch):
            for c in range(1):
                    for h in range(in_size):
                        for w in range(in_size):
                            print(img[n][c][h][w], file=f)

    # ----------- Count the Time -----------

    out = model.predict(img)
    # for upper in range(20, 101, 20):
    #     start = time.time()
    #     for i in range(upper):
    #         out = model.predict(img)
    #     end = time.time() - start
    #     speed = in_batch * upper / end
    #     print("%d Speed of keras is %f img/sec.\n"%(upper, speed))
    
    # ----------- Count the Time -----------

    print("Exporting Output")
    out_file = "unet2d-out.txt"
    with open(out_file, "w") as f:
        for n in range(0, in_batch):
            for c in range(0, 1):
                    for h in range(0, in_size):
                        for w in range(0, in_size):
                            print(out[n][c][h][w], file=f)


if __name__ == '__main__':
    run()