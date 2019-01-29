import math
import numpy as np


def train(model, sess,
    images, labels,
    epochs=10, batch_size=64):

    i = 0
    total_size = images.shape[0]
    num_minibatches = math.floor(total_size/batch_size)
    print('number of mini batchs', num_minibatches)
    while i < epochs:
        permutation = list(np.random.permutation(total_size))
        images = images[permutation,:,:,:]
        labels = labels[permutation,:]

        total_loss = 0.0
        for k in range(0, num_minibatches):
            mini_batch_images = images[k*batch_size:(k+1)*batch_size,:,:,:]
            mini_batch_labels = labels[k*batch_size:(k+1)*batch_size,:]
            loss, _ = model.fit(mini_batch_images, mini_batch_labels, sess)
            total_loss += loss
        if total_size%batch_size != 0:
            mini_batch_images = images[num_minibatches*batch_size:total_size,:,:,:]
            mini_batch_labels = labels[num_minibatches*batch_size:total_size,:]
            loss, _ = model.fit(mini_batch_images, mini_batch_labels, sess)
            total_loss += loss

        if i % 10 == 0:
            print('total loss', total_loss)
        if total_loss < 1:
            print('COOL loss below 1 !!!', total_loss)
            break
        i += 1