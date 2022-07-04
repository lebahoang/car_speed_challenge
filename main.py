import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from cv_process.video_process import extract_frames
from cv_process.image_process import get_opticalflow_dense, preprocess_a_img
from optical_flow_liked_approaches.optical_model import OpticalModel
from optical_flow_liked_approaches.auto_optical_model import AutoOpticalModel, stack_2_frames
from steps import train

private_key = "sensitive key"

def prepare_traintest_set_for_opticalflow_liked_approaches(opticalflow_convert_func,
    frames_path='data/frames', given_labels_path = 'data/train.txt', test_set_size=1000):

    given_labels = [0]
    with open(given_labels_path, 'r') as f:
        for line in f.readlines():
            given_labels.append(float(line))

    frame_min = frame_max = -1
    for frame_path in os.listdir(frames_path):
        frame_ind = int(frame_path.split('.')[0])
        if frame_min == -1 or frame_ind < frame_min:
            frame_min = frame_ind
        if frame_max == -1 or frame_ind > frame_max:
            frame_max = frame_ind

    # frame_min = 1
    # frame_max = 4000
    # test_set_size = 10
    if test_set_size > frame_max:
        raise Exception('test set size could not be larger than frames set size')
    
    # randomize a test set from raw frames
    train_set = [] # shape (n_samples, 2,h,w,c) or (n_samples, h,w,c)
    labels = [] # shape (n_samples, 1)
    test_set = [] # shape (n_samples, 2,h,w,c) or (n_samples, h,w,c)
    test_labels = [] # shape (n_samples, 1)

    frames = np.random.permutation([i for i in range(frame_min+1, frame_max+1)])
    test_frames = frames[:test_set_size]
    train_frames = frames[test_set_size:]
    for frame in test_frames:
        img = preprocess_a_img('data/frames/{:d}.jpg'.format(frame))
        t = np.array(img)
        img_prev = preprocess_a_img('data/frames/{:d}.jpg'.format(frame-1))
        t_prev = np.array(img_prev)
        test_set.append(opticalflow_convert_func(t_prev, t))

        # speed = np.mean([given_labels[frame-1], given_labels[frame]])
        speed = given_labels[frame]
        test_labels.append(speed)
    for frame in train_frames:
        img = preprocess_a_img('data/frames/{:d}.jpg'.format(frame))
        t = np.array(img)
        img_prev = preprocess_a_img('data/frames/{:d}.jpg'.format(frame-1))
        t_prev = np.array(img_prev)
        train_set.append(opticalflow_convert_func(t_prev, t))

        speed = np.mean([given_labels[frame-1], given_labels[frame]])
        labels.append(speed)
    
    train_set = np.array(train_set)
    labels = np.transpose(np.array([labels]))
    test_set = np.array(test_set)
    test_labels = np.transpose(np.array([test_labels]))
    print(train_set.shape, labels.shape, test_set.shape, test_labels.shape)
    return train_set, labels, test_set, test_labels




tf.flags.DEFINE_float('learning_rate', 0.001,
    'Learning rate for Optimizer.')
tf.flags.DEFINE_integer('batch_size', 32, 'size of each batch')
tf.flags.DEFINE_integer('epochs', 100, 'epochs')
tf.flags.DEFINE_string('device', '/cpu:0', '')
tf.flags.DEFINE_string('model', 'auto_optical_model', '')
FLAGS = tf.flags.FLAGS

if __name__ == '__main__':
    # extract_frames(video_path='data/train.mp4', saving_folder='data/frames')

    # img = preprocess_a_img('data/frames/1.jpg')
    # img_next = preprocess_a_img('data/frames/2.jpg')
    # flow = get_opticalflow_dense(img, img_next)
    # print(img.shape, img_next.shape, flow.shape)

    if FLAGS.model == 'optical_model':
        train_set, labels, test_set, test_labels = prepare_traintest_set_for_opticalflow_liked_approaches(get_opticalflow_dense)
        tf.reset_default_graph()
        model = OpticalModel([66, 220, 3], device=FLAGS.device)
        # init session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # train
        train(model, sess, train_set, labels, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size)
        # test
        test_result = model.predict_in_batch(test_set, sess)
        print('Mean squared error', mean_squared_error(test_labels, test_result))
    elif FLAGS.model == 'auto_optical_model':
        train_set, labels, test_set, test_labels = prepare_traintest_set_for_opticalflow_liked_approaches(stack_2_frames)
        tf.reset_default_graph()
        model = AutoOpticalModel([66, 220, 3], device=FLAGS.device)
        # init session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # train
        train(model, sess, train_set, labels, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size)
        # test
        test_result = model.predict_in_batch(test_set, sess)
        print('Mean squared error', mean_squared_error(test_labels, test_result))

    
