#--coding:utf-8--
from __future__ import division
import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.core.framework import summary_pb2

import mnist_inference

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZE_RATE = 0.0001
TRAINING_STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = 'model/'
MODEL_NAME = 'model.ckpt'
SUMMARY_PATH = '/path/log'
FLAGS = None
MAX_LOSS = 9999
loss_larger = 0
SUM_LOSS = 0
# sum_loss_tensor = tf.Variable(0.0, name='sum_loss', trainable=False)

print np.shape(SUM_LOSS)

def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])


def train(mnist):
    global sum_loss_tensor, SUM_LOSS

    with tf.name_scope('placeholder'):
        x = tf.placeholder(dtype=tf.float32, shape=[None, mnist_inference.INPUT_NODE], name='x_input')
        target = tf.placeholder(dtype=tf.float32, shape=[None, mnist_inference.HIDDEN_NODE_2], name='target')

    with tf.name_scope('input_reshape'):
        image_shaped = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', tensor=image_shaped, max_outputs=10)



    regularize = tf.contrib.layers.l2_regularizer(REGULARIZE_RATE)

    pred = mnist_inference.inference(x, regularize)

    global_step = tf.Variable(0, trainable=False)
    # sum_loss_tensor = tf.placeholder(dtype=tf.float32, shape=(), name='sum_loss')

    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)#decay, num_updates

    variable_average_op = variable_average.apply(tf.trainable_variables())

    # 遍历一次训练数据后进行学习率的衰减
    learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE_BASE, global_step=global_step,
                                               decay_steps=mnist.train.num_examples / BATCH_SIZE,
                                               decay_rate=LEARNING_RATE_DECAY)
    tf.summary.scalar('learning_rate', learning_rate)

    with tf.name_scope('loss'):
        with tf.name_scope('cross_entropy'):
            cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(target, 1), logits=pred)

            cross_entropy_loss_mean = tf.reduce_mean(cross_entropy_loss)
            tf.summary.scalar('cross_entropy', cross_entropy_loss_mean)

        with tf.name_scope('l2_loss'):
            l2_loss = tf.add_n(tf.get_collection('losses'))
            tf.summary.scalar('l2_loss', l2_loss)

        loss = cross_entropy_loss_mean #+ tf.add_n(tf.get_collection('losses'))
        tf.summary.scalar('loss_for_mnist', loss)

    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train')


    saver = tf.train.Saver()

    merged = tf.summary.merge_all()


    with tf.Session() as sess:
        global MAX_LOSS, sum_loss_tensor
        train_writer = tf.summary.FileWriter('log/train', sess.graph)

        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):#一次训练一个batch
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            # _, loss_value, lr, step = sess.run([train_op, loss, learning_rate, global_step], feed_dict={x:xs, target:ys})


            if i % (mnist.train.num_examples / BATCH_SIZE) == 0:
                run_options = tf.RunOptions()
                run_metadata = tf.RunMetadata()
                summary, _, loss_value, lr, step = sess.run([merged, train_op, loss, learning_rate, global_step],
                                                            feed_dict={x:xs, target:ys},
                                                            options=run_options,
                                                            run_metadata=run_metadata)

                train_writer.add_run_metadata(run_metadata=run_metadata, tag='step%03d'%i)
                train_writer.add_summary(summary=summary, global_step=i)

                SUM_LOSS += loss_value * BATCH_SIZE
                mean_SUM_LOSS = SUM_LOSS / mnist.train.num_examples
                print 'An epoch finished! After %d training steps, learning rate is %.6g, loss on training batch is %.6g, sum of loss is %.6g, mean of sum_loss is %.6g'%(i / (mnist.train.num_examples / BATCH_SIZE), lr, loss_value, SUM_LOSS, mean_SUM_LOSS)

                train_writer.add_summary(make_summary('epoch_loss', mean_SUM_LOSS), i / (mnist.train.num_examples / BATCH_SIZE))

                SUM_LOSS = 0

                # sum_loss_tensor = tf.Variable(0.0, name='sum_loss', trainable=False)
                # sess.run(sum_loss_tensor.initializer)

                # if loss_value < MAX_LOSS:
                #     MAX_LOSS = loss_value
                #     loss_larger = 0
                #
                # else:
                #     loss_larger += 1
                #
                #     if loss_larger > 100:
                #         print 'sess closed at %d, min loss is %g, current loss is %g' % (i, MAX_LOSS, loss_value)
                #         sess.close()
                #         break


                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

            else:
                summary, _, loss_value, lr, step = sess.run([merged, train_op, loss, learning_rate, global_step],
                                                            feed_dict={x: xs, target: ys})
                train_writer.add_summary(summary=summary, global_step=i)

                SUM_LOSS += loss_value * BATCH_SIZE

                print 'After %d training steps, learning rate is %.6g, loss on training batch is %.6g, sum of loss is %.6g'%(step, lr, loss_value, SUM_LOSS)

                # if loss_value < MAX_LOSS:
                #     MAX_LOSS = loss_value
                #     loss_larger = 0
                #
                # else:
                #     loss_larger += 1
                #
                #     if loss_larger > 100:
                #         print 'sess closed at %d, min loss is %g, current loss is %g' % (i, MAX_LOSS, loss_value)
                #         sess.close()
                #         break

    train_writer.close()



def main(argv=None):
    mnist = input_data.read_data_sets('/tmp/data', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
