import tensorflow as tf

INPUT_NODE = 784
HIDDEN_NODE_1 = 500
HIDDEN_NODE_2 = 10

def variable_summary(var, name):
    with tf.name_scope('summary'):
        tf.summary.histogram(name=name, values=var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/'+name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('std/'+name, stddev)
        tf.summary.scalar('min/'+name, tf.reduce_min(var))
        tf.summary.scalar('max/'+name, tf.reduce_max(var))
        

def get_weight(shape_def, regularize):
    weight = tf.get_variable('weights', shape=shape_def, initializer=tf.truncated_normal_initializer(stddev=0.1))

    if regularize:
        tf.add_to_collection('losses', regularize(weight))

    return weight


def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1_scope'):
        weight = get_weight(shape_def=[INPUT_NODE, HIDDEN_NODE_1], regularize=regularizer)
        variable_summary(weight, name='weight')
        
        bias = tf.get_variable(name='bias', shape=[HIDDEN_NODE_1], initializer=tf.constant_initializer(0.0))
        variable_summary(bias, name='bias')

        no_act = tf.matmul(input_tensor, weight) + bias
        variable_summary(no_act, name='no_act')
        
        layer1 = tf.nn.relu(no_act)
        variable_summary(layer1, name='act')


    with tf.variable_scope('layer2_scope'):
        weight = get_weight(shape_def=[HIDDEN_NODE_1, HIDDEN_NODE_2], regularize=regularizer)
        variable_summary(weight, name='weight')
        
        bias = tf.get_variable(name='bias', shape=[HIDDEN_NODE_2], initializer=tf.constant_initializer(0.0))
        variable_summary(bias, name='bias')

        layer2 = tf.matmul(layer1, weight) + bias
        variable_summary(layer2, name='no_act')

    return layer2

