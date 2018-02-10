import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
import numpy as np
import math
import loader

relation_detail = 'basic'
loader.set_relation_detail(relation_detail)

FLAGS = tf.app.flags.FLAGS
seq_len = 15
word_dim = 300
cluster_dim = loader.cluster_len
position_input_dim = 2
input_dim = word_dim + cluster_dim + position_input_dim
pos_tag_embed_dim = 50
position_embed_dim = 50
pos_min = 1 - seq_len
pos_max = seq_len - 1
pos_tag_vocab_len = 46
pos_vocab_len = pos_max - pos_min + 1
default_l2 = 3.0 # l2 rescaling factor
max_l2_val = default_l2 * default_l2 / 2 # rescaling factor to use for VariableClippingOptimizer
output_dim = len(loader.labels)
n_filters = 150
batch_size = 50
pool_size = 3
min_window = 2
max_window = 5
dropout = 0.5


def freeze():
    # By Omid Alemi - Jan 2017
    input_graph_path = relation_detail + '.txt'
    checkpoint_path = relation_detail + '.ckpt'
    input_saver_def_path = ""
    input_binary = False
    output_node_names = "output"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_frozen_graph_name = 'frozen_' + relation_detail + '.pb'
    output_optimized_graph_name = 'optimized_' + relation_detail + '.pb'
    clear_devices = True
    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_frozen_graph_name, clear_devices, "")
    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(output_frozen_graph_name, "rb") as f:
        data = f.read()
        input_graph_def.ParseFromString(data)

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        ['input', 'keep_prob'], # an array of the input node(s)
        ['output'], # an array of output nodes
        tf.float32.as_datatype_enum)
    # Save the optimized graph
    f = tf.gfile.FastGFile(output_optimized_graph_name, "w")
    f.write(output_graph_def.SerializeToString())


def _variable_with_weight_decay(name, shape, wd = 1e-4):
    """
    Taken from https://github.com/yuhaozhang/sentence-convnet/blob/master/model.py
    """
    var = tf.get_variable(name, shape)
    if wd is not None and wd != 0.:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name=name + '_weight_loss')
    else:
        weight_decay = tf.constant(0.0, dtype=tf.float32)
    return var, weight_decay


class Graph:
    def __init__(self):
        pass


def _auc_pr(true, prob, threshold):
    """ Ignore this """
    pred = tf.where(prob > threshold, tf.ones_like(prob), tf.zeros_like(prob))
    tp = tf.logical_and(tf.cast(pred, tf.bool), tf.cast(true, tf.bool))
    fp = tf.logical_and(tf.cast(pred, tf.bool), tf.logical_not(tf.cast(true, tf.bool)))
    fn = tf.logical_and(tf.logical_not(tf.cast(pred, tf.bool)), tf.cast(true, tf.bool))
    pre = tf.truediv(tf.reduce_sum(tf.cast(tp, tf.int32)), tf.reduce_sum(tf.cast(tf.logical_or(tp, fp), tf.int32)))
    rec = tf.truediv(tf.reduce_sum(tf.cast(tp, tf.int32)), tf.reduce_sum(tf.cast(tf.logical_or(tp, fn), tf.int32)))
    return pre, rec


def another_f(true, prob):
    """ Generates micro and macro averaged F1 scores """
    pred = tf.argmax(prob, axis = 1)
    act = tf.argmax(true, axis = 1)
    mat = tf.confusion_matrix(act, pred)

    micro_p = tf.trace(mat) / tf.reduce_sum(mat)
    micro_r = micro_p
    macro_p = tf.reduce_mean(tf.truediv(tf.diag(mat), tf.reduce_sum(mat, axis=0)))
    macro_r = tf.reduce_mean(tf.truediv(tf.diag(mat), tf.reduce_sum(mat, axis=1)))

    micro_f = 2 * micro_p * micro_r / (micro_p + micro_r)
    macro_f = 2 * macro_p * macro_r / (macro_p + macro_r)
    return micro_f, macro_f


def mf(true, prob):
    """ Another way of generating macroaveraged F1 score """
    pred = tf.argmax(prob, axis = 1)
    act = tf.argmax(true, axis = 1)
    mat = tf.confusion_matrix(act, pred)
    fs = []
    for i in range(1, output_dim):
        pr = tf.truediv(mat[i, i], tf.reduce_sum(mat[i, :]))
        re = tf.truediv(mat[i, i], tf.reduce_sum(mat[:, i]))
        f1 = tf.truediv(2 * pr * re, pr + re)
        fs.append(f1)
    macro_f = tf.reduce_mean(tf.stack(fs))
    return macro_f


def jet_prf(true, prob):
    """ Returns a Jet-style scoring, where the Other type doesn't count """
    pred = tf.argmax(prob, axis = 1)
    act = tf.argmax(true, axis = 1)
    responses = tf.logical_not(tf.equal(pred, tf.zeros_like(pred)))
    eqs = tf.equal(pred, act)
    corrects = tf.logical_and(eqs, responses)
    correct = tf.reduce_sum(tf.cast(corrects, tf.int32))
    resp_count = tf.reduce_sum(tf.cast(responses, tf.int32))
    gold_count = tf.reduce_sum(tf.cast(tf.logical_not(tf.equal(act, tf.zeros_like(act))), tf.int32))

    pre = tf.truediv(correct, resp_count)
    rec = tf.truediv(correct, gold_count)
    return pre, rec


def accuracy(true, prob):
    pred = tf.argmax(prob, axis = 1)
    act = tf.argmax(true, axis = 1)
    eqs = tf.equal(pred, act)
    return tf.reduce_mean(tf.cast(eqs, tf.float32))


def get_word_embedding(x_word, trainable = True):
    unk_word_embed = tf.get_variable(name = "unk_word_embed", shape = (1, word_dim), trainable = trainable)
    pretrained_word_embed = tf.get_variable(name = "pretrained_word_embed", shape = (loader.num_words, word_dim), initializer = tf.constant_initializer(loader.embeddings, dtype=tf.float32), trainable = trainable)
    word_embed = tf.concat([pretrained_word_embed, unk_word_embed], 0)
    x_word_embed = tf.nn.embedding_lookup(word_embed, x_word)
    return x_word_embed


def get_pos_tag_embedding(x_pos_tag):
    pos_tag_embed_mat = tf.get_variable("pos_tag_embedding_mat", [pos_tag_vocab_len, pos_tag_embed_dim])
    x_pos_tag_embed = tf.nn.embedding_lookup(pos_tag_embed_mat, x_pos_tag)
    return x_pos_tag_embed


def get_position_embedding(x_position):
    pos_embed_mat = tf.get_variable("pos_embedding_mat", [pos_vocab_len, position_embed_dim])
    x_position_embed = tf.nn.embedding_lookup(pos_embed_mat, tf.cast(tf.add(x_position, -pos_min), tf.int32))
    return tf.reshape(x_position_embed, [-1, seq_len, 2 * position_embed_dim])


def get_embedded_input(x_word = None, x_brown_cluster = None, x_pos_tag = None, x_position = None):
    inputs = []
    if x_word is not None:
        inputs.append(get_word_embedding(x_word))
    if x_brown_cluster is not None:
        inputs.append(x_brown_cluster)
    if x_pos_tag is not None:
        inputs.append(get_pos_tag_embedding(x_pos_tag))
    if x_position is not None:
        inputs.append(get_position_embedding(x_position))
    x = tf.concat(inputs, axis = 2)
    return x


def get_cnn(x):
    weights = []
    losses = []
    pool_tensors = []
    for k_size in range(min_window, max_window + 1):
        with tf.variable_scope('conv-%d' % k_size) as scope:
            kernel, wd = _variable_with_weight_decay('kernel-%d' % k_size, [k_size, x.get_shape()[2], n_filters])
            weights.append(kernel)
            losses.append(wd)
            conv = tf.nn.conv1d(value=x, filters=kernel, stride = 1, padding = 'VALID')
            biases = tf.get_variable(name='bias-%d' % k_size, shape=[n_filters])
            bias = tf.nn.bias_add(conv, biases)
            activation = tf.nn.tanh(bias, name=scope.name)
            # shape of activation: [batch_size, conv_len, n_filters]
            # conv_len = activation.get_shape()[1]
            expanded = tf.expand_dims(activation, 1)
            pool = tf.nn.max_pool(expanded, ksize=[1, 1, expanded.get_shape()[2], 1], strides=[1, 1, 1, 1], padding='VALID')
            # shape of pool: [batch_size, 1, 1, num_kernel]
            feature_size = int(pool.get_shape()[2] * pool.get_shape()[3])
            pool_tensors.append(tf.reshape(pool, [-1, feature_size]))
    pool_layer = tf.concat(pool_tensors, 1, name='pool')
    return pool_layer, weights, losses


def get_bigru(x, keep_prob, is_training):
    num_hidden = 60
    num_out = 30
    with tf.variable_scope('bigru'):
        kernel, wd = _variable_with_weight_decay('bigru_w', [2 * num_hidden, num_out])
        bias = tf.get_variable('bigru_b', [num_out])
        gru_cell_forward = tf.nn.rnn_cell.GRUCell(num_hidden)
        gru_cell_backward = tf.nn.rnn_cell.GRUCell(num_hidden)
        gru_cell_forward = tf.nn.rnn_cell.DropoutWrapper(gru_cell_forward, output_keep_prob=keep_prob)
        gru_cell_backward = tf.nn.rnn_cell.DropoutWrapper(gru_cell_backward, output_keep_prob=keep_prob)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(gru_cell_forward, gru_cell_backward, x, dtype=tf.float32)
        layer = tf.concat(outputs, 2)[:, -1, :]
        z = tf.nn.bias_add(tf.matmul(layer, kernel), bias)
        # normalized = tf.layers.batch_normalization(z, training = is_training)
        activation = tf.nn.tanh(z)
        return activation, [kernel], [wd]


def get_graph(x_word, x_brown_cluster, x_pos_tag, x_position, y):
    """
    Replicates the neural network by:
    Thien Huu Nguyen and Ralph Grishman. Relation extraction: Perspective from convolutional neural networks. In VS@ HLT-NAACL, pages 39–48, 2015
    """
    losses = []
    weights = []
    y_conv = None
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    is_training = tf.placeholder(tf.bool, name = 'is_training')
    x = get_embedded_input(x_word, x_brown_cluster, x_pos_tag, x_position)
    print(x)
    # Can choose between CNN and Bi-GRU
    x, _, layer_losses = get_cnn(x)
    # x, _, layer_losses = get_bigru(x, keep_prob, is_training)
    losses.extend(layer_losses)
    print(x)
    """ Some papers employ dropout at this layer
    with tf.variable_scope('dropout') as scope:
        x = tf.nn.dropout(x, keep_prob)
    """
    fc1_kernel, fc1_wd = _variable_with_weight_decay('fc1-kernel', [x.get_shape()[1], output_dim])
    weights.append(fc1_kernel)
    fc1_bias = tf.get_variable('fc1-bias', [output_dim])
    y_conv = tf.nn.bias_add(tf.matmul(x, fc1_kernel), fc1_bias, name = 'output')
    losses.append(fc1_wd)

    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = y_conv, labels = y,
                                                                name = 'cross_entropy_per_example')
        cross_entropy_loss = tf.reduce_mean(cross_entropy, name = 'cross_entropy_loss')
        losses.append(cross_entropy_loss)

    with tf.variable_scope('adam_optimizer'):
        total_loss = tf.add_n(losses, name='total_loss')
        tf.summary.histogram('histogram', total_loss)
        optimizer = tf.train.GradientDescentOptimizer(1e-1, name='adamOptimizer')
        train_step = optimizer.minimize(total_loss)
    """ This can be uncommented to enable weight clipping
    with tf.variable_scope('clipper'):
        clipper = tf.contrib.opt.VariableClippingOptimizer(optimizer, {a:list(range(len(a.shape.as_list()))) for a in weights}, max_l2_val, name='clippingOptimizer')
        train_step = clipper.minimize(cross_entropy_loss)

        #.minimize(cross_entropy_loss) # total_loss)
        #clips = [w.assign(tf.clip_by_norm(w, max_l2_val)) for w in tf.trainable_variables()]
    """
    with tf.variable_scope('evaluation') as scope:
        pre, rec = _auc_pr(y, tf.sigmoid(y_conv), 0.1)
        fscore = 2 * pre * rec / (pre + rec)
    with tf.variable_scope('jet_evaluation') as scope:
        jet_pre, jet_rec = jet_prf(y, tf.sigmoid(y_conv))
        jet_f = 2 * jet_pre * jet_rec / (jet_pre + jet_rec)
    with tf.variable_scope('micromacro_evaluation') as scope:
        micro_f, macro_f = another_f(y, tf.sigmoid(y_conv))
    with tf.variable_scope('accuracy') as scope:
        acc = accuracy(y, tf.sigmoid(y_conv))
    with tf.variable_scope('mf') as scope:
        macro_ff = mf(y, tf.sigmoid(y_conv))
    return y_conv, keep_prob, is_training, train_step, fscore, jet_f, micro_f, macro_f, acc, macro_ff


def turn_one_hot(X, n = output_dim):
    # Turns a matrix into one-hot tensor
    X = X.reshape((X.shape[0]))
    o = np.zeros((X.shape[0], n))
    for i in range(n):
        o[X == i, i] = 1
    return o


def main():
    x_train_words, x_train_brown_clusters, x_train_pos_tags, x_train_positions, y_train = loader.load_training()
    n_train = x_train_brown_clusters.shape[0]
    y_train = turn_one_hot(y_train, n = output_dim)
    x_test_words, x_test_brown_clusters, x_test_pos_tags, x_test_positions, y_test = loader.load_test()
    y_test = turn_one_hot(y_test, n = output_dim)
    x_word = tf.placeholder(tf.int32, [None, seq_len], name = 'input_word')
    x_brown_cluster = tf.placeholder(tf.float32, [None, seq_len, cluster_dim], name = 'input_brown_cluster')
    x_pos_tag = tf.placeholder(tf.int32, [None, seq_len], name = 'input_pos_tag')
    x_position = tf.placeholder(tf.float32, [None, seq_len, position_input_dim], name = 'input_position')
    y_ = tf.placeholder(tf.float32, [None, output_dim], name = 'actual_output')
    y_conv, keep_prob, is_training, train_step, fscore, jet_fscore, micro_f, macro_f, acc, macro_ff = get_graph(x_word, x_brown_cluster, x_pos_tag, x_position, y_)
    n_batches = math.ceil(n_train * 1.0 / batch_size)
    x_ws = [x_train_words[i * batch_size: min((i + 1) * batch_size, n_train)] for i in range(n_batches)]
    x_bcs = [x_train_brown_clusters[i * batch_size: min((i + 1) * batch_size, n_train)] for i in range(n_batches)]
    x_ptgs = [x_train_pos_tags[i * batch_size: min((i + 1) * batch_size, n_train)] for i in range(n_batches)]
    x_poss = [x_train_positions[i * batch_size: min((i + 1) * batch_size, n_train)] for i in range(n_batches)]
    ys = [y_train[i * batch_size: min((i + 1) * batch_size, n_train)] for i in range(n_batches)]
    saver = tf.train.Saver()
    best_f = -1
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        for i in range(40000):
            train_step.run(feed_dict={
                           x_word: x_ws[i % len(x_ws)],
                           x_brown_cluster: x_bcs[i % len(x_bcs)],
                           x_pos_tag: x_ptgs[i % len(x_ptgs)],
                           x_position: x_poss[i % len(x_poss)],
                           y_: ys[i % len(ys)],
                           keep_prob: 1 - dropout,
                           is_training: True})
            if i % 100 == 0:
                jet_f = jet_fscore.eval(feed_dict = {
                                        x_word: x_test_words,
                                        x_brown_cluster: x_test_brown_clusters,
                                        x_pos_tag: x_test_pos_tags,
                                        x_position: x_test_positions,
                                        y_: y_test,
                                        keep_prob: 1,
                                        is_training: False})
                macf = macro_ff.eval(feed_dict = {
                                     x_word: x_test_words,
                                     x_brown_cluster: x_test_brown_clusters,
                                     x_pos_tag: x_test_pos_tags,
                                     x_position: x_test_positions,
                                     y_: y_test,
                                     keep_prob: 1,
                                     is_training: False})
                print('step %d, jet-f-score %g' % (i, jet_f))
                print('step %d, macf %g' % (i, macf))
                if jet_f > best_f + 0.01 and jet_f > 0.4: # With long enough training, we beat 0.4 in all trials
                    best_f = jet_f
                    saver.save(sess, '%s.ckpt' % relation_detail)
                    tf.train.write_graph(sess.graph_def, '.', '%s.proto' % relation_detail, as_text=False)
                    tf.train.write_graph(sess.graph_def, '.', '%s.txt' % relation_detail, as_text=True)
                    freeze()

if __name__ == '__main__':
    main()
