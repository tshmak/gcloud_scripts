"""Class and function for penalized regressions with tensorflow."""
import os
import numpy as np
import pickle
import tensorflow as tf
import datetime


class tensorflow_models(object):
    """Tensorflow implementations."""

    def __init__(self, X, y, model_log, overwrite=False, type: str = 'b'):
        """Tensorflow implementations."""
        super(tensorflow_models, self).__init__()
        self.model_log = model_log
        self.overwrite = overwrite
        self.X = X
        self.y = y.reshape(len(y), 1)
        if not os.path.isfile(model_log):
            with open(model_log, 'w', encoding='utf-8') as f:
                pickle.dump([], f)
        elif overwrite:
            with open(model_log, 'wb') as f:
                pickle.dump([], f)
        assert os.path.isfile(self.model_log)

    def _write_model(self, param, coef, score, model_name):
        output = {}
        output['param'] = param
        output['coef'] = coef
        output['score'] = score
        output['time'] = str(datetime.datetime.now())
        output['name'] = model_name
        print(output)
        feed = pickle.load(open(self.model_log, 'rb'))
        with open(self.model_log, 'wb') as f:
            feed.append(output)
            pickle.dump(feed, f)

    def _l1_penal(self, x, lamb):
        """L1 penalty."""
        lamb = tf.constant(lamb)
        return lamb*tf.reduce_sum(tf.abs(x))

    def _l2_penal(self, x, lamb):
        """L1 penalty."""
        lamb = tf.constant(lamb)
        return lamb*tf.reduce_sum(tf.square(x))

    def _loss(self, y_, y):
        if self.type == 'c':
            lostfunction = tf.reduce_mean(tf.square(y_ - y))
            return lostfunction
        elif self.type == 'b':
            lostfunction = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_, labels=y))
            return lostfunction
        else:
            raise ValueError('type has to be either c or b')

    def run(self, penal='l1', lamb=0.01, l_rate=0.01, epochs=201):
        """L1 norm as implemented in tensorflow."""
        if penal == 'l1':
            reg_fun = self._l1_penal
        elif penal == 'l2':
            reg_fun = self._l2_penal
        else:
            raise ValueError('no valid regularzation parameter')

        n, p = self.X.shape
        x = tf.placeholder(tf.float32, shape=[n, p])
        y = tf.placeholder(tf.float32, shape=[n, 1])
        debug = True

        # model
        W = tf.Variable(tf.zeros([p, 1]))
        b = tf.Variable(tf.zeros([1]))
        y_ = tf.matmul(x, W) + b
        regularization = reg_fun(W, lamb)
        lostfunction = self._loss(y_, y)
        cost = lostfunction + regularization
        train_op = tf.train.AdagradOptimizer(l_rate).minimize(cost)
        feed_dict = {x: self.X, y: self.y}
        init = tf.global_variables_initializer()

        cost_history = np.empty([0], dtype=float)
        with tf.Session() as sess:
            sess.run(init)
            last_cost = cost.eval(feed_dict)
            for _ in range(epochs):
                op, current_cost = sess.run([train_op, cost], feed_dict=feed_dict)

                if ((_ % 100 == 0) and debug):
                    print("loss = %f" % np.round(current_cost, 5))
                    correct_prediction = tf.equal(tf.round(tf.sigmoid(y_)), self.y)
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    print('prediction:', accuracy.eval(feed_dict))
                    if ((last_cost - current_cost) <= 1e-3) and (_ > 50):
                        break
                    else:
                        last_cost = current_cost

                cost_history = np.append(cost_history, current_cost)

            coef = tf.cast(W, tf.float32).eval(feed_dict)
            param = {
                    'learning_rate': l_rate,
                    'epoch': epochs,
                    'lambda': lamb,
                    'type': self.type,
                    'norm': lamb}

            self._write_model(
                param, coef,
                accuracy.eval(feed_dict),
                penal + ' tensorflow')
