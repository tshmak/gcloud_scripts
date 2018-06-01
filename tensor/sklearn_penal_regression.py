"""L1 and L2 norm as implemented in sklearn."""
import sklearn.linear_model as lm
import numpy as np
import os
import pprint
import datetime
import pickle


class sklearn_models(object):
    """Sklearn models."""

    def __init__(self, X, y, model_log, overwrite=False):
        """Sklearn models."""
        super(sklearn_models, self).__init__()
        self.model_log = model_log
        self.overwrite = overwrite
        self.X = X
        self.y = y
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

    def l1_model(self):
        """Sklearn L1 model."""
        model = lm.LogisticRegression(penalty='l1')
        model.fit(self.X, self.y)
        param = model.get_params()
        coef = model.coef_
        score = model.score(self.X, self.y)
        show_output = {
            'Model': 'L1 Norm sklearn',
            'time': str(datetime.datetime.now()),
            'score': score,
            'Num. of non-zero coef': np.sum(coef != 0)}
        pprint.pprint(show_output)
        self._write_model(param, coef, score, 'L1 Norm sklearn')

    def l2_model(self):
        """Sklearn L1 model."""
        model = lm.LogisticRegression(penalty='l2')
        model.fit(self.X, self.y)
        param = model.get_params()
        coef = model.coef_
        score = model.score(self.X, self.y)
        show_output = {
            'Model': 'L2 Norm sklearn',
            'time': str(datetime.datetime.now()),
            'score': score,
            'Num. of non-zero coef': np.sum(coef != 0)}
        pprint.pprint(show_output)
        self._write_model(param, coef, score, 'L2 Norm sklearn')
