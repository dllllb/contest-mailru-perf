import pandas as pd
import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.externals.joblib import Parallel
from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition.truncated_svd import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble.bagging import BaggingRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection.univariate_selection import SelectKBest, f_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing.data import PolynomialFeatures
from xgboost import XGBRegressor


def update_model_stats(stats_file, params, results):
    import json
    import os.path
    
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    else:
        stats = []
        
    stats.append({**results, **params})
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)

        
def run_experiment(evaluator, params, stats_file):
    import time
    
    params = init_params(params)
    start = time.time()
    scores = evaluator(params)
    exec_time = time.time() - start
    update_model_stats(stats_file, params, {**scores, 'exec-time-sec': exec_time})
    

def init_params(overrides):
    defautls = {
        'validation-type': 'cv',
        'n_folds': 5,
        "eta": 0.1,
        "min_child_weight": 6,
        "subsample": 0.7,
        "colsample_bytree": 0.5,
        "max_depth": 4,
        "num_rounds": 10000,
        "num_es_rounds": 50,
        "es_share": .05,
        'objective': 'linear',
        'num_parallel_tree': 1,
        'bagging': False,
        'per_group_regr': False,
        'est_type': 'xgboost',
    }
    return {**defautls, **overrides}


def predict_with_id(est, fdf):
    return pd.Series(est.predict(fdf), index=fdf.index)


def fit_clone_with_key(estimator, features, labels, key):
    from sklearn.base import clone as sk_clone
    return key, sk_clone(estimator).fit(features, labels)


class PerGroupRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, estimator, split_condition, n_jobs=1, verbose=0):
        self.split_condition = split_condition
        self.estimator = estimator
        self.group_estimators = None
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y):
        self.group_estimators = dict(Parallel(n_jobs=self.n_jobs)(
            (
                (fit_clone_with_key, [self.estimator, gdf, y.ix[gdf.index], gkey], {})
                for gkey, gdf in X.groupby(self.split_condition)
            )
        ))

        return self

    def predict(self, X):
        preds = pd.concat(Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            (
                (predict_with_id, [self.group_estimators[gkey], gdf], {})
                for gkey, gdf in X.groupby(self.split_condition)
            )
        ))

        return preds


def mape(y_true, y_pred):
    return np.average(np.abs((y_pred - y_true) / y_true), axis=0)


def mape_evalerror(preds, dtrain):
    return 'mape', mape(dtrain.get_label(), preds)


def mape_obj(preds, dtrain):
    labels = dtrain.get_label()
    grad = (preds - labels) / labels
    hess = np.full(len(preds), 1.)
    return grad, hess


def ybin(y):
    return np.digitize(np.log2(y), bins=np.arange(0, 9))


def dataset(path):
    x = pd.read_csv(path)
    x['memFreq'] = x.memFreq.replace('None', np.nan).astype(np.float64)
    x['memtRFC'] = x.memtRFC.replace('None', np.nan).astype(np.float64)
    return x


def cv_test(est, n_folds):
    x = dataset('x_train.csv.gz')

    y = pd.read_csv('y_train.csv', squeeze=True)

    scores = cross_val_score(est, x, y, scoring=make_scorer(mape), cv=n_folds)
    print('mean: {mean}, std: {std}'.format(mean=scores.mean(), std=scores.std()))
    
    
def submission(est, name='results'):
    x_tr = dataset('x_train.csv.gz')
    y_tr = pd.read_csv('y_train.csv', squeeze=True)

    m = est.fit(x_tr, y_tr)

    x_test = dataset('x_test.csv.gz')

    y_pred = m.predict(x_test)

    res = pd.Series(y_pred, index=x_test.index, name='time')
    res.to_csv(name + '.csv', header=False, index=False)
    
    
def pred_vs_true(est, path):
    x_tr = dataset('x_train.csv.gz')
    y_tr = pd.read_csv('y_train.csv', squeeze=True)

    x_train, x_test, y_train, y_test = train_test_split(x_tr, y_tr, train_size=0.9)
    y_pred = est.fit(x_train, y_train).predict(x_test)

    pd.DataFrame({'pred': y_pred, 'true': y_test}).to_csv(path, index=False, sep='\t')
    
    
def drop_transform(x):
    return x.drop(['memType', 'os', 'cpuFull', 'cpuArch'], axis=1)


def create_union_transf(_):
    pca2c_transformer = make_pipeline(
        drop_transform,
        SimpleImputer(),
        StandardScaler(),
        PCA(n_components=2),
    )

    os_transformer = make_pipeline(
        FunctionTransformer(lambda x: x.os, validate=False),
        CountVectorizer(),
        TruncatedSVD(n_components=10),
    )

    arch_transformer = FunctionTransformer(lambda x: pd.get_dummies(x.cpuArch), validate=False)

    gmm_transformer = make_pipeline(
        drop_transform,
        SimpleImputer(),
        StandardScaler(),
        PCA(n_components=2),
        FunctionTransformer(lambda x: GaussianMixture(n_components=3).fit_predict(x)[np.newaxis].T)
    )

    transf = make_union(
        drop_transform,
        gmm_transformer,
        os_transformer,
        arch_transformer,
        pca2c_transformer,
    )
    return transf


def create_xgb_est(params):
    keys = {
        'learning_rate',
        'min_child_weight',
        'subsample',
        'colsample_bytree',
        'max_depth',
        'n_estimators',
    }
    
    xgb_params = {
        "objective": "reg:linear",
        **{k: v for k, v in params.items() if k in keys}
    }
    
    objective = params['objective']
    if objective == 'mape':
        xgb_params['objective'] = mape_obj
    elif objective == 'poisson':
        xgb_params['objective'] = 'count:poisson'

    class XGBC(XGBRegressor):
        def fit(self, x, y, **kwargs):
            f_train, f_val, t_train, t_val = train_test_split(x, y, test_size=params['es_share'])
            super().fit(
                f_train,
                t_train,
                eval_set=[(f_val, t_val)],
                eval_metric=mape_evalerror,
                early_stopping_rounds=params['num_es_rounds'],
                verbose=10)
        
    return XGBC(**xgb_params)


def create_gblinear_est(params):
    keys = {
        'learning_rate',
        'n_estimators',
    }
    
    xgb_params = {
        "booster": 'gblinear',
        "objective": "reg:linear",
        **{k: v for k, v in params.items() if k in keys}
    }

    class XGBC(XGBRegressor):
        def fit(self, x, y, **kwargs):
            f_train, f_val, t_train, t_val = train_test_split(x, y, test_size=params['es_share'])
            super().fit(
                f_train,
                t_train,
                eval_set=[(f_val, t_val)],
                eval_metric=mape_evalerror,
                early_stopping_rounds=params['num_es_rounds'],
                verbose=10)
        
    return XGBC(**xgb_params)


def validate(params):    
    transf_type = params['transf_type']
    
    if transf_type == 'drop':
        transf = FunctionTransformer(drop_transform, validate=False)
    elif transf_type == 'dr+inp+sc+pca':
        transf = make_pipeline(
            drop_transform,
            SimpleImputer(),
            StandardScaler(),
            PCA(n_components=params['n_pca_components']),
        )
    elif transf_type == 'dr+inp':
        transf = make_pipeline(
            drop_transform,
            SimpleImputer(),
        )
    elif transf_type == 'dr+inp+sc':
        transf = make_pipeline(
            drop_transform,
            SimpleImputer(),
            StandardScaler()
        )
    elif transf_type == 'union':
        transf = create_union_transf(params)
    elif transf_type == 'poly_kbest':
        transf = make_pipeline(
            drop_transform,
            SimpleImputer(),
            StandardScaler(),
            PolynomialFeatures(degree=2, interaction_only=True),
            SelectKBest(f_regression, params['best_features']),
        )
    else:
        raise AttributeError(f'unknown transformer type: {transf_type}')
        
    est_type = params['est_type']

    if est_type == 'xgboost':
        est = create_xgb_est(params)
    elif est_type == 'gblinear':
        est = create_gblinear_est(params)
    elif est_type == 'exttree':
        est = ExtraTreesRegressor(n_estimators=params['n_estimators'], n_jobs=-1)
    elif est_type == 'gp':
        est = GaussianProcessRegressor()
    elif est_type == 'ridge':
        est = Ridge(alpha=params['alpha'])
    else:
        raise AttributeError(f'unknown estimator type: {est_type}')
        
    if params['bagging']:
        BaggingRegressor(
            est,
            n_estimators=params['n_bag_estimators'],
            max_features=1.,
            max_samples=1.)
            
    pl = make_pipeline(transf, est)
            
    if params['per_group_regr']:
        pl = PerGroupRegressor(
            estimator=pl,
            split_condition=['os', 'cpuFreq', 'memSize_MB'],
            n_jobs=1,
            verbose=1
        )

    return cv_test(pl, n_folds=params['n_folds'])

 
def test_validate():
    params = {
        "subsample": 0.5,
        "colsample_bytree": 0.3,
        'num_rounds': 10,
        'n_fodls': 3,
        'transf_type': 'drop',
    }
    print(validate(init_params(params)))
