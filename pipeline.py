import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition.truncated_svd import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble.bagging import BaggingRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection.univariate_selection import SelectKBest, f_regression
from sklearn.gaussian_process import GaussianProcess
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.mixture.gmm import GMM
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing.data import PolynomialFeatures

import ds_tools.dstools.ml.transformers as tr
import ds_tools.dstools.ml.xgboost_tools as xgb

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


def create_union_transf(params):
    pca2c_transformer = make_pipeline(
        drop_transformer,
        Imputer(),
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
        drop_transformer,
        Imputer(),
        StandardScaler(),
        PCA(n_components=2),
        FunctionTransformer(lambda x: GMM(n_components=3).fit_predict(x)[np.newaxis].T)
    )

    transf = make_union(
        drop_transformer,
        gmm_transformer,
        os_transformer,
        arch_transformer,
        pca2c_transformer,
    )
    return transf


def create_xgb_est(params):
    keys = {
        'eta',
        'min_child_weight',
        'subsample',
        'colsample_bytree',
        'max_depth',
        'num_rounds',
        'num_es_rounds',
        'es_share',
        'num_parallel_tree'
    }
    
    xgb_params = {k: v for k, v in params.items() if k in keys}

    xgb_params_all = {
        "objective": "reg:linear",
        "silent": 0,
        'verbose': 10,
        'eval_func': mape_evalerror,
        'ybin_func': ybin,
        **xgb_params
    }
    
    objective = params['objective']
    if objective == 'mape':
        xgb_params_all['objective_func'] = mape_obj
    elif objective == 'poisson':
        xgb_params_all['objective'] = 'count:poisson'
        
    return xgb.XGBoostRegressor(**xgb_params_all)


def create_gblinear_est(params):
    keys = {
        'eta',
        'num_rounds',
        'num_es_rounds',
        'es_share',
        'lambda',
        'alpha',
    }
    
    xgb_params = {k: v for k, v in params.items() if k in keys}
    
    xgb_params_all = {
        "booster": 'gblinear',
        "objective": "reg:linear",
        "silent": 0,
        'verbose': 10,
        'eval_func': mape_evalerror,
        **xgb_params
    }
        
    return xgb.XGBoostRegressor(**xgb_params_all)


def validate(params):    
    transf_type = params['transf_type']
    
    if transf_type == 'drop':
        transf = FunctionTransformer(drop_transform, validate=False)
    elif transf_type == 'dr+inp+sc+pca':
        transf = make_pipeline(
            drop_transformer,
            Imputer(),
            StandardScaler(),
            PCA(n_components=params['n_pca_components']),
        )
    elif transf_type == 'dr+inp':
        transf = make_pipeline(
            drop_transformer,
            Imputer(),
        )
    elif transf_type == 'dr+inp+sc':
        transf = make_pipeline(
            drop_transformer,
            Imputer(),
            StandardScaler()
        )
    elif transf_type == 'union':
        transf = create_union_transf(params)
    elif transf_type == 'poly_kbest':
        transf = make_pipeline(
            drop_transformer,
            Imputer(),
            StandardScaler(),
            PolynomialFeatures(degree=2, interaction_only=True),
            SelectKBest(f_regression, params['best_features']),
        )
        
    est_type = params['est_type']

    if est_type == 'xgboost':
        est = create_xgb_est(params)
    elif est_type == 'gblinear':
        est = create_gblinear_est(params)
    elif est_type == 'exttree':
        est = ExtraTreesRegressor(n_estimators=params['n_estimators'], n_jobs=-1)
    elif est_type == 'gp':
        est = GaussianProcess(theta0=params['theta0'])
    elif est_type == 'ridge':
        est = Ridge(alpha=params['alpha'])
        
    if params['bagging']:
        BaggingRegressor(
            est,
            n_estimators=params['n_bag_estimators'],
            max_features=1.,
            max_samples=1.)
            
    pl = make_pipeline(transf, est)
            
    if params['per_group_regr']:
        pl = tr.PerGroupRegressor(
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
