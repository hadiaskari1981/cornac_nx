import cornac
import pandas as pd
from cornac.models import SVD
from cornac.eval_methods import RatioSplit, TimeSplit
from cornac import models
from cornac.metrics import MAE, RMSE, Precision, Recall, NDCG, AUC, MAP
from cornac.data.reader import Reader

ML_DATASETS = {
    "100K":
        {
            "url": "http://files.grouplens.org/datasets/movielens/ml-100k/u.data",
            "unzip": False,
            "path": "data/ml-100k/u.data",
            "sep": "\t",
            "skip": 0,
        }
}

reader = Reader()


nextory_se = reader.read("dataset/nextory_interactions_se.csv", "UIRD", sep=",", skip_lines=1)

ts = TimeSplit(data=nextory_se, rating_threshold=3.0, seed=123)

print('sdjqjdgjgq')
