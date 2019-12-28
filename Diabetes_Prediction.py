import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

tpot_data = pd.read_csv('D:/Machine Learning/iris/diabetes prediction/train.csv', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.7812072910912964
exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_classif, percentile=36),
    XGBClassifier(learning_rate=0.1, max_depth=7, min_child_weight=16, n_estimators=100, nthread=1, subsample=1.0)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
