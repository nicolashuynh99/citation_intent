import pickle
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix

edge_train_embeddings = pickle.load(open("embs_edgelist.pkl","rb"))
edge_test_embeddings = pickle.load(open("embs_test_edgelist.pkl","rb"))

print("load ok")


def scp_to_XY(edgelist):
    X = np.array([x[2][0] for x in edgelist])
    Y = np.array([x[3] for x in edgelist])
    return X, Y



X, y = scp_to_XY(edge_train_embeddings)


xgb_model = xgb.XGBClassifier(objective="multi:softmax", verbosity=2, reg_alpha=1, reg_lambda=.1, n_estimators=200, max_depth=12)
xgb_model.fit(X, y)

X_test, y_test = scp_to_XY(edge_test_embeddings)

y_pred = xgb_model.predict(X_test)

print(f1_score(y_test, y_pred,average='micro'))

print(confusion_matrix(y_test,y_pred))
