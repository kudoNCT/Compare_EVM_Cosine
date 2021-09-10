import sys

sys.path.append('/content/tiny_demo_face_recognition/models/insightface/deploy')
sys.path.append('/content/tiny_demo_face_recognition/models/insightface/src/common')

from keras.models import load_model
from imutils import paths
import face_preprocess
import numpy as np
import face_model
import argparse
import pickle
import time
import cv2
import os
from glob import glob
from train_evm import EVM
import scipy.spatial.distance
import sklearn.metrics.pairwise
from sklearn.base import BaseEstimator
from sklearn.datasets import load_digits
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV


def euclidean_cdist(X, Y):
    return sklearn.metrics.pairwise.pairwise_distances(X, Y, metric="euclidean", n_jobs=1)


def euclidean_pdist(X):
    return sklearn.metrics.pairwise.pairwise_distances(X, metric="euclidean", n_jobs=1)


def cosine_cdist(X, Y):
    return sklearn.metrics.pairwise.pairwise_distances(X, Y, metric="cosine", n_jobs=1)


def cosine_pdist(X):
    return sklearn.metrics.pairwise.pairwise_distances(X, metric="cosine", n_jobs=1)


dist_func_lookup = {
    "cosine": {"cdist": cosine_cdist,
               "pdist": cosine_pdist},

    "euclidean": {"cdist": euclidean_cdist,
                  "pdist": euclidean_pdist}
}

ap = argparse.ArgumentParser()

ap.add_argument("--embeddings", default="outputs/embeddings.pickle",
                help='Path to embeddings')
ap.add_argument("--weibull", default="outputs/weibull.pickle",
                help='Path to weibull')
ap.add_argument('--image-size', default='112,112', help='')
ap.add_argument('--model', default='./models/insightface/models/model-y1-test2/model,0', help='path to load model.')
ap.add_argument('--ga-model', default='', help='path to load model.')
ap.add_argument('--gpu', default=0, type=int, help='gpu id')
ap.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
ap.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
ap.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')

args = ap.parse_args()

# Load embeddings and labels
data = pickle.loads(open(args.embeddings, "rb").read())

embeddings = np.array(data['embeddings'])
labels = data['names']

# Initialize faces embedding model
embedding_model = face_model.FaceModel(args)


# Define distance function
def findCosineDistance(vector1, vector2):
    """
    Calculate cosine distance between two vector
    """
    vec1 = vector1.flatten()
    vec2 = vector2.flatten()

    a = np.dot(vec1.T, vec2)
    b = np.dot(vec1.T, vec1)
    c = np.dot(vec2.T, vec2)
    return (a / (np.sqrt(b) * np.sqrt(c)))


def CosineSimilarity(test_vec, source_vecs):
    """
    Verify the similarity of one vector to group vectors of one class
    """
    cos_dist = 0
    for source_vec in source_vecs:
        cos_dist += findCosineDistance(test_vec, source_vec)
    return cos_dist / len(source_vecs)


def get_accuracy(predictions, labels):
    return sum(predictions == labels) / float(len(predictions))


# Setup some useful arguments
cosine_threshold = 0.8
weibull = EVM()
weibull = pickle.loads(open(args.weibull, "rb").read())
UUK = "Unknown"

kkc_uuc = ['test_close']
low_score = 0
for type_class in kkc_uuc:
    y_true = []
    y_pred = []
    for path_img in glob(f'./VN_celeb_openset/{type_class}/*/*'):
        true_lb = path_img.split('/')[-2]
        if type_class == 'unknown_set':
            y_true.append(UUK)
        else:
            y_true.append(true_lb)
        nimg = cv2.imread(path_img)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        nimg = np.transpose(nimg, (2, 0, 1))
        embedding = embedding_model.get_feature(nimg).reshape(1, -1)

        final_score = 0
        final_lb = 0
        # Calculate cosine similarity
        (lb_pred, score_pred) = weibull.predict_with_prob(embedding)
        if lb_pred == true_lb and score_pred < 0.5:
            # print(f'low score: {path_img}')
            low_score += 1
        y_pred.append(lb_pred)
    print(f'num low score: {low_score}')
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy = get_accuracy(y_pred, y_true)
    print(f"{type_class} : accuracy: {round(accuracy * 100, 2)}%")



