import zipfile
import random
import glob
import os
import sklearn
import sklearn.svm
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.metrics
import pickle
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer


def preprocess(data):
    """
    Function which removes
    not important characters

    """

    final_message =\
    data.strip("Subject").strip(":").strip("\n").strip("\r").strip("re").strip("-")
    return final_message


def get_features(messages):

    """
    Function which converts the
    messages into feature or vectorize form

    """
#    vectorizer = CountVectorizer(min_df=1)
#    X_count_vec = vectorizer.fit_transform(messages)
#    transformer = TfidfTransformer(smooth_idf=False)
    vect = HashingVectorizer(decode_error='ignore', n_features=2**21,
                            preprocessor=None)
    X = vect.transform(messages)
    return X


def read_data(curr_dir):
    """
    Function which reads the
    two directories and preprocess
    and creates messages and labels list

    """

    ham_data = {}
    spam_data = {}
    messages_list = []
    temp_list = []
    messages = []
    labels = []

    # Reading Ham data
    ham_folder = glob.glob(curr_dir + "/dataset/ham" + "/*.txt")
    for file_item in ham_folder:
        with open(file_item, "r") as fp:
            ham_message = preprocess(fp.readline())
            messages_list.append({ham_message: 1})


    # Reading Spam data
    spam_folder = glob.glob(curr_dir + "/dataset/spam" + "/*.txt")
    for file_item in spam_folder:
        with open(file_item, "r") as fp:
            spam_message = preprocess(fp.readline())
            temp_list.append({spam_message: 0})
            messages_list.append({spam_message: 0})

    random_spam_mssgs = random.sample(temp_list, 1000)
    messages_list.extend(random_spam_mssgs)

    random.shuffle(messages_list)
    for item in messages_list:
        for key,value in item.items():
            messages.append(key)
            labels.append(value)

    return messages, labels


if __name__ == "__main__":


    curr_dir = dir_path = os.path.dirname(os.path.realpath(__file__))

    messages, labels = read_data(curr_dir)
    features = get_features(messages)
    X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                       test_size=0.33, random_state=42)

    print("\n\nSGD Classifier\n")
    clf = SGDClassifier(loss='log', random_state=1, n_iter=1)

    clf = clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    print(y_predicted)

    accuracy = accuracy_score(y_predicted,y_test)
    print("SGD Classifier Accuracy:", accuracy)
    print("SGD Classifier Report\n")
    print(sklearn.metrics.classification_report(y_test, y_predicted,
    target_names=None))

    print("SGD Classifier Confusion Matrix\n")
    print(sklearn.metrics.confusion_matrix(y_test, y_predicted))


    dest = os.path.join('pkl_objects')
    if not os.path.exists(dest):
        os.makedirs(dest)

    pickle.dump(clf, open(os.path.join(dest, 'classifier.pkl'), 'wb'))

