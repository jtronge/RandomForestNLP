import csv
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def load_csv(fname, dialect='excel'):
    """Load a CSV file in and return it."""
    with open(fname) as fp:
        csv_reader = csv.reader(fp, dialect)
        return [tuple(line) for line in csv_reader]

def clean_text(text):
    """Clean with regex."""
    return re.sub('\W', ' ', text).split()

def convert_to_reals(words):
    """Do a very basic conversion from words to reals."""
    calculate_real = lambda w: sum(list(w)) / (len(w) * 255)
    return [calculate_real(bytes(w, 'utf-8')) for w in words]

def load_training_data(fname):
    """Load the training data."""
    all_data = load_csv(fname, 'excel-tab')

    labels = [rec[2] == 'OFF' for rec in all_data]
    data = [convert_to_reals(clean_text(rec[1])) for rec in all_data]
    max_features = max([len(rec) for rec in data])

    # Pad the data
    for rec in data:
        rec.extend([0.0] * (max_features - len(rec)))

    return labels, data, max_features
        
def load_test_data(label_fname, data_fname):
    """Load the test data and clean it for the training."""
    labels = load_csv(label_fname)
    data = load_csv(data_fname, 'excel-tab')

    # Join all data together on the ids given in the files
    joined_data = {}
    for label in labels:
        id = label[0]
        joined_data[id] = {'class': label[1]}
    for rec in data:
        id = rec[0]
        if id in joined_data:
            joined_data[id]['data'] = rec[1]

    # Clean and convert the data to reals
    max_features = 0
    for id in joined_data:
        words = clean_text(joined_data[id]['data'])
        reals = convert_to_reals(words)
        joined_data[id]['data'] = reals
        if len(reals) > max_features:
            max_features = len(reals)

    # Pad the data
    for id in joined_data:
        reals = joined_data[id]['data']
        joined_data[id]['data'] = reals + (max_features - len(reals)) * [0.0]

    # Prepare the data for training
    training_data = np.array([joined_data[id]['data'] for id in joined_data])
    training_labels = [joined_data[id]['class'] == 'OFF' for id in joined_data]
    return training_labels, training_data, max_features

def fit_features(data, max_features):
    """Squeeze/expand the feature data to fit max_features """
    ndata = []
    for rec in data:
        rec = list(rec)
        if len(rec) > max_features:
            rec = rec[:max_features]
        elif len(rec) < max_features:
            rec = rec + (max_features - len(rec)) * [0.0]
        ndata.append(rec)
    return np.array(ndata)

def main():
    training_labels, training_data, max_features = load_training_data(
        'OLID/olid-training-v1.0.tsv'
    )
    # Run Random Forest Algorithm
    model = RandomForestClassifier(n_estimators=100,
                                   bootstrap=True,
                                   max_features='sqrt')

    # Fit to the training data
    model.fit(training_data, training_labels)

    test_labels, test_data, _ = load_test_data('OLID/labels-levela.csv',
                                               'OLID/testset-levela.tsv')
    test_data = fit_features(test_data, max_features)
    predicted_labels = model.predict(test_data)

    print('Average true:', sum([1.0 if label else 0.0
                                for label in predicted_labels])
                           / len(predicted_labels) * 100)
    print('Average false:', sum([1.0 if not label else 0.0
                                 for label in predicted_labels])
                            / len(predicted_labels) * 100)
    print('Accuracy:', sum([1.0 if test_labels[i] == predicted_labels[i] else 0.0
                            for i in range(len(predicted_labels))])
                       / len(predicted_labels) * 100)

if __name__ == '__main__':
    main()
