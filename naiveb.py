import numpy as np
import random
from collections import Counter, defaultdict
from sys import argv

def getData(file_name, data):
    file = open(file_name, 'r')
    for file_data in file:
        if file_data[0] != '@' and file_data[0] != '%' and file_data[0] != '\n':
            data.append(file_data.split(','))

def splitData(data, split):
    random.shuffle(data)
    train_data = data[:int(split*len(data))]#percent wise
    test_data = data[int(split*len(data)):]
    return train_data, test_data

def stripTarget(data):
    for index, item in enumerate(data):
        item[-1] = item[-1].strip()

def is_number(a):
    # will be True also for 'NaN'
    try:
        number = float(a)
        return True
    except ValueError:
        return False

def separateData(data):
    numeric = []
    categoric = []
    for index, item in enumerate(data): # index 0,1,2,3 item are  array of 19 elements
        num =[]
        cat =[]
        for i in range(len(item)):
            if is_number(item[i]):
                num.append(item[i])
            else:
                cat.append(item[i])
        numeric.append(num)
        categoric.append(cat)
    return numeric, categoric

def targetVal(data):
    temp = []
    for i in range(len(data)):
        temp.append(data[i][-1])
    return temp

def separateByClass(dataset):
    separated ={}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        else:
            separated[vector[-1]].append(vector)
    return separated

def NB(training_target, train_no_target, testing_data):
    classes = np.unique(training_target)
    data, labels = np.shape(train_no_target)
    likelihoods = {}
    answer = {}
    temp =[]
    for res in classes:
        likelihoods[res] = defaultdict(list)
    propabilities_train = occurrances(training_target)
    #print('propabilities', propabilities_train)
    for res in classes:
        for i in range(len(training_target)):
            if res == training_target[i]:
                temp.append(i)
        subset = train_no_target[temp, :]
        val_in, lab_in = np.shape(subset)
        for i in range(0, lab_in):
            likelihoods[res][i] += list(subset[:, i])
        temp =[]
    for res in classes:
        for i in range(0, labels):
            likelihoods[res][i] = occurrances(likelihoods[res][i])
    for res in classes:
        single_prob_train = propabilities_train[res]
        for i in range(0, len(testing_data)):
            pre_val =likelihoods[res][i]
            if testing_data[i] in pre_val.keys():
                single_prob_train *= pre_val[testing_data[i]]
            else:
                single_prob_train *= 0
            answer[res] = single_prob_train
    return answer

def occurrances(data):
    data_length = len(data)
    prob = dict(Counter(data))
    for item in prob.keys():
        prob[item] = prob[item] /float(data_length)
    return prob

def getRidOfTarget(data):
    for index, item in enumerate(data):
        item.pop()
    return data

def getPredictions(train_target, train_no_target, test_no_target):
    result = []
    for index, item in enumerate(test_no_target):
        result.append(NB(train_target, train_no_target, item))
    return result

def highestElement(data):
    myMax = data[0]
    for num in data:
        if myMax < num:
            myMax = num
    return myMax

def higest(data):
    myMax = data[0]
    temp =[]
    for index, item in enumerate(data):
        if myMax < item:
            myMax = item
    return (data.index(myMax),myMax)

def weightAnswer(prediction, target):
    temp = []
    high_idx = []
    result =[]
    classes = np.unique(target)
    class_len = len(classes)
    for index, item in enumerate(prediction):
        temp.append((list(item.values()), list(classes)))
    for index, item in enumerate(temp):
        len_item = len(item)
        if class_len >= len_item:
            high_idx.append(higest(item[0]))
        else:
            raise ValueError('Error: There are missing values', 'target values', class_len, 'number of items', len_item)
    #print(classes)
    #print(high_idx)
    for index, item in enumerate(high_idx):
        result.append(classes[item[0]])
    return result

def accuracy(test_target, predictions):
    correctness = 0
    for i in range(len(test_target)):
        if test_target[i] == predictions[i]:
            correctness +=1
    return (correctness/ float(len(test_target)))*100.0

def chechForNumderic(data):
    if is_number(data):
        raise ValueError('Error: There is a numeric value in data')

if __name__ == '__main__':

    command, filename, testSet= argv
    target_pr = []
    data =[]
    train = []
    test = []
    #file_name = '/home/plague/PycharmProjects/CS295JupyterHW/arffFiles/Train/contact-lenses.arff'
    getData(filename, train)
    getData(testSet, test)
    #rint("test data len ", len(test))
    #print('train data len', len(train))
    #train, test = splitData(data, 0.67)
    stripTarget(train)
    stripTarget(test)
    train_num, train_cat = separateData(train)
    test_num, test_cat = separateData(test)

    target_train = targetVal(train_cat)    # Training target value
    target_test = targetVal(test_cat)     # Test target vlaue

    train_no_target = np.asarray(getRidOfTarget(train_cat))
    train_target = np.asarray(target_train)

    test_no_target = np.asarray(getRidOfTarget(test_cat))
    test_target = np.asarray(target_test)

    result = getPredictions(train_target, train_no_target, test_no_target)
    #print(result)
    predictions = weightAnswer(result, target_test)
    #print(predictions)
    print(accuracy(target_test, predictions))

    #------------------------------------------------
