import numpy as np
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
import cPickle as pickle
import os
from random import sample

pairsFname = "data/pairs.txt"
embeddingsFname = "data/lfw_embedding.pkl"

def loadEmbeddings(embeddingsFname):
    return pickle.load(open(embeddingsFname, "rb"))
    
def main():
    embeddings = loadEmbeddings(embeddingsFname)
    pairs = loadPairs(pairsFname, embeddings)
    verifyExp("data", pairs, embeddings)
    #plotVerifyExp()
    #writeROC("data/roc.txt", np.arange(0,1,0.001), embeddings, pairs)
    
    
def loadPairs(pairsFname, embeddings):    
    cache_path = "data/lfw_pairs.pkl"
    if os.path.exists(cache_path):
        print "Reading pairs ..."
        return pickle.load(open(cache_path, "rb"))
    print "Filtering pairs ..."
    pairs = []
    with open(pairsFname) as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            try:           
                getEmbeddings(pair, embeddings)
                pairs.append(pair)
            except:
                pass
    
    num = len(pairs) 
    print num
    rand_sample = sample(pairs, 6000-num)
    pairs += rand_sample
    assert len(pairs) == 6000
    pairs = np.array(pairs)
    pickle.dump(pairs, open(cache_path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    return pairs


def getEmbeddings(pair, embeddings):
    if len(pair) == 3:
        name1 = "{}_{}".format(pair[0], pair[1].zfill(4))
        name2 = "{}_{}".format(pair[0], pair[2].zfill(4))
        actual_same = True
    elif len(pair) == 4:
        name1 = "{}_{}".format(pair[0], pair[1].zfill(4))
        name2 = "{}_{}".format(pair[2], pair[3].zfill(4))
        actual_same = False
    else:
        raise Exception(
            "Unexpected pair length: {}".format(len(pair)))

    (x1, x2) = (embeddings[name1], embeddings[name2])
    return (x1, x2, actual_same)


def writeROC(fname, thresholds, embeddings, pairsTest):
    with open(fname, "w") as f:
        f.write("threshold,tp,tn,fp,fn,tpr,fpr\n")
        tp = tn = fp = fn = 0
        for threshold in thresholds:
            tp = tn = fp = fn = 0      
            for pair in pairsTest:
                (x1, x2, actual_same) = getEmbeddings(pair, embeddings)
                dist = np.dot(x1.T, x2)
                predict_same  = dist > threshold
                
                if predict_same and actual_same:
                    tp += 1
                elif predict_same and not actual_same:
                    fp += 1
                elif not predict_same and not actual_same:
                    tn += 1
                elif not predict_same and actual_same:
                    fn += 1
                    
            if tp + fn == 0:
                tpr = 0
            else:
                tpr = float(tp) / float(tp + fn)
            if fp + tn == 0:
                fpr = 0
            else:
                fpr = float(fp) / float(fp + tn)
                
            f.write(",".join([str(x)
                for x in [threshold, tp, tn, fp, fn, tpr, fpr]]))
            f.write("\n")
            if tpr == 1.0 and fpr == 1.0:
                # No further improvements.
                f.write(",".join([str(x)
                                  for x in [4.0, tp, tn, fp, fn, tpr, fpr]]))
                return            
            

def getDistances(embeddings, pairsTrain):
    list_dist = []
    y_true = []
    for pair in pairsTrain:
        (x1, x2, actual_same) = getEmbeddings(pair, embeddings)
        dist = np.dot(x1.T, x2)
        list_dist.append(dist)
        y_true.append(actual_same)
    return np.asarray(list_dist), np.array(y_true)        



def evalThresholdAccuracy(embeddings, pairs, threshold):
    distances, y_true = getDistances(embeddings, pairs)
    y_predict = np.zeros(y_true.shape)
    y_predict[np.where(distances > threshold)] = 1

    y_true = np.array(y_true)
    accuracy = accuracy_score(y_true, y_predict)
    return accuracy, pairs[np.where(y_true != y_predict)]


def findBestThreshold(thresholds, embeddings, pairsTrain):
    bestThresh = bestThreshAcc = 0
    distances, y_true = getDistances(embeddings, pairsTrain)
    for threshold in thresholds:
        y_predlabels = np.zeros(y_true.shape)
        y_predlabels[np.where(distances > threshold)] = 1
        accuracy = accuracy_score(y_true, y_predlabels)
        if accuracy >= bestThreshAcc:
            bestThreshAcc = accuracy
            bestThresh = threshold
        else:
            # No further improvements.
            return bestThresh
    return bestThresh


def verifyExp(workDir, pairs, embeddings):
    print("  + Computing accuracy.")
    folds = KFold(n=6000, n_folds=10, shuffle=False)
    thresholds = np.arange(0, 1, 0.001)

    #if os.path.exists("{}/accuracies.txt".format(workDir)):
        #print("{}/accuracies.txt already exists. Skipping processing.".format(workDir))
    if 0:
        pass
    else:
        accuracies = []
        with open("{}/accuracies.txt".format(workDir), "w") as f:
            f.write('fold, threshold, accuracy\n')
            for idx, (train, test) in enumerate(folds):
                fname = "{}/l2-roc.fold-{}.csv".format(workDir, idx)
                writeROC(fname, thresholds, embeddings, pairs[test])

                bestThresh = findBestThreshold(
                    thresholds, embeddings, pairs[train])
                accuracy, pairs_bad = evalThresholdAccuracy(
                    embeddings, pairs[test], bestThresh)
                accuracies.append(accuracy)
                f.write('{}, {:0.2f}, {:0.2f}\n'.format(
                    idx, bestThresh, accuracy))
            avg = np.mean(accuracies)
            std = np.std(accuracies)
            f.write('\navg, {:0.4f} +/- {:0.4f}\n'.format(avg, std))
            print('    + {:0.4f}'.format(avg))

                
        
    
if __name__ == "__main__":
    main()