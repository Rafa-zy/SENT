import numpy as np 
from sklearn.metrics import roc_auc_score,roc_curve,precision_recall_fscore_support,accuracy_score

def opt_threshold(y_true, y_probs,type='acc',beta = 0.0625):
    if type == 'auc':
        auc = roc_auc_score(y_true, y_probs)
        fpr,tpr,thresholds = roc_curve(y_true,y_probs)
        threshold = thresholds[np.argmax(tpr-fpr)]
        target = auc
    else:
        A = list(zip(y_true, y_probs))
        A = sorted(A, key=lambda x: x[1])
        labels = np.array([t[0] for t in A])
        leng = len(A)
        F1s = []
        Accs = []
        Pres = []
        fx_scores = []
        def zero_division(n,d):
            return n / d if d else 0
        for i,(_,prob) in enumerate(A):
            thres = prob
            preds = np.array([0 for _ in range(i)] + [1 for _ in range(i,leng)])
            # beta=0.0625
            precision, recall, f1_score, support = precision_recall_fscore_support(labels, preds, average='binary')
            fx_score = (1+beta**2)*(precision*recall) / (beta**2*precision+recall)
            acc = accuracy_score(labels,preds)
            F1s.append((f1_score,thres))
            Accs.append((acc,thres))
            Pres.append((precision,thres))
            fx_scores.append((fx_score,thres))
        Accs = sorted(Accs, key=lambda x: x[0])
        F1s = sorted(F1s, key=lambda x: x[0])
        Pres = sorted(Pres, key=lambda x: (x[0],-x[1]))
        fx_scores = sorted(fx_scores, key=lambda x: x[0])
        if type == 'acc':
            acc,threshold= Accs[-1][0],Accs[-1][1]
            target = acc
        elif type == 'precision':
            pre,threshold =  Pres[-1][0],Pres[-1][1]
            target = pre
        elif type == 'f_x':
            fx_score,threshold =  fx_scores[-1][0],fx_scores[-1][1]
            target = fx_score
        else:
            assert type == 'f1'
            f1,threshold =  F1s[-1][0],F1s[-1][1]
            target = f1
        
    y_pred = [1 if t >= threshold else 0 for t in y_probs]
    precision, recall, f1_score, support = precision_recall_fscore_support(y_true, y_pred, average='binary')
    acc = accuracy_score(y_true,y_pred)
    return target,threshold,precision, recall, f1_score,acc,fx_score,beta

# if __name__ == "__main__":
#     y_true = [1,   0,   1,   0,  1,  1,0,0,1,1,0]
#     y_pred = [0.62,0.3,0.8,0.6,0.7,0.9,0.3,0.5,0.64,0.1,0.8]
#     target,threshold,precision, recall, f1_score,acc = opt_threshold(y_true,y_pred,'f1')
#     print(1)