from sklearn.metrics import make_scorer
import pickle

def score(y_true, y_pred, show):
    cost_matrix = [[0, 1, 2, 2, 2],
                   [1, 0, 2, 2, 2],
                   [2, 1, 0, 2, 2],
                   [3, 2, 2, 0, 2],
                   [4, 2, 2, 2, 0]
                   ]
    cost = 0
    size = y_true.size
    # print count & percentage matrix
    if show:
        count = [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 'x']
                 ]
        for i in range(size):
            cost += cost_matrix[y_true[i]][y_pred[i]]  # y_ture should be dataframe.values
            count[y_true[i]][y_pred[i]] += 1
        
        for i in range(5):
            if 0 == sum(count[i]):
                count[i][5] = 0
            else:
                count[i][5] = float(count[i][i])/ sum(count[i])
    
        for i in range(5):
            if 0 == sum(count[j][i] for j in range(5)):
                count[5][i] = 0
            else:
                count[5][i] = float(count[i][i]) / sum(count[j][i] for j in range(5))
        print("-----------------------------------------------")
        for i in range(5):
            print("%7d %7d %7d %7d %7d %1.5f" % (
                count[i][0], count[i][1], count[i][2], count[i][3], count[i][4], count[i][5]))
        print("%1.5f %1.5f %1.5f %1.5f %1.5f" % (
            count[5][0], count[5][1], count[5][2], count[5][3], count[5][4]))
        print("score: ", float(cost) / size)
        print("-----------------------------------------------")
    else:
        for i in range(size):
            cost += cost_matrix[y_true[i]][y_pred[i]]
    return float(cost) / size




def scorer(show):
    return make_scorer(score, show=show, greater_is_better=False)
