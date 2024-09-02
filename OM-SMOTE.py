

import numpy as np
import copy

import torch
from imblearn.metrics import geometric_mean_score
from imblearn.under_sampling import EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN,NeighbourhoodCleaningRule,CondensedNearestNeighbour
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize,MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from torch import nn, tensor
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
import smote_variants as sv
from gsmote import GeometricSMOTE
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

def one_hot(y, M):

    en = np.zeros([y.shape[0], M])
    for i in range(y.shape[0]):
        en[i, int(y[i])] = en[i, int(y[i])] + 1
    return en

def om-smote(name,algorithm,mul,a,normalize = True):
    repeat = 1
    # a = 0.2
    output = False
    # mul = 5
    Epoch = 10000
    LR = 0.003

    class AutoEncoder(nn.Module):
        def __init__(self):
            super(AutoEncoder, self).__init__()

            self.encoder = nn.Sequential(
                nn.Linear(dim, h_dim, bias=True),
                nn.Tanh()
            )

            self.decoder = nn.Sequential(
                nn.Linear(h_dim, dim, bias=True),
                nn.Sigmoid()
            )
            self.out = nn.Sequential(
                nn.Linear(h_dim , 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            out = self.out(encoded)
            return encoded, decoded, out

    data_name = '../DataSet/' + name + '/' + name + '.npy'
    data = np.load(data_name)
    X = data[:, :-1]
    y = data[:, -1]

    if normalize:

        minmax = MinMaxScaler()
        X = minmax.fit_transform(X)


    accuracy1 = []
    g_mean1 = []
    f11 = []
    auc1 = []

    accuracy2 = []
    g_mean2 = []
    f12 = []
    auc2 = []

    accuracy3 = []
    g_mean3 = []
    f13 = []
    auc3 = []

    accuracy4 = []
    g_mean4 = []
    f14 = []
    auc4 = []

    accuracy5 = []
    g_mean5 = []
    f15 = []
    auc5 = []

    case1 = 0
    case2 = 0
    case3 = 0
    case4 = 0



    for i in range(repeat):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y)


        # split min and maj
        minority = []
        majority = []
        for x in range(X_train.shape[0]):
            if y_train[x] == 1.0:
                minority.append(X_train[x])
            else:
                majority.append(X_train[x])
        majority = np.array(majority)
        minority = np.array(minority)
        new_dataset = np.concatenate((minority, majority), axis=0)
        new_dataset = torch.tensor(new_dataset)


        y0 = np.concatenate((np.ones([minority.shape[0], 1]), np.zeros([majority.shape[0], 1])), axis=0)
        y0 = torch.tensor(y0)


        dim = new_dataset.shape[1]
        h_dim = dim * mul
        autoencoder = AutoEncoder()
        optim = torch.optim.Adam(autoencoder.parameters(), lr=LR)
        loss1 = nn.MSELoss(reduction='sum')
        loss2 = nn.BCELoss()



        for epoch in range(Epoch):
            encoded, decoded, out = autoencoder(new_dataset)
            l1 = loss1(decoded, new_dataset)
            l2 = loss2(out.float(), y0.float())
            loss = a * l1 +(1-a)*l2
            optim.zero_grad()
            loss.backward()
            optim.step()
        print(" loss:{}+{}".format(l1,l2))

        w = autoencoder.decoder[0].weight
        b = autoencoder.decoder[0].bias
        w = w.t()



        X1 = encoded.detach().numpy()
        hidden_min = X1[0:minority.shape[0], :]
        hidden_maj = X1[-(majority.shape[0]):, :]

        min_center = np.reshape(np.mean(hidden_min, axis=0), [1, hidden_min.shape[1]])
        maj_center = np.reshape(np.mean(hidden_maj, axis=0), [1, hidden_maj.shape[1]])
        min_raduis = np.max(np.abs(hidden_min-min_center))
        maj_raduis = np.max(np.abs(hidden_maj-maj_center))


        min_far = []
        count = np.zeros(hidden_min.shape[0])
        minority_copy = copy.deepcopy(hidden_min)
        # minority_copy = minority
        for i in range(hidden_min.shape[0]):
            max = 0
            for j in range(hidden_min.shape[0]):
                if count[j] >= 2:
                    minority_copy[j, :] = min_center

                dis = np.sqrt(np.sum((minority_copy[i, :] - minority_copy[j, :]) ** 2))
                if dis >= max:
                    max = dis
                    idx = j
            min_far.append(idx)
            count[idx] += 1


        sny_minority = []
        m_c = min_center.tolist()[0]
        for i in range(hidden_maj.shape[0] - hidden_min.shape[0]):
            j = np.random.randint(0, hidden_min.shape[0])
            dis0 = np.sqrt(np.sum((hidden_min[j, :] - maj_center) ** 2))
            dis1 = np.sqrt(np.sum((hidden_min[min_far[j], :] - maj_center) ** 2))
            if (dis0 >= maj_raduis) & (dis1 >= maj_raduis):
                case1 +=1
                new_data = hidden_min[j, :] + np.random.random() * (hidden_min[min_far[j], :] - hidden_min[j, :])
                sny_minority.append(new_data)

            elif (dis0 >= maj_raduis) & (dis1 < maj_raduis):

                case2 += 1
                mid = (hidden_min[min_far[j], :] + hidden_min[j, :]) / 2
                dis = np.sqrt(np.sum((mid - maj_center) ** 2))
                while (np.abs(dis - maj_raduis) <= 0.0001):
                    if dis > maj_raduis:

                        mid = (mid + hidden_min[min_far[j], :]) / 2
                        dis = np.sqrt(np.sum((mid - maj_center) ** 2))
                    else:
                        mid = (mid + hidden_min[j, :]) / 2
                        dis = np.sqrt(np.sum((mid - maj_center) ** 2))
                new_data = hidden_min[j, :] + np.random.random() * (mid - hidden_min[j, :])
                sny_minority.append(new_data)
            elif (dis0 < maj_raduis) & (dis1 >= maj_raduis):

                case3 +=1
                mid = (hidden_min[min_far[j], :] + hidden_min[j, :]) / 2
                dis = np.sqrt(np.sum((mid - maj_center) ** 2))
                while (np.abs(dis - maj_raduis) <= 0.0005):
                    if dis > maj_raduis:

                        mid = (mid + hidden_min[min_far[j], :]) / 2
                        dis = np.sqrt(np.sum((mid - maj_center) ** 2))
                    else:
                        mid = (mid + hidden_min[j, :]) / 2
                        dis = np.sqrt(np.sum((mid - maj_center) ** 2))
                new_data = mid + np.random.random() * (mid - hidden_min[min_far[j], :])
                sny_minority.append(new_data)
            else:
                case4 += 1


                new_data = hidden_min[j, :] + np.random.random() * (hidden_min[j, :] - m_c)
                sny_minority.append(new_data)


        sny_minority = np.array(sny_minority)
        new_min = np.concatenate((hidden_min, sny_minority), axis=0)

        new_min = torch.tensor(new_min,dtype=torch.float32)
        sny = torch.mm(new_min, w) + b
        sigmoid = nn.Sigmoid()
        sny = sigmoid(sny)
        sny = sny.detach().numpy()
        new_minority = sny




        new_X = np.concatenate((new_minority, majority), axis=0)
        new_y = np.concatenate((np.ones([new_minority.shape[0], 1]),
                                np.zeros([majority.shape[0], 1])), axis=0)



        if algorithm == 'bayes':
            clf1 = GaussianNB()

        elif algorithm == 'svm':
            clf1 = svm.SVC(C=100, gamma='auto')


        elif algorithm == 'DT':
            new_X = np.around(new_X, 3)
            clf1 = DecisionTreeClassifier(criterion='gini')

        elif algorithm == 'KNN':
            clf1 = KNeighborsClassifier()

        elif algorithm == 'RF':
            new_X = np.around(new_X, 3)
            clf1 = RandomForestClassifier()

        elif algorithm == 'LR':
            clf1 = LogisticRegression()

        elif algorithm == 'GBC':
            new_X = np.around(new_X, 3)
            clf1 = GradientBoostingClassifier()


        clf1.fit(new_X, new_y.ravel())
        test_predict1 = clf1.predict(X_test)
        accuracy1.append(accuracy_score(y_test.astype('int'), test_predict1))
        g_mean1.append(geometric_mean_score(y_test.astype('int'), test_predict1, average='binary'))
        f11.append(f1_score(y_test.astype('int'), test_predict1))
        auc1.append(roc_auc_score(y_test.astype('int'), test_predict1))


    print('mo-smote:-----------------------------')
    print("case1:{}; case2:{}; case3:{}; case4:{}".format(case1, case2, case3, case4))
    print('a：' ,a)
    print('auc:' + str(np.mean(auc1)) + '+' + str(np.std(auc1)))
    print('G-mean：' + str(np.mean(g_mean1)) + '+' + str(np.std(g_mean1)))
    print('f1-score：' + str(np.mean(f11)) + '+' + str(np.std(f11)))
    print('accuracy：' + str(np.mean(accuracy1)) + '+' + str(np.std(accuracy1)))


    if output:
        with open('../moResults/' + algorithm + '/' + name + '_result.txt', 'a', encoding='utf-8') as out:
            out.write('{} {} {}  {}  {} {}{} {}{} {} {}\n\t{} {} {} {}\n\t {} {} {} {} {} {} {} {}\n\t{}\n\t{}\n\t{}\n\t{}\n\n'.format("mo-smote",'重复：',repeat,"升维倍数：",mul,'损失函数权重',a,
                                                                        'Epoch:',Epoch,'LR:',LR,
                                                                        'loss:',l1,'+',l2,"case1",case1,'case2',case2,'case3',case3,'case4',case4,
                                                                        'accuracy：' + str(
                                                                            round(np.mean(accuracy1), 3)) + '±' + str(
                                                                            round(np.std(accuracy1), 3)),
                                                                        'G-mean：' + str(
                                                                            round(np.mean(g_mean1), 3)) + '±' + str(
                                                                            round(np.std(g_mean1), 3)),
                                                                        'f1-score：' + str(
                                                                            round(np.mean(f11), 3)) + '±' + str(
                                                                            round(np.std(f11), 3)),
                                                                        'auc:' + str(
                                                                            round(np.mean(auc1), 3)) + '±' + str(
                                                                            round(np.std(auc1), 3))))


if __name__ == '__main__':
    dataSet = ['ecoli1', 'ecoli2', 'ecoli3', 'ecoli4', 'yeast1', 'yeast3', 'yeast5', 'yeast-0-5-6-7-9_vs_4',
            'yeast-1-2-8-9_vs_7', 'yeast-1_vs_7', 'yeast-2_vs_8', 'glass0', 'glass2', 'glass4', 'glass6',
            'glass-0-1-6_vs_2', 'glass-0-1-6_vs_5', 'vowel0', 'new-thyroid1', 'shuttle-C0_vs_C4', 'segment0',
            'vehicle0', 'page-blocks0', 'page-blocks-1-3_vs_4', 'abalone9_18', 'abalone19', 'wisconsin', 'haberman', 'pima']
  

for name in dataSet:
    for mul in range(2,9):
        for a in np.arange(0.1,1,0.1):
            om-smote(name,'svm',mul,a, normalize = True) #
            om-smote(name, 'DT',mul,a, normalize=True) #
            om-smote(name,'RF',mul,a, normalize=True)
            om-smote(name, 'GBC',mul,a,  normalize=True) #
            om-smote(name, 'bayes',mul,a,  normalize=True) #
            om-smote(name, 'KNN',mul,a,  normalize=True)
            om-smote(name, 'LR',mul,a,  normalize=True)
