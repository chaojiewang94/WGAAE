import numpy as np

from scipy.optimize import linear_sum_assignment as linear_assignment
import tensorflow as tf

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

def Cosine(Theta_1, Theta_2):
    # Theta_1 N*K
    # Theta_2 N*K

    Theta_1_norm = Theta_1 / np.sqrt(np.sum(Theta_1 * Theta_1, axis=1, keepdims=True))
    Theta_2_norm = Theta_2 / np.sqrt(np.sum(Theta_2 * Theta_2, axis=1, keepdims=True))

    return np.matmul(Theta_1_norm, Theta_2_norm.T)


def MutualInfo(L11, L22):

    # L11 is the groudtrue : (nsamples,)
    # L22 is the pre_label : (nsamples,)
    L1 = L11.copy()
    L2 = L22.copy()
    n_gnd = L1.shape[0]
    n_label = L2.shape[0]
    # assert n_gnd == n_label

    Label = np.unique(L1)
    nClass = len(Label)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    if nClass2 < nClass:
        L1 = np.concatenate((L1, Label))
        L2 = np.concatenate((L2, Label))
    else:
        L1 = np.concatenate((L1, Label2))
        L2 = np.concatenate((L2, Label2))

    G = np.zeros([nClass, nClass])
    for i in range(nClass):
        for j in range(nClass):
            G[i, j] = np.sum((L1 == Label[i]) * (L2 == Label[j]))

    sum_G = np.sum(G)
    P1 = np.sum(G, axis=1)
    P1 = P1/sum_G
    P2 = np.sum(G, 0)
    P2 = P2/sum_G
    if np.sum((P1 == 0)) > 0 or np.sum((P2 == 0)):
        print('error ! Smooth fail !')
    else:
        H1 = np.sum(-P1 * np.log2(P1))
        H2 = np.sum(-P2*np.log2(P2))
        P12 = G/sum_G
        PPP = P12 / np.tile(P2, (nClass,1) ) / np.tile(P1.reshape(-1, 1), (1, nClass))
        PPP[np.where(abs(PPP) < 1E-12)] = 1
        MI = np.sum(P12 * np.log2(PPP))
        MIhat = MI / np.max((H1, H2))

        return MIhat


def Accuracy(y, ypred):
    """
    :param ypred: pred_label, shape:(n_sample,)
    :param y: the ground_true,
    :return: accuracy of cluster
    """
    # print(len(np.unique(ypred)), len(np.unique(y)))
    # assert len(y) > 0
    # assert len(np.unique(ypred)) == len(np.unique(y))

    s = np.unique(ypred)
    t = np.unique(y)

    N = len(np.unique(ypred))
    C = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            idx = np.logical_and(ypred == s[i], y == t[j])
            C[i][j] = np.count_nonzero(idx)

    # convert the C matrix to the 'true' cost
    Cmax = np.amax(C)
    C = Cmax - C
    indices = np.transpose(np.array(linear_assignment(C)))
    row = indices[:][:, 0]
    col = indices[:][:, 1]
    # calculating the accuracy according to the optimal assignment
    count = 0
    for i in range(N):
        idx = np.logical_and(ypred == s[row[i]], y == t[col[i]])
        count += np.count_nonzero(idx)

    return 1.0 * count / len(y)

def get_acc(edges_pos, edges_neg, emb=None):
    # if emb is None:
    #     feed_dict.update({placeholders['dropout']: 0})
    #     emb = sess.run(model.z_decoder_a, feed_dict=feed_dict)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def beta(x):
        return 1 - np.exp(-x)

    adj_rec = np.dot(emb, emb.T)
    acc_num = 0
    for e in edges_pos:
        # temp = sigmoid(adj_rec[e[0], e[1]])
        temp = beta(adj_rec[e[0], e[1]])
        # temp = adj_rec[e[0], e[1]]
        if temp > 0.5:
            acc_num += 1
        # preds.append(adj_rec[e[0], e[1]])
        # pos.append(adj_orig[e[0], e[1]])
    for e in edges_neg:
        # temp = sigmoid(adj_rec[e[0], e[1]])
        # temp = adj_rec[e[0], e[1]]
        temp = beta(adj_rec[e[0], e[1]])
        if temp < 0.5:
            acc_num += 1
    return acc_num/(len(edges_neg)+len(edges_pos))

def get_roc_score(edges_pos, edges_neg, adj_orig, emb=None):
    # if emb is None:
    #     feed_dict.update({placeholders['dropout']: 0})
    #     emb = sess.run(model.z_decoder_a, feed_dict=feed_dict)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def beta(x):
        return 1 - np.exp(-x)

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    # print(adj_rec,'**************')
    for e in edges_pos:
        # preds.append(sigmoid(adj_rec[e[0], e[1]]))
        # preds.append(adj_rec[e[0], e[1]])
        preds.append(beta(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        # preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        # preds_neg.append(adj_rec[e[0], e[1]])
        preds_neg.append(beta(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score