import warnings
warnings.filterwarnings("ignore")

import numpy as np
import scipy
import scipy.sparse as sp
import scipy.io as sio
import tensorflow as tf
import _pickle as cPickle
import pickle
from util import *
import time
from Time import Timer
import matplotlib.pyplot as plt
import h5py
import argparse
from metric import *
from sklearn.cluster import k_means
from input_data import load_data_class
from preprocessing import mask_test_edges, preprocess_graph, sparse_to_tuple
from GAT.gat_hwithBAM_classification_cora import *
from GAT.utils import process
from sampling import get_distribution, node_sampling, preprocess_graph4sample, construct_feed_dict
np.random.seed(123)
tf.set_random_seed(123)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora', help='pubmed, cora, citeseer')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--id', type=str, default='default')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--task', type=str, default='classfication', help='if cluster task') # cluster, prediction
parser.add_argument('--sub_sample', type=bool, default=True, help='Whether sample sub-graph')
parser.add_argument('--sub_update', type=bool, default=False, help='whether update sub-Phi')
parser.add_argument('--nb_node_sample', type=int, default=1000, help='number of sample nodes')
parser.add_argument('--k', type=float, default=1.0, help='sample distribution')

# Variational attention
parser.add_argument('--att_type', type=str, default='soft_weibull',
                    help='soft_attention, soft_weibull, soft_lognormal')
parser.add_argument('--k_weibull', type=float, default=1000.0,
                    help='initialization of k in weibull distribution.')
parser.add_argument('--att_kl', type=float, default=1.0,
                    help='weights for KL term in variational attention.')
parser.add_argument('--kl_anneal_rate', type=float, default=1.0,
                    help='KL anneal rate.')
parser.add_argument('--att_prior_type', type=str, default='constant',
                    help='contextual, constant, parameter, which type of prior used in variational attention.')
parser.add_argument('--alpha_gamma', type=float, default=1.0,
                    help='initialization of alpha in gamma distribution.')
parser.add_argument('--beta_gamma', type=float, default=1.0,
                    help='initialization of beta in gamma distribution.')
parser.add_argument('--sigma_normal_prior', type=float, default=1.0,
                    help='initialization of sigma in prior normal distribution.')
parser.add_argument('--sigma_normal_posterior', type=float, default=1.0,
                    help='initialization of sigma in posterior normal distribution.')
parser.add_argument('--att_contextual_se', type=int, default=0,
                        help='whether to use squeeze and excite in prior computation.')
parser.add_argument('--att_se_hid_size', type=int, default=10,
                    help='squeeze and excite factor in attention prior.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='dropout.')
parser.add_argument('--l2_coef', type=float, default=5e-4,
                    help='l2_coef.')

args = parser.parse_args()


##====================== Load Data ======================##
# load data
if args.dataset == '20ng':

    train_data = sp.load_npz('./data_sparse.npz')
    train_data = train_data.toarray()
    train_data = np.array(train_data.T, order='C')  # V*N

    train_label = np.load('./20ng_label.npy')
    train_label = train_label.reshape(len(train_label), 1)
    train_class = np.unique(train_label)

elif args.dataset == 'mnist':

    data = sio.loadmat('./mnist_gray')
    train_data = np.array(np.ceil(data['train_mnist']*25), order='C')[:, 0:1000]  # 0-1    V*N
    test_data = np.array(np.ceil(data['test_mnist']*25), order='C')[:, 0:1000]  # 0-1    V*N
    train_label = data['train_label']
    test_label = data['test_label']

elif args.dataset == 'coil':

    data = sio.loadmat('./COIL20_GRAPH.mat')
    train_data = np.array(np.transpose(np.ceil(data['fea']*25)), order='C')  # V*N We use the feature after L2 norm
    train_label = data['gnd']
    train_graph = data['A']  # diagonal is zeros
    train_n_class = np.unique(train_label).size

elif args.dataset == 'pie':

    data = sio.loadmat('./PIE_pose27_GRAPH.mat')
    train_data = np.array(np.transpose(np.ceil(data['fea'])), order='C')  # V*N We use the orginal feature
    train_data = np.ceil(train_data)
    train_label = data['gnd']
    train_graph = data['A']
    train_n_class = np.unique(train_label).size

else:
    args.task = 'prediction'
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_class(args.dataset)
    # train_data = features.todense().T  ## V * N
    train_data = preprocess_features(features).T
    train_graph = adj.toarray()
    train_label = None
    train_n_class = y_train.shape[1]
    print('class : ', train_n_class)
    # if args.dataset == ''

##====================== Graph Preprocess ======================##
# preprocess for graph
gat_adj = adj
adj = sp.csc_matrix(train_graph)
num_nodes = adj.shape[0]

# for metric
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

if args.task == 'prediction':
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train


# for graph_LH
adj_label = adj + sp.eye(adj.shape[0])
adj_label = sparse_to_tuple(adj_label)




##====================== Setting for PGBN ======================##
# setting
Setting = Empty()
Setting.V = train_data.shape[0]
Setting.N = train_data.shape[1]
Setting.K = [512]  # len(np.unique(train_label))
Setting.T = len(Setting.K)

# online setting
Setting.SweepTimes = args.epochs
Setting.Minibatch = Setting.N
Setting.Burnin = 5
Setting.Collection = 5
Setting.Iterall = Setting.SweepTimes
Setting.tao0FR = 0
Setting.kappa0FR = 0.9
Setting.tao0 = 20
Setting.kappa0 = 0.7
Setting.epsi0 = 1
Setting.FurCollapse = 1  # 1 or 0
Setting.Flag = 0

# superparams
Supara = Empty()
Supara.eta = np.ones(Setting.T)*0.1  # 0.01

# params
Params = Empty()
Params.Phi = []
Params.Eta = []
Params.Xt_to_t1 = [0]*Setting.T
Params.WSZS = [0]*Setting.T
Params.NDot = [0]*Setting.T
Params.EWSZS = [0]*Setting.T
Params.ForgetRate = np.power((Setting.tao0FR + np.linspace(1, Setting.Iterall, Setting.Iterall)), - Setting.kappa0FR)
Params.epsit = np.power((Setting.tao0 + np.linspace(1, Setting.Iterall, Setting.Iterall)), -Setting.kappa0)
Params.epsit = Setting.epsi0 * Params.epsit / Params.epsit[0]
for t in range(Setting.T):
    Params.Eta.append(Supara.eta[t])
    if t == 0:
        Params.Phi.append(0.2 + 0.8 * np.float64(np.random.rand(Setting.V, Setting.K[t])))
    else:
        Params.Phi.append(0.2 + 0.8 * np.float64(np.random.rand(Setting.K[t - 1], Setting.K[t])))
    # Params.Phi[t] = Params.Phi[t] / np.maximum(1e-30, Params.Phi[t].sum(0))
Params.r_k = np.ones([Setting.K[Setting.T-1], 1]) / Setting.K[Setting.T-1]


##====================== Setting for VAE ======================##
# define layer
def log_max(input_x):
    return tf.log(tf.maximum(input_x, real_min))


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01, dtype=tf.float32))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape, dtype=tf.float32))


def encoder_right_sparse(input_x, i):

    # params
    K_dim = Setting.K

    k_tmp = tf.layers.dense(input_x, K_dim[i])
    l_tmp = tf.layers.dense(input_x, K_dim[i])

    k_tmp = tf.nn.softplus(k_tmp)
    l_tmp = tf.nn.softplus(l_tmp)

    k_tmp = tf.minimum(k_tmp, 1.0/real_min)
    k = tf.maximum(k_tmp, 0.1)

    l_tmp = tf.maximum(l_tmp, real_min)
    l = l_tmp / tf.exp(tf.lgamma(1 + 1.0 / k))

    return tf.transpose(k), tf.transpose(l)


def InnerProductDecoder(x, dropout):
    x = tf.nn.dropout(x, 1-dropout)
    x_t = tf.transpose(x)
    x = tf.matmul(x, x_t)
    out = tf.reshape(x, [-1])
    return out


def reparameterization(Wei_shape, Wei_scale, i, batch_size):
    K_dim = Setting.K

    sample_num = 50
    eps = tf.random_uniform(shape=[sample_num, np.int32(K_dim[i]), batch_size], dtype=tf.float32)
    Wei_shape = tf.tile(tf.expand_dims(Wei_shape, 0), [sample_num, 1, 1])
    Wei_scale = tf.tile(tf.expand_dims(Wei_scale, 0), [sample_num, 1, 1])
    theta = Wei_scale * tf.pow(-log_max(1 - eps), 1.0 / Wei_shape)
    theta = tf.reduce_mean(theta, axis=0, keep_dims=False)

    # eps = tf.random_uniform(shape=[np.int32(K_dim[i]), batch_size], dtype=tf.float32)    # K_dim[i] * none
    # theta = Wei_scale * tf.pow(-log_max(1-eps), 1/Wei_shape)
    return theta                                                                         # K_dim[i] * none

def KL_GamWei(Gam_shape, Gam_scale, Wei_shape, Wei_scale):  # K_dim[i] * none

    eulergamma = 0.5772
    KL_Part1 = eulergamma * (1 - 1/Wei_shape) + log_max(Wei_scale/Wei_shape) + 1 + Gam_shape * log_max(Gam_scale)
    KL_Part2 = -tf.lgamma(Gam_shape) + (Gam_shape - 1) * (log_max(Wei_scale) - eulergamma/Wei_shape)
    KL = KL_Part1 + KL_Part2 - Gam_scale * Wei_scale * tf.exp(tf.lgamma(1 + 1/Wei_shape))
    return KL


def bern_possion_link(x):
        return 1.0 - tf.exp(-x)


def sample_subgraph(sp_adj, nb_node_samples, replace):  # gat_adj
    ######## Sampling Subgraph Nodes ##########
    # Node-level p_i degree-based, core-based or uniform distribution
    measure = 'degree'
    alpha = 2.0
    k = args.k
    print('sample nodes = ', nb_node_samples)

    # node distribution
    node_distribution = get_distribution(measure, alpha, sp_adj)

    # Node sampling
    sampled_nodes, sub_adj_label, adj_sampled_sparse = node_sampling(sp_adj,
                                                                 node_distribution,
                                                                 nb_node_samples,
                                                                 k,
                                                                 replace)
    placeholders = {
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'sampled_nodes': tf.placeholder_with_default(sampled_nodes, shape=[nb_node_samples])
    }
    sub_adj_ori = tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False), [-1])

    sub_adj_norm = preprocess_graph4sample(sp_adj)
    return [sub_adj_norm, sub_adj_label, sub_adj_ori, sampled_nodes, adj_sampled_sparse, placeholders, node_distribution]


def get_graphA_LH(adj_ori, sampled_nodes, adj_sampled_sparse, Theta_concat):
    dropout = 0.0
    num_sampled = adj_sampled_sparse.shape[0]
    sum_sampled = adj_sampled_sparse.sum()
    pos_weight = float(num_sampled * num_sampled - sum_sampled) / sum_sampled
    norm = num_sampled * num_sampled / float((num_sampled * num_sampled - sum_sampled) * 2)
    sub_rec = tf.gather(Theta_concat, sampled_nodes)
    inner_product = InnerProductDecoder(sub_rec, dropout)
    graph_re = bern_possion_link(inner_product)
    print('shape : ', sub_rec.shape)
    graphA_LH = norm * tf.reduce_sum(
        adj_ori * log_max(graph_re) * pos_weight + (1.0 - adj_ori) * log_max(1.0 - graph_re))

    return graphA_LH  # minus


##====================== Build Graph ======================##
Batch_Size = tf.placeholder(tf.int32)
X_NV = tf.placeholder(tf.float32, shape=[Setting.N, Setting.V])  # N*V
X_VN = tf.transpose(X_NV)  # V*N
A_NN = tf.placeholder(tf.float32, shape=[Setting.N, Setting.N])  # origin
D_NN = tf.matrix_diag(tf.reduce_sum(A_NN, axis=1))
L_NN = D_NN - A_NN

Y_labels = tf.placeholder(tf.float32, shape=(None, y_train.shape[1]))
Y_labels_mask = tf.placeholder(tf.int32)
attn_drop = tf.placeholder(dtype=tf.float32, shape=())
ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
is_train = tf.placeholder(dtype=tf.float32, shape=())

Phi_1 = tf.placeholder(tf.float32, shape=[Setting.V, Setting.K[0]])
dropout = tf.placeholder_with_default(0., shape=())
# graph_input = tf.sparse_placeholder(tf.float32)  # for gcn
graph_label = tf.sparse_placeholder(tf.float32)  # for loss
graph_label_full = tf.reshape(tf.sparse_tensor_to_dense(graph_label, validate_indices=False), [-1])

GAT = SpGAT()
GAT.GAT_init(X_NV, train_n_class)
bias = process.preprocess_adj_bias(gat_adj)
bias_in = tf.sparse_placeholder(dtype=tf.float32)

# layer
input = tf.expand_dims(X_NV, 0)
h_1 = GAT.inference(input, GAT.nb_classes, GAT.nb_nodes,
                                 bias_mat=bias_in,
                                 attn_drop=attn_drop,
                                 ffd_drop=ffd_drop,
                                 hid_units=GAT.hid_units, n_heads=GAT.n_heads,
                                 residual=GAT.residual, training=is_train, args=args
                    )

h_1 = tf.squeeze(h_1, [0])
print('h shape: ',h_1.shape)

k_1, l_1 = encoder_right_sparse(h_1, 0)
Theta_1 = reparameterization(k_1, l_1, 0, Setting.Minibatch)  # K*N
# res = tf.transpose(Theta_1)

res = tf.layers.dense(tf.concat([h_1, tf.transpose(Theta_1)], axis=1), train_n_class)

Label_loss = masked_softmax_cross_entropy(res, Y_labels, Y_labels_mask)
Label_acc = masked_accuracy(res, Y_labels, Y_labels_mask)
regularizer = tf.contrib.layers.l2_regularizer(1e-2)
# reg_term = tf.contrib.layers.apply_regularization(regularizer, [w_k, w_l])

data_re = tf.matmul(Phi_1, Theta_1)
lambda_re = weight_variable(shape=[1, 1])
Theta_concat = lambda_re * Theta_1
if args.sub_sample:
    nb_node_samples = args.nb_node_sample
    replace = True
    eta = 0.000002 # depends on nb_node_samples
    outs = sample_subgraph(gat_adj, nb_node_samples, replace)
    [sub_adj_norm, sub_adj_label, sub_adj_ori, sampled_nodes, adj_sampled_sparse, placeholders, node_distribution] = outs
    graph_LH = eta * get_graphA_LH(sub_adj_ori, sampled_nodes, adj_sampled_sparse, tf.transpose(Theta_concat))

else:
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    inner_product = InnerProductDecoder(tf.transpose(Theta_concat), dropout)
    graph_re = bern_possion_link(inner_product)  # (N*N) * 1
    eta = 0.00000004
    graph_LH = eta * norm * tf.reduce_sum(
        graph_label_full * log_max(graph_re) * pos_weight + (1 - graph_label_full) * log_max(1 - graph_re))

# optimizer

Theta_1_KL = tf.reduce_mean(KL_GamWei(np.float32(1.0), np.float32(1.0), k_1, l_1))  # / tf.cast(Batch_Size, tf.float32)
data_LH = 0.0000002 * tf.reduce_sum(X_VN*log_max(tf.matmul(Phi_1, Theta_1)) - tf.matmul(Phi_1, Theta_1) - tf.lgamma(X_VN + 1))  # / tf.cast(Batch_Size, tf.float32)
lap_loss = tf.trace(tf.matmul(tf.matmul(Theta_1, L_NN), tf.transpose(Theta_1)))

loss = -1 * Label_loss + 1 * data_LH + 1 * graph_LH #- 1 * reg_term  # + 0 * Theta_1_KL - 0 * lap_loss
LowerBound = Label_loss
if args.att_type != 'soft_attention':
    KL_loss = tf.add_n(tf.get_collection('kl_list')) / len(tf.get_collection('kl_list'))
    # KL_loss = tf.exp(Setting.SweepTimes * args.kl_anneal_rate) / (1 + tf.exp(Setting.SweepTimes * args.kl_anneal_rate)) * KL_loss
    Loss = loss - KL_loss * args.att_kl
else:
    KL_loss = tf.constant(0.0)
    Loss = loss
# prediction metric


Optimizer = tf.train.AdamOptimizer(0.001)
threshold = 0.01
grads_vars = Optimizer.compute_gradients(-Loss)
capped_gvs = []
for grad, var in grads_vars:

    if grad is not None:
        grad = tf.where(tf.is_nan(grad), threshold * tf.ones_like(grad), grad)
        grad = tf.where(tf.is_inf(grad), threshold * tf.ones_like(grad), grad)
        capped_gvs.append((tf.clip_by_value(grad, -threshold, threshold), var))
Opt_op = Optimizer.apply_gradients(capped_gvs)

tf.set_random_seed(seed=0)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

##====================== Train Phase ======================##

import PGBN_sampler

import os
dataset_path = './' + args.dataset
if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

file_path = './' + args.dataset + '/' + str(Setting.T) + '_layer/'
if not os.path.exists(file_path):
    os.mkdir(file_path)


Best_MI = 0.0
Best_Ac = 0.0
Best_AUC = 0.0
Best_AP = 0.0
Best_Label_Ac = 0.0
def updatePhi(miniBatch, Phi, Theta, MBratio, MBObserved):
    Xt = miniBatch  # V*N
    real_min_phi = 1e-30
    for t in range(len(Phi)):
        if t == 0:
            Params.Xt_to_t1[t], Params.WSZS[t] = PGBN_sampler.Multrnd_Matrix(Xt.astype('double'), Phi[t], Theta[t])
        else:
            Params.Xt_to_t1[t], Params.WSZS[t] = PGBN_sampler.Crt_Multirnd_Matrix(Params.Xt_to_t1[t - 1], Phi[t], Theta[t])
        Params.EWSZS[t] = MBratio * Params.WSZS[t]

        if (MBObserved == 0):
            Params.NDot[t] = Params.EWSZS[t].sum(0)
        else:
            Params.NDot[t] = (1 - Params.ForgetRate[MBObserved]) * Params.NDot[t] + Params.ForgetRate[MBObserved] * Params.EWSZS[t].sum(0)

        tmp = Params.EWSZS[t] + 0.1
        tmp = (1 / (np.maximum(Params.NDot[t], real_min_phi))) * (tmp - tmp.sum(0) * Phi[t])
        tmp1 = (2 / (np.maximum(Params.NDot[t], real_min_phi))) * Phi[t]
        tmp = Phi[t] + Params.epsit[MBObserved] * tmp + np.sqrt(Params.epsit[MBObserved] * tmp1) * np.random.randn(Phi[t].shape[0],
                                                                                                 Phi[t].shape[1])
        # tmp = tmp / (tmp.sum(0) + real_min)
        Phi[t] = PGBN_sampler.ProjSimplexSpecial(tmp, Phi[t], 0)
        Phi[t] = Phi[t]/np.maximum(real_min, Phi[t].sum(0))

    return Phi

def get_feed_dict(sampled_nodes):
    k = args.k
    graph_feed_dict = construct_feed_dict(sub_adj_norm, sub_adj_label, placeholders)
    # Update sampled subgraph
    graph_feed_dict.update({placeholders['sampled_nodes']: sampled_nodes})
    # New node sampling
    sampled_nodes, adj_label, _ = node_sampling(gat_adj, node_distribution, nb_node_samples,
                                                k, replace)

    return graph_feed_dict

val = []
tl = []
Tc = Timer()

for sweepi in range(Setting.SweepTimes):

    N = Setting.N
    N_batch = Setting.Minibatch

    idxall = np.linspace(0, N-1, N)
    # np.random.shuffle(idxall)
    MBratio = np.floor(N/N_batch).astype('int')
    Loss_t = 0
    Likelihood_t = 0
    start_time = time.time()

    graph_feed_dict = {}
    if args.sub_sample:
        graph_feed_dict = get_feed_dict(sampled_nodes)

    Tc.start()
    # update theta
    for MBt in range(MBratio):

        MBObserved = (sweepi*MBratio + MBt).astype('int')
        MB_index = idxall[MBt*N_batch + np.arange(N_batch)].astype('int')
        X_batch = np.array(train_data[:, MB_index], order='C', dtype=np.float32)  # V*N
        #X_batch = np.array(train_data[:, sampled_nodes], order='C', dtype=np.float32)
        if sweepi < 100:
            SGD_Iter = 1
        else:
            SGD_Iter = 1
        # print('shape : ',np.transpose(X_batch))
        for i in range(SGD_Iter):

            train_feed_dict = {X_NV: np.transpose(X_batch),
                                                 bias_in: bias,
                                                 Phi_1: np.float32(Params.Phi[0]),
                                                 Batch_Size: N_batch,
                                                 A_NN: np.float32(train_graph),
                                                 # graph_input: adj_norm,
                                                 graph_label: adj_label,
                                                 Y_labels: y_train,
                                                 Y_labels_mask: train_mask,
                                                 attn_drop: 0.6,
                                                 ffd_drop: 0.6,
                                                 is_train: 1.0,
                                                 dropout: 0.0
                                }

            feed_dict = {**graph_feed_dict, **train_feed_dict}

            Outs = sess.run([Opt_op], feed_dict=feed_dict)


    # update Phi

    updata_feed_dict = {X_NV: np.transpose(X_batch),
                                         bias_in: bias,
                                         Phi_1: np.float32(Params.Phi[0]),
                                         Batch_Size: N_batch,
                                         A_NN: np.float32(train_graph),
                                         # graph_input: adj_norm,
                                         graph_label: adj_label,
                                         Y_labels: y_train,
                                         Y_labels_mask: train_mask,
                                         attn_drop: 0.6,
                                         ffd_drop: 0.6,
                                         is_train: 1.0,
                                         dropout: 0.0
                                         }

    feed_dict = {**graph_feed_dict, **updata_feed_dict}

    Theta = sess.run(Theta_1, feed_dict=feed_dict)


    Params.Theta = [np.float64(Theta)]
    if args.sub_update:
        sub_Theta = [np.array(np.float64(Theta[:, sampled_nodes]), order='C')]
        minibatch = np.array(train_data[:, sampled_nodes], order='C').astype('double')
        MBr = np.floor(Setting.N / nb_node_samples).astype('int')

        Params.Phi = updatePhi(miniBatch=minibatch, Phi=Params.Phi, Theta=sub_Theta, MBratio=MBr, MBObserved=1)
    else:
        for t in range(Setting.T):
            if t == 0:
                Xt = np.array(train_data, order='C') #Xt.astype('double')
                Params.Xt_to_t1[t], Params.WSZS[t] = PGBN_sampler.Multrnd_Matrix(Xt.astype('double'),
                                                                                 Params.Phi[t],
                                                                                 Params.Theta[t])
                ttt = Params.WSZS[t].sum(0)
            else:
                Params.Xt_to_t1[t], Params.WSZS[t] = PGBN_sampler.Crt_Multirnd_Matrix(Params.Xt_to_t1[t-1].astype('double'),
                                                                                      Params.Phi[t],
                                                                                      Params.Theta[t])
            Params.Phi[t][:, :] = PGBN_sampler.Sample_Phi(Params.WSZS[t], Params.Eta[t])

    tl.append(Tc.stop())
    end_time = time.time()

    if np.mod(sweepi, 1) == 0:

        train_feed_dict = {X_NV: np.transpose(X_batch),
                                    bias_in: bias,
                                    Phi_1: np.float32(Params.Phi[0]),
                                    Batch_Size: N_batch,
                                    A_NN: np.float32(train_graph),
                                    # graph_input: adj_norm,
                                    graph_label: adj_label,
                                    Y_labels: y_test,
                                    Y_labels_mask: test_mask,
                                    attn_drop: 0.0,
                                    ffd_drop: 0.0,
                                    is_train: 0.0,
                                    dropout: 0.0}

        feed_dict = {**graph_feed_dict, **train_feed_dict}

        outs = sess.run([Label_acc, Label_loss, data_LH, graph_LH, KL_loss],
                         feed_dict=feed_dict)

        [Label_acc_t, Label_loss_t, data_LH_t, graph_LH_t, kl_loss] = outs



        print('Epoch: {:3d}'.format(sweepi))

        print('Ac: {:<8.4f}, Label_LH: {:<8.4f}, Data_LH: {:<8.4f}, Graph_LH: {:<8.4f}, KL_loss: {:<8.4f}'.format(Label_acc_t, Label_loss_t, data_LH_t, graph_LH_t, args.att_kl * kl_loss))

        if Label_acc_t>Best_Label_Ac:
            Best_Label_Ac = Label_acc_t
        print('Best_Ac: {:<8.4f}'.format(Best_Label_Ac))

        val.append(Label_acc_t)

# t = np.arange(0, Setting.SweepTimes)
# tmp = 0
# for i in range(len(tl)):
#     tl[i] = tmp + tl[i]
#     tmp = tl[i]
# f = open('./cora_acc/WGAAE_T100nodes_Phi.pkl', 'wb')
# dict = {'tl': tl, 'val': val}
# pickle.dump(dict, f)
# f.close()


