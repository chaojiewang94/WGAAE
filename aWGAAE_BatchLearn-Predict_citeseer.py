import copy
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
from input_data import load_data
from preprocessing import mask_test_edges, preprocess_graph, sparse_to_tuple
from GAT.gat_hwithBAM_prediction_citeseer import *
from GAT.utils import process
from sampling import get_distribution, node_sampling, preprocess_graph4sample, construct_feed_dict

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='citeseer', help='pubmed, cora, citeseer')
parser.add_argument('--task', type=str, default='prediction', help='prediction, classification, clustering')
parser.add_argument('--epochs', type=int, default=30000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--id', type=str, default='default')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--sub_sample', type=bool, default=False, help='Whether sample sub-graph')
parser.add_argument('--sub_update', type=bool, default=False, help='')
parser.add_argument('--nb_node_sample', type=int, default=1000, help='number of sample nodes')
parser.add_argument('--k', type=float, default=1.0, help='sample distribution')
parser.add_argument('--pred_dim', type=int, default=8, help='Dimension for output')

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

    data = sio.loadmat('./data/mnist_gray')
    train_data = np.array(np.ceil(data['train_mnist']*25), order='C')[:, 0:50]  # 0-1    V*N
    test_data = np.array(np.ceil(data['test_mnist']*25), order='C')[:, 0:50]  # 0-1    V*N
    train_label = data['train_label']
    test_label = data['test_label']

    train_graph = Cosine_Simlarity(train_data.T, train_data.T)  # replace this train_graph with your user-user network
    train_graph[(np.arange(train_graph.shape[0]), np.arange(train_graph.shape[0]))] = 0  # diag is zeros
    train_graph[np.where(train_graph < 0.5)] = 0
    train_graph = np.ceil(train_graph*50)
    test_graph = Cosine_Simlarity(test_data.T, test_data.T)
    test_graph[(np.arange(test_graph.shape[0]), np.arange(test_graph.shape[0]))] = 0  # diag is zeros
    test_graph = np.ceil(test_graph*50)

elif args.dataset == 'coil':

    data = sio.loadmat('./COIL20_GRAPH.mat')
    train_data = np.array(np.transpose(np.ceil(data['fea']*25)), order='C')  # V*N We use the feature after L2 norm
    train_label = data['gnd']
    train_graph = data['A']  # diagonal is zeros
    train_n_class = np.unique(train_label).size

elif args.dataset == 'pie':

    data = sio.loadmat('./PIE_pose27_GRAPH.mat')
    train_data = np.array(np.transpose(np.ceil(data['fea'])), order='C')  # V*N We use the orginal feature
    train_data = np.ceil(train_data / 5)
    train_label = data['gnd']
    train_graph = data['A']
    train_n_class = np.unique(train_label).size

else:
    args.task = 'prediction'
    adj, features = load_data(args.dataset)

    train_data = features.todense().T  ## V * N
    train_graph = adj.toarray()
    train_label = None
    train_n_class = 0
    # if args.dataset == ''

##====================== Graph Preprocess ======================##
# preprocess for graph
print('train graph shape : ',train_graph.shape)
gat_adj = copy.deepcopy(adj)
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
adj_label = adj + sp.eye(adj.shape[0])  # adj_train + eye
adj_label = sparse_to_tuple(adj_label)

##====================== Setting for PGBN ======================##
# setting
# np.random.seed(12)
# tf.set_random_seed(123)

Setting = Empty()
Setting.V = train_data.shape[0]
Setting.N = train_data.shape[1]
Setting.K = [256, 256, 256]  # len(np.unique(train_label))
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
Params.Xt_to_t1 = [0] * Setting.T
Params.WSZS = [0] * Setting.T
Params.NDot = [0] * Setting.T
Params.EWSZS = [0] * Setting.T
Params.ForgetRate = np.power((Setting.tao0FR + np.linspace(1, Setting.Iterall, Setting.Iterall)), - Setting.kappa0FR)
Params.epsit = np.power((Setting.tao0 + np.linspace(1, Setting.Iterall, Setting.Iterall)), -Setting.kappa0)
Params.epsit = Setting.epsi0 * Params.epsit / Params.epsit[0]
for t in range(Setting.T):
    Params.Eta.append(Supara.eta[t])
    if t == 0:
        Params.Phi.append(0.2 + 0.8 * np.float64(np.random.rand(Setting.V, Setting.K[t])))
    else:
        Params.Phi.append(0.2 + 0.8 * np.float64(np.random.rand(Setting.K[t - 1], Setting.K[t])))
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


def sample_subgraph(sp_adj, nb_node_samples, replace): # gat_adj
    ######## Sampling Subgraph Nodes ##########
    # Node-level p_i degree-based, core-based or uniform distribution
    measure = 'degree'
    alpha = 1.0
    k = args.k
    print('sample nodes = ', nb_node_samples)

    # node distribution
    node_distribution = get_distribution(measure, alpha, sp_adj)

    # Node sampling
    sampled_nodes, adj_label, adj_sampled_sparse = node_sampling(sp_adj,
                                                      node_distribution,
                                                      nb_node_samples,
                                                      k,
                                                      replace)
    placeholders = {
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'sampled_nodes': tf.placeholder_with_default(sampled_nodes, shape=[nb_node_samples])
    }
    adj_ori = tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False), [-1])

    adj_norm = preprocess_graph4sample(sp_adj)
    return [adj_norm, adj_label, adj_ori, sampled_nodes, adj_sampled_sparse, placeholders, node_distribution]

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

    return graphA_LH # minus

##====================== Build Graph ======================##
Batch_Size = tf.placeholder(tf.int32)
X_NV = tf.placeholder(tf.float32, shape=[Setting.N, Setting.V])  # N*V
X_VN = tf.transpose(X_NV)  # V*N
A_NN = tf.placeholder(tf.float32, shape=[Setting.N, Setting.N])  # origin
D_NN = tf.matrix_diag(tf.reduce_sum(A_NN, axis=1))
L_NN = D_NN - A_NN

attn_drop = tf.placeholder(dtype=tf.float32, shape=())
ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
is_train = tf.placeholder(dtype=tf.float32, shape=())

# GAT
GAT = SpGAT()
GAT.GAT_init(X_NV)
bias = GAT.preprocess_adj_bias(gat_adj)
bias_in = tf.sparse_placeholder(dtype=tf.float32)

# layers
input = tf.expand_dims(X_NV, 0)
h_1 = GAT.inference(input, GAT.nb_classes, GAT.nb_nodes,
                                 bias_mat=bias_in,
                                 attn_drop=attn_drop,
                                 ffd_drop=ffd_drop,
                                 hid_units=GAT.hid_units, n_heads=GAT.n_heads,
                                 residual=GAT.residual, training=is_train, args=args
                    )

h_2 = GAT.inference(h_1, GAT.nb_classes, GAT.nb_nodes,
                                 bias_mat=bias_in,
                                 attn_drop=attn_drop,
                                 ffd_drop=ffd_drop,
                                 hid_units=GAT.hid_units, n_heads=GAT.n_heads,
                                 residual=GAT.residual, training=is_train, args=args
                    )

h_3 = GAT.inference(h_2, GAT.nb_classes, GAT.nb_nodes,
                                 bias_mat=bias_in,
                                 attn_drop=attn_drop,
                                 ffd_drop=ffd_drop,
                                 hid_units=GAT.hid_units, n_heads=GAT.n_heads,
                                 residual=GAT.residual, training=is_train, args=args
                    )

h_1 = tf.squeeze(h_1, [0])
h_2 = tf.squeeze(h_2, [0])
h_3 = tf.squeeze(h_3, [0])


Phi_1 = tf.placeholder(tf.float32, shape=[Setting.V, Setting.K[0]])
Phi_2 = tf.placeholder(tf.float32, shape=[Setting.K[0], Setting.K[1]])
Phi_3 = tf.placeholder(tf.float32, shape=[Setting.K[1], Setting.K[2]])

dropout = tf.placeholder_with_default(0., shape=())
# # graph_input = tf.sparse_placeholder(tf.float32)  # for gcn
graph_label = tf.sparse_placeholder(tf.float32)  # for loss
graph_label_full = tf.reshape(tf.sparse_tensor_to_dense(graph_label, validate_indices=False), [-1])


# cal k and l
k_3, l_3 = encoder_right_sparse(h_3, 2)  # K_3*N
Theta_3 = reparameterization(k_3, l_3, 2, Setting.Minibatch)  # K_3*N
k_2, l_2 = encoder_right_sparse(h_2, 1)  # K_2*N
Theta_2 = reparameterization(k_2, l_2, 1, Setting.Minibatch)  # K_2*N
k_1, l_1 = encoder_right_sparse(h_1, 0)  # K_1*N
Theta_1 = reparameterization(k_1, l_1, 0, Setting.Minibatch)  # K_1*N

data_re = tf.matmul(Phi_1, Theta_1)
u_1 = weight_variable(shape=[1, 1])
u_2 = weight_variable(shape=[1, 1])
u_3 = weight_variable(shape=[1, 1])
res = tf.layers.dense(tf.concat([h_1, tf.transpose(Theta_1),
                                 h_2, tf.transpose(Theta_2),
                                 h_3, tf.transpose(Theta_3)], axis=1), args.pred_dim)

Theta_concat = tf.transpose(tf.concat([u_1 * Theta_1, u_2 * Theta_2, u_3 * Theta_3], axis=0))

# optimizer

Theta_3_KL = tf.reduce_mean(KL_GamWei(np.float32(1.0), np.float32(1.0), k_3, l_3))
Theta_2_KL = tf.reduce_mean(KL_GamWei(tf.matmul(Phi_3, Theta_3), np.float32(1.0), k_2, l_2))
Theta_1_KL = tf.reduce_mean(KL_GamWei(tf.matmul(Phi_2, Theta_2), np.float32(1.0), k_1, l_1))  # / tf.cast(Batch_Size, tf.float32)
#
# lap_loss_1 = tf.trace(tf.matmul(tf.matmul(Theta_1, L_NN), tf.transpose(Theta_1)))
# lap_loss_2 = tf.trace(tf.matmul(tf.matmul(Theta_2, L_NN), tf.transpose(Theta_2)))
# lap_loss_3 = tf.trace(tf.matmul(tf.matmul(Theta_3, L_NN), tf.transpose(Theta_3)))

data_LH = tf.reduce_sum(X_VN * log_max(tf.matmul(Phi_1, Theta_1)) - tf.matmul(Phi_1, Theta_1) - tf.lgamma(X_VN + 1))  # / tf.cast(Batch_Size, tf.float32)


if args.sub_sample:
    nb_node_samples = args.nb_node_sample
    replace = False
    eta = 0.001
    outs = sample_subgraph(gat_adj, nb_node_samples, replace)
    [sub_adj_norm, sub_adj_label, sub_adj_ori, sampled_nodes, adj_sampled_sparse, placeholders, node_distribution] = outs
    graph_LH = eta * get_graphA_LH(sub_adj_ori, sampled_nodes, adj_sampled_sparse, Theta_concat)

else:
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    eta = 0.001
    inner_product = InnerProductDecoder(Theta_concat, dropout)
    graph_re = bern_possion_link(inner_product)  # (N*N) * 1
    graph_LH = eta * norm * tf.reduce_sum(
        graph_label_full * log_max(graph_re) * pos_weight + (1 - graph_label_full) * log_max(1 - graph_re))


loss = data_LH + graph_LH
       # + 0 * Theta_1_KL + 0 * Theta_2_KL + 0 * Theta_3_KL\
       # - 0 * lap_loss_1 - 0 * lap_loss_2 - 0 * lap_loss_3  # 1 1e-2 for citeer dropout 0.1// 1 1e-2 for cora

LowerBound = data_LH + Theta_1_KL + Theta_2_KL + Theta_3_KL

vars = tf.trainable_variables()
for v in vars:
    print('v name : ',v.name)
LossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not
                   in ['bias', 'gamma', 'b', 'g', 'beta']]) * args.l2_coef

if args.att_type != 'soft_attention':
    KL_loss = args.att_kl * tf.add_n(tf.get_collection('kl_list')) / len(tf.get_collection('kl_list'))
    # KL_loss = tf.exp(Setting.SweepTimes * args.kl_anneal_rate) / (1 + tf.exp(Setting.SweepTimes * args.kl_anneal_rate)) * KL_loss
    Loss = loss - KL_loss - LossL2
else:
    Loss = loss - LossL2

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

    # update theta
    for MBt in range(MBratio):

        MBObserved = (sweepi*MBratio + MBt).astype('int')
        MB_index = idxall[MBt*N_batch + np.arange(N_batch)].astype('int')
        X_batch = np.array(train_data[:, MB_index], order='C', dtype=np.float32)  # V*N

        if sweepi < 100:
            SGD_Iter = 20
        else:
            SGD_Iter = 5

        for i in range(SGD_Iter):

            train_feed_dict = {X_NV: np.transpose(X_batch),
                                                 bias_in: bias,
                                                 Phi_1: np.float32(Params.Phi[0]),
                                                 Phi_2: np.float32(Params.Phi[1]),
                                                 Phi_3: np.float32(Params.Phi[2]),
                                                 Batch_Size: N_batch,
                                                 graph_label: adj_label,
                                                 # A_NN: np.float32(train_graph)
                                                 attn_drop: args.dropout,
                                                 ffd_drop: args.dropout,
                                                 is_train: 1.0,
                                                 dropout: 0.0}

            feed_dict = {**graph_feed_dict, **train_feed_dict}

            Outs = sess.run([Opt_op], feed_dict=feed_dict)


    # update Phi

    updata_feed_dict = {X_NV: np.transpose(X_batch),
                                            bias_in: bias,
                                            Phi_1: np.float32(Params.Phi[0]),
                                            Phi_2: np.float32(Params.Phi[1]),
                                            Phi_3: np.float32(Params.Phi[2]),
                                            Batch_Size: N_batch,
                                            graph_label: adj_label,
                                            # A_NN: np.float32(train_graph)
                                            attn_drop: args.dropout,
                                            ffd_drop: args.dropout,
                                            is_train: 1.0,
                                            dropout: 0.0}

    feed_dict = {**graph_feed_dict, **updata_feed_dict}

    Theta = sess.run([Theta_1, Theta_2, Theta_3], feed_dict=feed_dict)


    Params.Theta = [np.float64(Theta[0]), np.float64(Theta[1]), np.float64(Theta[2])]

    if args.sub_update:
        sub_Theta = [np.array(np.float64(Theta[0][:, sampled_nodes]), order='C'),
                     np.array(np.float64(Theta[1][:, sampled_nodes]), order='C'),
                     np.array(np.float64(Theta[2][:, sampled_nodes]), order='C')]
        minibatch = np.array(train_data[:, sampled_nodes], order='C').astype('double')
        MBr = np.floor(Setting.N / nb_node_samples).astype('int')

        Params.Phi = updatePhi(miniBatch=minibatch, Phi=Params.Phi, Theta=sub_Theta, MBratio=MBr, MBObserved=1)
    else:
        for t in range(Setting.T):
            if t == 0:
                Xt = np.array(train_data, order='C')
                Params.Xt_to_t1[t], Params.WSZS[t] = PGBN_sampler.Multrnd_Matrix(Xt.astype('double'),
                                                                                 Params.Phi[t],
                                                                                 Params.Theta[t])
            else:
                Params.Xt_to_t1[t], Params.WSZS[t] = PGBN_sampler.Crt_Multirnd_Matrix(
                    Params.Xt_to_t1[t - 1].astype('double'),
                    Params.Phi[t],
                    Params.Theta[t])
            Params.Phi[t][:, :] = PGBN_sampler.Sample_Phi(Params.WSZS[t], Params.Eta[t])

    end_time = time.time()

    if np.mod(sweepi, 1) == 0:

        train_feed_dict = {X_NV: np.transpose(X_batch),
                           bias_in: bias,
                           Phi_1: np.float32(Params.Phi[0]),
                           Phi_2: np.float32(Params.Phi[1]),
                           Phi_3: np.float32(Params.Phi[2]),
                           Batch_Size: N_batch,
                           graph_label: adj_label,
                           # A_NN: np.float32(train_graph)
                           attn_drop: 0.0,
                           ffd_drop: 0.0,
                           is_train: 0.0,
                           dropout: 0.0
                           }

        feed_dict = {**graph_feed_dict, **train_feed_dict}

        outs = sess.run([Loss, data_LH, graph_LH, Theta_1_KL, LowerBound, KL_loss, LossL2,
                         Theta_1, Theta_concat], feed_dict=feed_dict)

        [Loss_t, data_LH_t, graph_LH_t, KL1_t, LowerBound_t, kl_loss, lossL2, Theta_t, Theta_concat_t] = outs


        # prediction metric
        try:
            AUC_score, AP_score = get_roc_score(test_edges, test_edges_false, adj_orig, emb=Theta_concat_t)
            predict_test_acc = get_acc(test_edges, test_edges_false, emb=Theta_concat_t)
        except:
            AUC_score, AP_score = 0.0, 0.0
            predict_test_acc = 0.0

        Best_AUC = np.maximum(Best_AUC, AUC_score)
        Best_AP = np.maximum(Best_AP, AP_score)

        print('Epoch: {:3d}'.format(sweepi))

        print('Loss: {:<8.4f}, Data_LH: {:<8.4f}, Graph_LH: {:<8.4f}, KL_1: {:<8.4f}, KL_loss: {:<8.4f}, lossl2: {:<8.4f}'.format(
            Loss_t, data_LH_t, graph_LH_t, KL1_t, kl_loss, lossL2))
        print('Prediction_AUC: {:<8.4f}, Prediction_AP: {:<8.4f}'.format(
            AUC_score, AP_score))
        print('Best_Prediction_AUC: {:<8.4f}, Best_Prediction_AP: {:<8.4f}'.format(
            Best_AUC, Best_AP))

        # u = sess.run([u_1, u_2, u_3])
        # print(u)

        # save_file_name = './WGCAI-'
        #
        # sio.savemat(save_file_name + 'lambda-' + str(args.lamda) + '-' + args.graph_lh + '-' + args.dataset + '-theta.mat',
        #             {'Theta': np.transpose(Theta),
        #              'adj_train': (adj + sp.eye(adj.shape[0])).toarray()})


