import numpy as np
import tensorflow as tf
import scipy.sparse as sp
import networkx as nx

conv1d = tf.layers.conv1d

def LeakyRelu(x, leak=0.2, name="LeakyRelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * tf.abs(x)

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense() #, sparse_to_tuple(features)

class SpGAT():
    def GAT_init(self,features):
        self.nb_nodes = features.shape[0].value
        self.ft_size = features.shape[1].value
        print('num class = ',self.nb_nodes)
        print('ft size = ',self.ft_size)
        self.nb_classes = 256 #64
        self.hid_units = [32]
        self.n_heads = [8, 1]
        self.residual = False
        self.activation = tf.nn.elu
        self.mid_activation = lambda x: x
        self.last_activation = tf.nn.softplus

    def preprocess_adj_bias(self, adj):
        num_nodes = adj.shape[0]
        adj = adj + sp.eye(num_nodes)  # self-loop
        adj[adj > 0.0] = 1.0
        if not sp.isspmatrix_coo(adj):
            adj = adj.tocoo()
        adj = adj.astype(np.float32)
        indices = np.vstack(
            (adj.col, adj.row)).transpose()  # This is where I made a mistake, I used (adj.row, adj.col) instead
        # return tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape)
        return indices, adj.data, adj.shape

    def sp_attn_head(self,seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False, training=1.0, args=None):
        eps = 1e-20
        use_bias = True
        with tf.name_scope('sp_attn'):
            if in_drop != 0.0:
                seq = tf.nn.dropout(seq, 1.0 - in_drop)

            seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

            # simplest self-attention possible
            f_1 = tf.layers.conv1d(seq_fts, 1, 1)
            f_2 = tf.layers.conv1d(seq_fts, 1, 1)

            f_1 = tf.reshape(f_1, (nb_nodes, 1))
            f_2 = tf.reshape(f_2, (nb_nodes, 1))
            f_1 = adj_mat * f_1
            f_2 = adj_mat * tf.transpose(f_2, [1, 0])

            logits = tf.sparse_add(f_1, f_2)
            lrelu = tf.SparseTensor(indices=logits.indices,
                                    values=LeakyRelu(logits.values),
                                    dense_shape=logits.dense_shape)
            coefs = tf.sparse_softmax(lrelu)
            eps_tensor = tf.SparseTensor(indices=logits.indices,
                                         values=tf.ones_like(logits.values, dtype=tf.float32) * eps,
                                         dense_shape=logits.dense_shape)
            one_tensor = tf.SparseTensor(indices=logits.indices,
                                         values=tf.ones_like(logits.values, dtype=tf.float32),
                                         dense_shape=logits.dense_shape)
            # eps_tensor = tf.cast(tf.constant([eps]), tf.float32)
            logprobs = tf.SparseTensor(indices=logits.indices,
                                       values=tf.log(tf.sparse_add(coefs, eps_tensor).values),
                                       dense_shape=logits.dense_shape)
            if args.att_prior_type == 'contextual':
                kernel_initializer = tf.contrib.keras.initializers.he_normal()  # glorot_normal
                bias_initializer = tf.zeros_initializer()
                if args.att_type == 'soft_weibull':
                    dot_gamma = tf.layers.dense(seq_fts, args.att_se_hid_size, activation=None, use_bias=use_bias,
                                                kernel_initializer=kernel_initializer,
                                                bias_initializer=bias_initializer)
                    dot_gamma = tf.nn.relu(dot_gamma)
                    dot_gamma = tf.layers.dense(dot_gamma, 1, activation=None, use_bias=use_bias,
                                                kernel_initializer=kernel_initializer,
                                                bias_initializer=bias_initializer)

                    dot_gamma = tf.transpose(dot_gamma, [0, 2, 1])
                    print('****************************dot_gamma shape', dot_gamma.get_shape().as_list())
                    alpha_gamma = tf.nn.softmax(dot_gamma, dim=-1) * args.beta_gamma
                    prior_att_weights = alpha_gamma / tf.reduce_sum(alpha_gamma, axis=-1, keep_dims=True)
                if args.att_type == 'soft_lognormal':
                    dot_mu = tf.layers.dense(seq_fts, args.att_se_hid_size, activation=None, use_bias=use_bias,
                                             kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
                    dot_mu = tf.nn.relu(dot_mu)
                    dot_mu = tf.layers.dense(dot_mu, 1, activation=None, use_bias=use_bias,
                                             kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
                    dot_mu = tf.transpose(dot_mu, [0, 2, 1])
                    dot_mu = tf.nn.softmax(dot_mu)
                    mean_normal_prior = tf.log(dot_mu + eps)
                    prior_att_weights = dot_mu
            else:
                alpha_gamma = args.alpha_gamma
                mean_normal_prior = 0.0

            if args.att_type == 'soft_weibull':
                print('weibull attention !')
                rand_uniform = tf.random_uniform(shape=[12431])
                sample_weibull = tf.SparseTensor(indices=logits.indices,
                                                 values=logprobs.values - tf.lgamma(1 + 1.0 / args.k_weibull) + tf.log(
                                                     -tf.log(tf.ones_like(
                                                         logits.values) - rand_uniform + eps * tf.ones_like(
                                                         logits.values)) + eps * tf.ones_like(
                                                         logits.values)) / args.k_weibull,
                                                 dense_shape=logits.dense_shape)
                lambda_weibull = tf.SparseTensor(indices=logits.indices,
                                                 values=tf.exp(logprobs.values - tf.lgamma(1 + 1.0 / args.k_weibull)),
                                                 dense_shape=logits.dense_shape)
                sample_weibull = tf.sparse_softmax(sample_weibull)
                mean_weibull = tf.sparse_softmax(logprobs)
                out_coefs = tf.SparseTensor(indices=logits.indices,
                                            values=sample_weibull.values * training + mean_weibull.values * (
                                                    1 - training),
                                            dense_shape=logits.dense_shape)
                if args.att_prior_type == 'contextual':
                    alpha_gamma_sparse = tf.SparseTensor(indices=logits.indices,
                                                         values=tf.gather_nd(tf.squeeze(tf.squeeze(alpha_gamma, 0), 0),
                                                                             tf.expand_dims(logits.indices[:, 1],
                                                                                            axis=1)),
                                                         dense_shape=logits.dense_shape)

                    KL_1 = tf.SparseTensor(indices=logits.indices,
                                           values=(logprobs.values - tf.lgamma(1 + 1.0 / args.k_weibull)) *
                                                  alpha_gamma_sparse.values - args.beta_gamma * lambda_weibull.values *
                                                  tf.exp(tf.lgamma(1 + 1.0 / args.k_weibull)),
                                           dense_shape=logits.dense_shape)
                    KL_1_mean = tf.sparse_reduce_sum(KL_1) / tf.sparse_reduce_sum(one_tensor)
                    KL_2_mean = tf.reduce_mean(- np.euler_gamma * alpha_gamma_sparse.values / args.k_weibull +
                                               alpha_gamma_sparse.values * tf.log(args.beta_gamma + eps) -
                                               tf.lgamma(alpha_gamma_sparse.values + eps))
                    KL_backward = - (KL_1_mean + KL_2_mean)
                else:
                    KL = tf.SparseTensor(indices=logits.indices,
                                         values=(logprobs.values - tf.lgamma(1 + 1.0 / args.k_weibull)) * alpha_gamma -
                                                args.beta_gamma * lambda_weibull.values *
                                                tf.exp(tf.lgamma(1 + 1.0 / args.k_weibull)),
                                         dense_shape=logits.dense_shape)
                    KL_backward = - tf.sparse_reduce_sum(KL) / tf.sparse_reduce_sum(one_tensor)
                tf.add_to_collection('kl_list', KL_backward)

            elif args.att_type == 'soft_lognormal':
                print('lognormal attention !')
                mean_normal_posterior = logprobs.values - args.sigma_normal_posterior ** 2 / 2
                sample_normal_value = mean_normal_posterior + args.sigma_normal_posterior * tf.random_normal(
                    shape=[12431], dtype=tf.float32
                )
                sample_normal = tf.SparseTensor(indices=logits.indices,
                                                values=sample_normal_value,
                                                dense_shape=logits.dense_shape)
                sample_normal = tf.sparse_softmax(sample_normal)
                mean_normal = tf.sparse_softmax(logprobs)
                out_coefs = tf.SparseTensor(indices=logits.indices,
                                            values=sample_normal.values * training + mean_normal.values * (
                                                    1 - training),
                                            dense_shape=logits.dense_shape)
                if args.att_prior_type == 'contextual':
                    mean_normal_prior_sparse = tf.SparseTensor(indices=logits.indices,
                                                               values=tf.gather_nd(
                                                                   tf.squeeze(tf.squeeze(mean_normal_prior, 0), 0),
                                                                   tf.expand_dims(logits.indices[:, 1], axis=1)),
                                                               dense_shape=logits.dense_shape)

                    KL = tf.reduce_mean((args.sigma_normal_posterior ** 2 + (
                            mean_normal_prior_sparse.values - mean_normal_posterior) ** 2) / (
                                                2 * args.sigma_normal_prior ** 2))
                    KL_backward = KL
                else:
                    # Only include terms that have gradients.
                    KL = tf.reduce_mean((args.sigma_normal_posterior ** 2 + (
                            mean_normal_prior - mean_normal_posterior) ** 2) / (
                                                2 * args.sigma_normal_prior ** 2))
                    KL_backward = KL
                tf.add_to_collection('kl_list', KL_backward)
            else:
                print('orignal~')
                out_coefs = coefs

            if coef_drop != 0.0:
                out_coefs = tf.SparseTensor(indices=out_coefs.indices,
                                        values=tf.nn.dropout(out_coefs.values, 1.0 - coef_drop),
                                        dense_shape=out_coefs.dense_shape)
            if in_drop != 0.0:
                seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

            # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
            # here we make an assumption that our input is of batch size 1, and reshape appropriately.
            # The method will fail in all other cases!
            out_coefs = tf.sparse_reshape(out_coefs, [nb_nodes, nb_nodes])
            seq_fts = tf.squeeze(seq_fts)
            vals = tf.sparse_tensor_dense_matmul(out_coefs, seq_fts)
            vals = tf.expand_dims(vals, axis=0)
            vals.set_shape([1, nb_nodes, out_sz])
            ret = tf.contrib.layers.bias_add(vals)

            # residual connection
            if residual:
                if seq.shape[-1] != ret.shape[-1]:
                    ret = ret + conv1d(seq, ret.shape[-1], 1)  # activation
                else:
                    ret = ret + seq

            return activation(ret)  # activation

    def inference(self, inputs, nb_classes, nb_nodes, attn_drop, ffd_drop,
                  bias_mat, hid_units, n_heads,
                  residual=False, training=1.0, args=None):

        attns = []
        for _ in range(n_heads[0]):
            attns.append(self.sp_attn_head(inputs,
                                             adj_mat=bias_mat,
                                             out_sz=hid_units[0], activation=self.activation, nb_nodes=nb_nodes,
                                             in_drop=ffd_drop, coef_drop=attn_drop, residual=False, training=training, args=args))
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(self.sp_attn_head(h_1,
                                                 adj_mat=bias_mat,
                                                 out_sz=hid_units[i], activation=self.activation, nb_nodes=nb_nodes,
                                                 in_drop=ffd_drop, coef_drop=attn_drop, residual=residual, training=training, args=args))
            h_1 = tf.concat(attns, axis=-1)
        out = []
        for i in range(n_heads[-1]):
            out.append(self.sp_attn_head(h_1, adj_mat=bias_mat,
                                           out_sz=nb_classes, activation=self.mid_activation, nb_nodes=nb_nodes,
                                           in_drop=ffd_drop, coef_drop=attn_drop, residual=False, training=training, args=args))
        logits = tf.add_n(out) / n_heads[-1]
        # logits = tf.nn.softplus(logits)
        #logits = h_1
        return logits
