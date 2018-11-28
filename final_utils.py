
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.special import factorial
from edward.models import Poisson
import tensorflow as tf


# In[ ]:


# EM Algorithm
def EM(K, N, data):
    pi_EM = np.ones(K) / K
    lam_EM = 30 * np.random.rand(K)
    q_EM = np.zeros([K, N])
    fact = factorial(data)
    for _ in range(100):
        for i in range(K):
            for j in range(N):
                q_EM[i, j] = pi_EM[i] * lam_EM[i] ** \
                             data[j] * fact[j] * np.exp(-lam_EM[i])
        q = np.tile(np.sum(q_EM, axis=0), [K, 1])
        p = np.divide(q_EM, q)
        z = np.sum(p, axis=1)
        pi_EM = z / N
        lam_EM = np.divide(p.dot(data), z)
    return pi_EM, lam_EM, q_EM


# In[ ]:


# draw samples from the posterior distributions of the parameters
def sample_parameters(pi_dist, lam_dist, S, K):
    pi_samples = np.zeros([S, K])
    lam_samples = np.zeros([S, K])
    pi_samples = pi_dist.sample(S).eval()
    lam_samples = lam_dist.sample(S).eval()
    return pi_samples, lam_samples


# In[ ]:


# calculate probability and log-probability
def calc_prob(pi_samples, lam_samples, y, S, K):
    log_prob = tf.constant(0.0, dtype=tf.float64)
    prob = tf.constant(0.0, dtype=tf.float64)
    for s in range(S):
        p_y = tf.gather_nd(pi_samples, [s, 0]) * \
              Poisson(tf.gather_nd(lam_samples, [s, 0])).prob(y)
        for j in range(1, K):
            p_y += tf.gather_nd(pi_samples, [s, j]) * \
                   Poisson(tf.gather_nd(lam_samples, [s, j])).prob(y)
        log_prob += tf.log(tf.cast(p_y, tf.float64))
        prob += tf.cast(p_y, tf.float64)
    log_prob = log_prob / S
    prob = prob / S
    return log_prob.eval(), prob.eval()


# In[ ]:


# DIC, WAIC and lppd
def info_crit(qpi, qlam, K, data):
    pi_Bayes = tf.constant(np.array(qpi.mean().eval(), ndmin=2))
    lam_Bayes = tf.constant(np.array(qlam.mean().eval(), ndmin=2))
    elpd_DIC = 0
    lppd = 0
    p_WAIC = 0
    post_prob = np.zeros(31)
    for i in range(31):
        log_Bayes, _ = calc_prob(pi_Bayes, lam_Bayes, np.float32(i), 1, K)
        pi_samples, lam_samples = sample_parameters(qpi, qlam, 100, K)
        log_post, prob_post = calc_prob(pi_samples,
                                        lam_samples, np.float32(i), 100, K)
        post_prob[i] = prob_post
        num = data.count(i)
        elpd_DIC += (2 * log_post - log_Bayes) * num
        lppd += np.log(prob_post) * num
        p_WAIC += 2*(np.log(prob_post) - log_post) * num
    DIC = -2 * elpd_DIC
    WAIC = -2 * (lppd - p_WAIC)
    return DIC, WAIC, lppd, post_prob


# AIC
def AIC_calc(pi, lam, K, data):
    pi_EM = tf.constant(np.array(pi, ndmin=2))
    lam_EM = tf.constant(np.array(lam, ndmin=2))
    elpd_AIC = 0
    for i in range(31):
        log_EM, _ = calc_prob(pi_EM, lam_EM, np.float64(i), 1, K)
        elpd_AIC += log_EM * data.count(i)
    elpd_AIC -= 2 * K
    AIC = -2 * elpd_AIC
    return AIC
