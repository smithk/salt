from __future__ import print_function
import sys
from salt.parameters.param import Uniform, GaussianMixture, Normal
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp


true_distribution = GaussianMixture(means=[1., 8., 12.],
                                    stdevs=[2.5, 1., 1.2],
                                    weights=[.5, 0.3, 0.2])
noise = Normal(0., .00001)
alpha = 0.5


def get_score(value):
    return np.exp(true_distribution.eval([value])) + noise.get_sample()


def good_fit(bucket, learned):
    some_true_values = true_distribution.get_sample(len(bucket)).ravel()
    _, p_value = ks_2samp(bucket, some_true_values)
    print(p_value, p_value >= alpha)
    return p_value >= alpha


def get_fit(bucket, learned):
    some_true_values = true_distribution.get_sample(len(bucket)).ravel()
    _, p_value = ks_2samp(bucket, some_true_values)
    return p_value


def hyperparameter_estimation(prior, alpha, penalty, true_distribution, noise):
    # Testing data initialization
    num_points = 4000
    bins = 50
    true_data = true_distribution.get_sample(num_points)
    testing_data = true_data # + noise_data

    # Plot true data
    x_points = np.linspace(np.min(true_data) - 5, np.max(true_data) + 5, 3 * bins)
    y_points = np.exp(true_distribution.eval(x_points))
    plt.plot(x_points, y_points, 'b', lw=2)
    #plt.hist(true_data, bins, normed=True, alpha=0.5)
    # End of testing data initialization

    prior_freq = 1.
    run_count = 0
    learned = GaussianMixture()
    print("doing...", end='')
    sys.stdout.flush()
    samples = None
    best_score = -np.inf
    bucket_size = 10
    bucket = [None] * bucket_size
    n_prior = 0
    rejections = 0

    '''

    while run_count < 4000:
        use_prior = np.random.rand() <= prior_freq
        if use_prior:
            for i in xrange(bucket_size):
                value = prior.get_sample()
                score = get_score(value)
                if score > best_score:
                    best_score = score
                if np.random.rand() <= score / best_score:
                    if samples is None:
                        samples = [value]
                    else:
                        samples = np.r_[samples, [value]]
                else:
                    rejections += 1
                learned.combine(value)
        else:
            for i in xrange(bucket_size):
                value = learned.get_sample()
                score = get_score(value)
                if score > best_score:
                    best_score = score
                if np.random.rand() <= score / (2 * best_score):
                    if samples is None:
                        samples = [value]
                    else:
                        samples = np.r_[samples, [value]]
                bucket[i] = value
            p_val = get_fit(bucket, learned)
            if p_val >= alpha:
                prior_freq = max(prior_freq - alpha * p_val, 0.3)
            else:
                prior_freq = min(prior_freq + alpha * p_val, 1.)
            learned.combine(bucket)
        run_count += bucket_size
        print(run_count)
        print(prior_freq)

    '''
    while run_count < 12000:
        use_prior = np.random.rand() <= prior_freq or True
        n_prior += use_prior
        distribution = prior if use_prior else learned
        value = distribution.get_sample()
        if type(value) is np.ndarray:
            value = value.item()
        score = get_score(value)
        if score > best_score:
            best_score = score
        if np.random.rand() <= score / best_score or distribution == learned:
            if samples is None:
                samples = [value]
            else:
                samples = np.r_[samples, [value]]
            learned.combine(value)
        else:
            rejections += 1
        bucket[run_count % bucket_size] = value
        if run_count > 0 and run_count % bucket_size == 0:
            p_val = get_fit(bucket, learned)
            if p_val >= alpha:
                prior_freq = max(prior_freq - alpha * p_val, 0.3)
            else:
                prior_freq = min(prior_freq + alpha * p_val, 1.)
            #if good_fit(bucket, learned):
            #    prior_freq = max(prior_freq - penalty, 0.3)
            #else:
            #    prior_freq = min(prior_freq + penalty, 1.)
            print(run_count, prior_freq)
        run_count += 1


    print("done.")
    print("prior used {0} times".format(n_prior))
    print("{0} rejections".format(rejections))
    print("model complexity: {0}".format(learned.gmm.n_components))

    # plt.show(block=True)
    plt.hist(samples, bins, normed=True, alpha=0.6)
    #plt.show(block=True)
    # Plot learned distribution
    x_points = np.linspace(-20, 20, 3 * bins)
    from sklearn.mixture import GMM
    y_points = np.exp(learned.eval(x_points))
    plt.plot(x_points, y_points)
    plt.title("Learned distribution")
    #plt.hist(true_data, bins, normed=True, alpha=0.5)
    plt.show(block=True)
    # End of testing data initialization
    # plt.scatter(samples, np.arange(len(samples)))
    # plt.hist(samples, normed=True)
    # plt.show(block=True)

if __name__ == '__main__':
    hyperparam_prior = Uniform(-25, 25)
    penalty = 0.008
    hyperparameter_estimation(hyperparam_prior, alpha, penalty, true_distribution, noise)
