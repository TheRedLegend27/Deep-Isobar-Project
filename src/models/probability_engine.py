from scipy.stats import norm


def probability_above_threshold(mean, std, threshold):

    probability = 1 - norm.cdf(threshold, mean, std)

    return probability