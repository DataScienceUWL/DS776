import numpy as np

def create_single_spiral( n, std ):

    theta = np.random.uniform( 0, 3*np.pi, n)
    radius = 5*theta/(3*np.pi)
    xx = radius * np.cos(theta) + np.random.normal(0,std,n)
    yy = radius * np.sin(theta) + np.random.normal(0,std,n)
    X = np.transpose(np.stack( [xx, yy]))
    return X

def make_spirals(n_samples = 400, noise = 35, random_state = 42):

    noise = np.clip( noise, 0, 50)
    std = 0.4 * noise/50

    n0 = n_samples // 2
    n1 = n_samples - n0

    np.random.seed(random_state)

    X0 = create_single_spiral( n0, std)
    X1 = -create_single_spiral( n1, std) # rotate 180

    X = np.concatenate([X0, X1], axis = 0)
    y = np.concatenate( [np.zeros(n0), np.ones(n1)] ).astype(int)

    return X,y