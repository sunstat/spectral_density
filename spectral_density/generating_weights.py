import numpy as np
from scipy.linalg import block_diag
from spectral_density import *

p_values = [12,48,96]

def generate_upper_block(p, diag_val, off_set = 0.2):
    if p == 1:
        return np.diag(np.repeat(diag_val, p))
    block = np.diag(np.repeat(diag_val, p))
    for i in range(p-1):
        block[i, i+1] = diag_val - off_set
    return block


def generate_upper_block_full(p, diag_val, off_value):
    block = np.diag(np.repeat(diag_val, p))
    for i in range(p - 1):
        for j in range(p-1):
            if i != j:
                block[i, j] = off_value
    return block


def generate_lower_block(p, diag_val):
    if p == 1:
        return np.diag(np.repeat(diag_val, p))
    block = np.diag(np.repeat(diag_val, p))
    for i in range(1, p):
        block[i, i-1] = diag_val - 0.2
    return block


def generate_upper_block_plus(p, diag_val, off_value = 0.4):
    if p == 1:
        return np.diag(np.repeat(diag_val, p))
    block = np.diag(np.repeat(diag_val, p))
    for i in range(p-1):
        block[i, i+1] = diag_val + off_value
    return block



def generate_block_diagnal(p, diag_val):
    return np.diag(np.repeat(diag_val, p))



def generate_weights_homo(p, gen_mode):
    assert p in [3, 12, 48, 96]
    if p == 3:
        return generate_upper_block(3, 0.5, -0.4)
    if gen_mode == 'ma':
        block = generate_upper_block(3, 0.5, -0.4)
    elif gen_mode == 'var':
        block = generate_upper_block(3, 0.5, -0.4)
    if p == 12:
        ls = [block, block, block, block]
        return block_diag(*ls)
    if p == 48:
        ls = []
        for _ in range(4):
            ls.append(generate_weights_homo(12, gen_mode))
        return  block_diag(*ls)
    if p == 96:
        return block_diag(generate_weights_homo(48, gen_mode), generate_weights_homo(48, gen_mode))




def generate_weights_heter(p, gen_mode):
    block1 = generate_upper_block(p//3, 0.1, 0.1)
    block2 = generate_upper_block(2*p//3, 0.1, -0.4)
    blocks = [block1, block2]
    return block_diag(*blocks)


def fetch_weights(p, ho_heter_mode, gen_model):
    '''
    :param p: numer of parameters
    :param ho_heter_mode:  homogeneous or heterogeneous
    :param gen_model: ma or var
    :return:
    '''
    identity = np.diag(np.repeat(1, p))
    if ho_heter_mode == 'ho':
        if gen_model == 'ma':
            return [identity, generate_weights_homo(p, gen_model)]
        elif gen_model == 'var':
            return [generate_weights_homo(p, gen_model)]
    elif ho_heter_mode == 'he':
        if gen_model == 'ma':
            return [identity, generate_weights_heter(p, gen_model)]
        elif gen_model == 'var':
            return [generate_weights_heter(p, gen_model)]




if __name__ == "__main__":
    print(fetch_weights(p=12, ho_heter_mode = 'ho', gen_model = 'ma'))
