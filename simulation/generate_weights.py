import numpy as np
from scipy.linalg import block_diag
from spectral_density import *


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
    assert p in [12, 48, 96]
    if gen_mode == 'ma':
        block = generate_upper_block(6, 0.4, -0.5)
    elif gen_mode == 'var':
        block = generate_upper_block(6, 0.2, 0.1)
    if p == 12:
        ls = [block, block]
        return block_diag(*ls)
    if p == 48:
        ls = []
        for _ in range(4):
            ls.append(generate_weights_homo(12, gen_mode))
        return  block_diag(*ls)
    if p == 96:
        return block_diag(generate_weights_homo(48, gen_mode), generate_weights_homo(48, gen_mode))

def generate_weights_heter1(p, gen_mode):
    if gen_mode == 'ma':
        block1 = generate_block_diagnal(int(2*p/3), 0.4)
        block2 = generate_upper_block(int(p/3), 0.4, -0.5)
    elif gen_mode == 'var':
        block1 = generate_upper_block(int(2 * p / 3), 0.1, 0.1)
        block2 = generate_upper_block(int(p / 3), 0.1, -0.4)
    ls = [block1, block2]
    return block_diag(*ls)


def fetch_weights(p, mode, gen_mode):
    identity = np.diag(np.repeat(1, p))
    if mode == 'ho':
        if gen_mode == 'ma':
            return [identity, generate_weights_homo(p, gen_mode)]
        elif gen_mode == 'var':
            return [generate_weights_homo(p, gen_mode)]
    elif mode == 'he1':
        if gen_mode == 'ma':
            return [identity, generate_weights_heter1(p, gen_mode)]
        elif gen_mode == 'var':
            return [generate_weights_heter1(p, gen_mode)]



def test_generate_hom_weights():
    A1 = fetch_ma_weights(12, 'ho')
    A2 = fetch_ma_weights(48, 'he1')
    A3 = fetch_ma_weights(96, 'he2')

    print(A1[0])
    print(A1[1])
    print(A3[1][12:24, 12:24])


if __name__ == "__main__":
    print(generate_ma_weights_heter3(12))
