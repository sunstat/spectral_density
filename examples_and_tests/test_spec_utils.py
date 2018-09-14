import os
import sys

dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print(os.path.join(dir_path, 'spectral_density', 'utils'))
wd = os.path.join(dir_path, 'spectral_density', 'utils')

sys.path.append(wd)


from utilities import  *

single_matrics = {-2:np.array([[1,2],[2,-1]]), -1:np.array([[1,2],[-2,-1]]),
                  0:np.array([[1,0],[0,1]]), 1:np.array([[2,0], [0,2]]), 2:np.array([[2,3],[2,3]])}


print(single_matrics)

print(smooth_matrices(single_matrics, list(single_matrics.keys()), 1))

smooth_matrices = smooth_matrices(single_matrics, list(single_matrics.keys()), 1)

smooth_matrices = smooth_diagonal_correct(single_matrics, list(single_matrics.keys()), [0,2], smooth_matrices)

print(smooth_matrices)