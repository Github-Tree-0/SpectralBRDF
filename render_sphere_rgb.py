import os
import sys
import argparse
import time

from numpy.lib.format import dtype_to_descr
import cv2
import numpy as np
import shutil
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R_trans

import RGBBRDF

def calc_rotate_angle(vec):
    """
    :param np.ndarray vec: (3, )
    :return: (theta, phi)
    """
    assert np.linalg.norm(vec) > 0

    theta = np.arccos(vec[2] / np.linalg.norm(vec))
    # Angle of xy to x
    if np.linalg.norm([vec[0], vec[1]]) == 0:
        phi = 0.0
    else:
        phi = np.sign(vec[1]) * np.arccos(vec[0] / np.linalg.norm([vec[0], vec[1]]))

    return -theta, -phi


def render(i, brdf, n, L, v):
    theta, phi = calc_rotate_angle(n)
    R = R_trans.from_euler('zyx', [phi, theta, 0], degrees=False).as_matrix()

    wo = (R @ v).tolist()

    ret = np.zeros([len(L), 3], dtype=float)
    for j in range(len(L)):
        wi = (R @ L[j]).tolist()

        rho = np.array(brdf.eval(wo, wi), dtype=float)
        rho[rho < 0] = 0
        ret[j] = rho

    return i, ret


def main(brdf_dir, obj_file, obj_num, N_map_file, mask_file, L_file, out_dir, n_jobs, n_direction, eps=0.001, obj_range=None):
    obj_names = np.loadtxt(obj_file, dtype=np.str)
    if obj_range != None:
        obj_names = obj_names[obj_range[0]:obj_range[1]]

    if obj_num != None:
        obj_index = np.random.choice(np.arange(0, len(obj_names), dtype=np.int), obj_num, replace=False)
        obj_names = obj_names[obj_index]
    N_map = np.load(N_map_file) # Depends on the input data
    mask = cv2.imread(mask_file, 0)
    N = N_map[mask > 0] # (P, 3)
    N = N / np.linalg.norm(N, axis=1, keepdims=True)

    L = np.loadtxt(L_file) # (L, 3)
    L = L / np.linalg.norm(L, axis=1, keepdims=True)
    v = np.array([0., 0., 1.], dtype=np.float)

    for i_obj, obj_name in enumerate(obj_names):

        out_path = os.path.join(out_dir, obj_name)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            print('===== {} - {} start ====='.format(i_obj, obj_name))

            brdf = RGBBRDF.RGBBRDFClass(os.path.join(brdf_dir, obj_name + '_spec.bsdf'))
            wave_lambda = brdf.getWavelength()

            wave_index = np.random.choice(np.arange(0, len(wave_lambda), dtype=np.int), n_direction, replace=False)
            L_index = np.random.choice(np.arange(0, len(L), dtype=np.int), n_direction, replace=False)
            L_ = L

            ret = Parallel(n_jobs=n_jobs, verbose=5, prefer='threads')([delayed(render)(i, brdf, N[i], L_, v) for i in range(len(N))])
            ret.sort(key=lambda x: x[0])
            M = np.array([x[1] for x in ret], dtype=np.float)

            # imgs 需要塑形成input
            imgs = np.zeros((n_direction, N_map.shape[0], N_map.shape[1], 3))
            imgs[:, mask > 0] = M.transpose(1, 0, 2)

            wave_lambda_ = wave_lambda[wave_index]

            print('Saving images...')
            np.save(os.path.join(out_path, 'wave_lambda.npy'), wave_lambda_)
            np.save(os.path.join(out_path, 'normal.npy'), N_map)
            np.save(os.path.join(out_path, 'imgs.npy'), imgs)
            np.savetxt(os.path.join(out_path, 'light_directions.txt'), L_)
            np.save(os.path.join(out_path, 'mask.npy'), mask)

            # names = []
            # for i in range(n_direction):
            #     filename = 'l_%03d.png' % i
            #     names.append(filename)
            #     imsave(os.path.join(out_path, filename), np.transpose(np.tile(imgs[i], (3, 1, 1)), (1, 2, 0)))
            
            # np.savetxt(os.path.join(out_path, 'names.txt'), names, fmt="%s", delimiter="\n")
            # np.save(os.path.join(out_path, 'wave_lambda.npy'), wave_lambda_)
            # imsave(os.path.join(out_path, 'normal.png'), (N_map+1.0)/2.0)
            # np.savetxt(os.path.join(out_path, 'light_directions.txt'), L_)
            # imsave(os.path.join(out_path, 'mask.png'), mask)
            # np.savetxt(os.path.join(out_path, 'light_intensity.txt'), light_intensities)

            print('===== {} - {} done ====='.format(i_obj, obj_name))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--min", type=int, default=0)
    parser.add_argument("--max", type=int, default=3)
    parser.add_argument("--out", type=str, default='../bunny_and_ball_rgb/ball')
    parser.add_argument("--normal", type=str, default='supp_info/ball_normal.npy')
    parser.add_argument("--mask", type=str, default='supp_info/ball_mask.png')
    parser.add_argument("--obj_min", type=int, default=0)
    parser.add_argument("--obj_max", type=int, default=100)
    args = parser.parse_args()

    brdf_dir = './brdf_data/isotropic_39/'
    obj_file = './supp_info/obj_isotropic_ex.txt' # specify material
    out_file = args.out
    n_jobs = 128

    N_map_file = args.normal
    mask_file = args.mask
    out_dir = args.out
    L_file = 'supp_info/L_32.txt'
    obj_range = [args.obj_min, args.obj_max]

    main(brdf_dir=brdf_dir,
        obj_file=obj_file,
        obj_num=None,
        N_map_file=N_map_file,
        mask_file=mask_file,
        L_file=L_file,
        out_dir=out_dir,
        n_jobs=n_jobs,
        n_direction=32,
        obj_range=obj_range)

    # objects = os.listdir(out_file)
    # np.savetxt(out_file+'/objects.txt', objects, fmt="%s", delimiter="\n")
