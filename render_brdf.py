import os
import sys
import argparse
import cv2
import numpy as np
import shutil
import itertools
from tqdm import tqdm
from joblib import Parallel, delayed

import SpecBRDF
from scipy.spatial.transform import Rotation as R_trans

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



def render(i, brdf, wi, wo):
    rho = np.array(brdf.eval(wo, wi), dtype=float)
    rho[rho < 0] = 0

    return i, rho


def main(wis, wos, brdf_dir, obj_file, out_dir, n_jobs, min_obj, max_obj):
    obj_names = np.loadtxt(obj_file, dtype=np.str)[min_obj:max_obj]

    w_shape = wis.shape

    for i_obj, obj_name in enumerate(obj_names):

        out_path = os.path.join(out_dir, obj_name)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        print('===== {} - {} start ====='.format(i_obj, obj_name))

        brdf = SpecBRDF.SpecBRDFClass(os.path.join(brdf_dir, obj_name + '_spec.bsdf'))
        wave_lambda = brdf.getWavelength()
        n_lambda = len(wave_lambda)

        ret = Parallel(n_jobs=n_jobs, verbose=5, prefer='threads')([delayed(render)(i, brdf, wi, wo) for i, (wi, wo) in enumerate(itertools.product(wis, wos))])
        ret.sort(key=lambda x: x[0])
        M = np.array([x[1] for x in ret], dtype=np.float)

        imgs = M.reshape(w_shape[0], w_shape[0], n_lambda)

        print('Saving images...')
        np.save(os.path.join(out_path, 'imgs.npy'), imgs)
        np.save(os.path.join(out_path, 'wave_lambda.npy'), wave_lambda)

        print('===== {} - {} done ====='.format(i_obj, obj_name))


if __name__ == '__main__':
    mt_type = 'isotropic'

    parser = argparse.ArgumentParser()
    parser.add_argument('--brdf_dir', type=str, required=False, default='../SpectralBRDF_trans/brdf_data/{}/'.format(mt_type)) # ok
    parser.add_argument('--obj_file', type=str, required=False, default='./supp_info/obj_paper_blue.txt')              # ok
    parser.add_argument('--out_dir', type=str, required=False, default='/mnt/data/kedaxiaoqiu_data/brdf_out')
    parser.add_argument('--min', type=int, default=0)
    parser.add_argument('--max', type=int, default=51)
    parser.add_argument('--n_jobs', type=int, default=128)
    args = parser.parse_args()

# vecs, brdf_dir, obj_file, out_dir, n_jobs, min_obj, max_obj
    # wis = np.load('wis.npy')
    # wos = np.load('wos.npy')
    N_sample = 91
    vecs = []
    for i in range(N_sample):
        theta = np.pi / (2*N_sample) * i
        vec = [np.sin(theta), 0, np.cos(theta)]
        vecs.append(vec)
    vecs = np.array(vecs)

    main(wis=vecs,
        wos=vecs,
        brdf_dir=args.brdf_dir,
        obj_file=args.obj_file,
        out_dir=args.out_dir,
        n_jobs=args.n_jobs,
        min_obj=args.min,
        max_obj=args.max)
