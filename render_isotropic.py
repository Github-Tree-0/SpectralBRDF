import os
import sys
import argparse
import cv2
import numpy as np
import shutil
from tqdm import tqdm
from joblib import Parallel, delayed

import SpecBRDF
from scipy.spatial.transform import Rotation as R_trans

def render(ii, brdf, n, L, v, n_lambda):
    cos_v = n @ v.T
    sin_v = np.sqrt(np.clip(1-cos_v**2, 0, 1))
    wo = np.array([0, sin_v, cos_v])
    if sin_v < 1e-6:
        # 使用叉乘产生一个和n垂直的向量
        temp_vec = n + [1, 0, 0]
        temp_vec = temp_vec / np.sqrt(np.sum(temp_vec**2))
        j = np.cross(n, temp_vec)
        j = j / np.sqrt(np.sum(j**2))
    else:
        j = (v - cos_v*n) / sin_v
    i = np.cross(j, n)
    i = i / np.sqrt(np.sum(i**2))

    ret = np.zeros([len(L), n_lambda], dtype=float)
    for index in range(len(L)):
        l = L[index]
        wi = np.array([l @ i.T, l @ j.T, l @ n.T])

        rho = np.array(brdf.eval(wo, wi), dtype=float)
        rho[rho < 0] = 0
        ret[index] = rho

    return ii, ret


def main(brdf_dir, obj_file, N_map_file, mask_file, L_file, out_dir, n_jobs, min_obj, max_obj):
    obj_names = np.loadtxt(obj_file, dtype=np.str)[min_obj:max_obj]
    N_map = np.load(N_map_file)
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

            brdf = SpecBRDF.SpecBRDFClass(os.path.join(brdf_dir, obj_name + '_spec.bsdf'))
            wave_lambda = brdf.getWavelength()
            n_lambda = len(wave_lambda)

            ret = Parallel(n_jobs=n_jobs, verbose=5, prefer='threads')([delayed(render)(i, brdf, N[i], L, v, n_lambda) for i in range(len(N))])
            ret.sort(key=lambda x: x[0])
            M = np.array([x[1] for x in ret], dtype=np.float)

            imgs = np.zeros((len(L),N_map.shape[0], N_map.shape[1], n_lambda ))
            imgs[:, mask > 0] = M.transpose(1, 0, 2)

            print('Saving images...')
            np.save(os.path.join(out_path, 'imgs.npy'), imgs)
            np.save(os.path.join(out_path, 'wave_lambda.npy'), wave_lambda)
            np.save(os.path.join(out_path, 'normal.npy'), N_map)
            np.save(os.path.join(out_path, 'mask.npy'), mask)
            shutil.copyfile(L_file, os.path.join(out_path, 'light_directions.txt'))

            print('===== {} - {} done ====='.format(i_obj, obj_name))


if __name__ == '__main__':
    mt_type = 'isotropic'

    parser = argparse.ArgumentParser()
    parser.add_argument('--brdf_dir', type=str, required=False, default='./brdf_data/{}/'.format(mt_type)) # ok
    parser.add_argument('--obj_file', type=str, required=False, default='./supp_info/obj_isotropic.txt')              # ok
    parser.add_argument('--N_map_file', type=str, required=False, default='./supp_info/sphere_normal_100.npy')  # ok
    parser.add_argument('--mask_file', type=str, required=False, default='./supp_info/sphere_mask_100.png')     # ok
    parser.add_argument('--L_file', type=str, required=False, default='./supp_info/L_32.txt')                                  # ok
    parser.add_argument('--out_dir', type=str, required=False, default='/mnt/data/kedaxiaoqiu_data/mitsuba_verify')
    parser.add_argument('--min', type=int, default=0)
    parser.add_argument('--max', type=int, default=100)
    parser.add_argument('--n_jobs', type=int, default=128)
    args = parser.parse_args()

    main(brdf_dir=args.brdf_dir,
        obj_file=args.obj_file,
        N_map_file=args.N_map_file,
        mask_file=args.mask_file,
        L_file=args.L_file,
        out_dir=args.out_dir,
        n_jobs=args.n_jobs,
        min_obj=args.min,
        max_obj=args.max)
