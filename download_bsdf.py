import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import subprocess
import numpy as np

URL = "http://rgl.s3.eu-central-1.amazonaws.com/media/materials/"


def main(obj_file, out_dir):
    obj_names = np.loadtxt(obj_file, dtype=np.str)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for obj_name in obj_names:
        if os.path.exists(os.path.join(out_dir, obj_name+'_rgb.bsdf')):
            print(obj_name, ' exists.')
            continue
        subprocess.call(['wget', '-P', out_dir, URL+obj_name+'/'+obj_name+'_rgb.bsdf'])

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_file', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()

    main(args.obj_file, args.out_dir)
