import os

for i in range(11):
    min_num = i * 5
    max_num = (i+1) * 5
    command = 'nohup python render_isotropic.py --min {} --max {} > logs/{}.log 2>&1 &'.format(min_num, max_num, i)
    if not os.system(command):
        print('{} success!'.format(i))