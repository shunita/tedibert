import os

if __name__ == '__main__':
    os.environ['LANG'] = 'C.UTF-8'
    # os.system('srun -p cs -A public --gres=gpu:1 /home/shunita/miniconda3/envs/fairemb/bin/python '
    #           '/home/shunita/fairemb/contra/experimental/experimental_torch.py')
    os.system('srun --gres=gpu:1 /home/shunita/miniconda3/envs/fairemb/bin/python '
              '/home/shunita/fairemb/contra/experimental/experimental_torch.py')

