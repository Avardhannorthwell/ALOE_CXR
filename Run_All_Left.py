import os

#print('running remaining in chexnet right none...')
# # check for reproducibility
os.system('PYTHONHASHSEED=0 python Final_RESNET_052022.py --lung left --sampling over --fold 1 --segment none')
os.system('PYTHONHASHSEED=0 python Final_RESNET_052022.py --lung right --sampling over --fold 1 --segment none')

os.system('PYTHONHASHSEED=0 python Final_RESNET_052022.py --lung left --sampling over --fold 1 --segment spine')
os.system('PYTHONHASHSEED=0 python Final_RESNET_052022.py --lung right --sampling over --fold 1 --segment spine')

os.system('PYTHONHASHSEED=0 python Final_RESNET_052022.py --lung left --sampling over --fold 1 --segment lung')
os.system('PYTHONHASHSEED=0 python Final_RESNET_052022.py --lung right --sampling over --fold 1 --segment lung')
