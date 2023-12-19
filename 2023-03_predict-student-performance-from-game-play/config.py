

import socket
if socket.gethostname().startswith('wx2014'):
    PATH_INPUT = '/home/datashare/datasets/kaggle/predict-student-performance-from-game-play/input'
    PATH_WORKING = '/home/datashare/datasets/kaggle/predict-student-performance-from-game-play/working'
    
elif socket.gethostname().startswith('wx2186'):
    PATH_INPUT = '/home/share/datasets/kaggle/predict-student-performance-from-game-play/input'
    PATH_WORKING = '/home/share/datasets/kaggle/predict-student-performance-from-game-play/working'

SEED = 4243