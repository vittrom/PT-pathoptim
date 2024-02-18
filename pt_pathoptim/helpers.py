import argparse
import os

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def createFolders(foldername):
    # Create folders for saving if not existing
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    if not  os.path.exists(os.path.join(foldername, "path")):
        os.makedirs(os.path.join(foldername, "path"))
    if not os.path.exists(os.path.join(foldername, "path_log")):
        os.makedirs(os.path.join(foldername, "path_log"))
    if not os.path.exists(os.path.join(foldername, "Figures")):
        os.makedirs(os.path.join(foldername, "Figures"))