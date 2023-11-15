import argparse
import os 
import SoccerNet



import configparser
import math
import os
# import argparse
import numpy as np
import cv2  # pip install opencv-python (==3.4.11.41)
import imutils  # pip install imutils
import skvideo.io
from tqdm import tqdm

import json

import random
from SoccerNet.utils import getListGames
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.DataLoader import Frame, FrameCV

############################################################################
import psutil # To monitor RAM usage
############################################################################
from torch.utils.data import DataLoader
from pytorch_lightning.trainer import Trainer
### Import model here ###
import timm
from EfficientNet import EfficientNet
from OUR_EfficientNet import OUR_EfficientNet
### Import dataset here ###
from SoccernetDataset import SoccernetDataset
############################################################################

CHECKPOINT_NAME =   'OUR_EfficientNet/OUR_EfficientNet-epoch=204-valid_loss_epoch=0.265.ckpt'
CHECKPOINT_PATH =   '/workspace/mysocnet/.mnt/scratch/models/'
DATASET_PATH =      '/workspace/mysocnet/.mnt/dataset/'
FEATURES_PATH =     '/workspace/mysocnet/.mnt/scratch/dataset/'


def CheckMemoryUsage():
    ramUsage = psutil.virtual_memory()[2]
    print(f"***RAM memory % used: {psutil.virtual_memory()[2]}***")
    if ramUsage >= 75 :
        print("***RAM usage exceeded 75%... Exiting program***")
        exit()

class FeatureExtractor():
    def __init__(self, rootFolder,
                 feature="ResNET",
                 video="LQ",
                 back_end="whole_yf_efficientnet",
                 overwrite=False,
                 transform="crop",
                 tmp_HQ_videos=None,
                 grabber="opencv",
                 FPS=2.0,
                 split="all"):
        self.rootFolder = rootFolder
        self.feature = feature
        self.video = video
        self.back_end = back_end
        self.verbose = True
        self.transform = transform
        self.overwrite = overwrite
        self.grabber = grabber
        self.FPS = FPS
        self.split = split

        self.tmp_HQ_videos = tmp_HQ_videos
        if self.tmp_HQ_videos:
            self.mySoccerNetDownloader = SoccerNetDownloader(self.rootFolder)
            self.mySoccerNetDownloader.password = self.tmp_HQ_videos

        # self.model = YFEfficientNet.load_from_checkpoint(os.path.join(CHECKPOINT_PATH, CHECKPOINT_NAME), output="unpooled_no_classifier")
        self.model = OUR_EfficientNet.load_from_checkpoint(os.path.join(CHECKPOINT_PATH, CHECKPOINT_NAME), output="features")
        # self.model = YFEfficientNet(output="unpooled")



    def extractAllGames(self):
        list_game = getListGames(self.split)
        for i_game, game in enumerate(tqdm(list_game)):
            try:
                CheckMemoryUsage()

                self.extractGameIndex(i_game)
            except Exception as e:
                print(f"*issue with game {i_game}, {game}")
                print(e)

    def extractGameIndex(self, index):
        print(getListGames(self.split)[index])
        if self.video =="LQ":
            for vid in ["1_720p.mkv","2_720p.mkv"]:
                self.extract(video_path=os.path.join(self.rootFolder, getListGames(self.split)[index], vid), index=getListGames(self.split)[index], vid=vid)

        elif self.video == "HQ":
            
            # read config for raw HD video
            config = configparser.ConfigParser()
            config.read(os.path.join(self.rootFolder, getListGames(self.split)[index], "video.ini"))

            # lopp over videos
            for vid in config.sections():
                vid = vid.replace("HQ", "720p")
                video_path = os.path.join(self.rootFolder, getListGames(self.split)[index], vid)

                # cehck if already exists, then skip
                feature_path = video_path[:-4] + f"_{self.feature}_{self.back_end}.npy"
                if os.path.exists(feature_path) and not self.overwrite:
                    print("already exists, early skip")
                    continue

                # extract feature for video
                self.extract(video_path=video_path,
                            start=float(config[vid]["start_time_second"]), 
                            duration=float(config[vid]["duration_second"]))
                

    def extract(self, video_path, start=None, duration=None, index=None, vid=None):
        print("extract video", video_path, "from", start, duration)
        feature_path = os.path.join(FEATURES_PATH, index, vid)[:-9] + f"_{self.feature}_{self.back_end}.npy"
        frames_path = os.path.join(FEATURES_PATH, index, vid)[:-9] + f"_frames.npy"

        if os.path.exists(feature_path) and not self.overwrite:
            return
      
        try:
            # First try to load precomputed frames
            try:
                print(f"*Trying to load frames: {frames_path}")
                os.makedirs(os.path.dirname(frames_path), exist_ok=True)
                # os.makedirs(os.path.dirname(new_frames_path), exist_ok=True)
                frames = np.load(frames_path)
                print(f"*...frames loaded")
            except:
                print("*No frames found. Computing from scratch")
                if self.grabber=="skvideo":
                    videoLoader = Frame(video_path, FPS=self.FPS, transform=self.transform, start=start, duration=duration)
                elif self.grabber=="opencv":
                    videoLoader = FrameCV(video_path, FPS=self.FPS, transform=self.transform, start=start, duration=duration)
                frames = videoLoader.frames

                if duration is None:
                    duration = videoLoader.time_second
                if self.verbose:
                    print("frames", frames.shape, "fps=", frames.shape[0]/duration)

                # Store frames so they don't need to be computed everytime
                print(f"*Storing computed frames to: {frames_path}")
                np.save(frames_path, np.array(frames))

            # return # TODO: delete me
            dataset = SoccernetDataset(frames)
            data_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=False)
            trainer = Trainer()
            features = trainer.predict(self.model, data_loader)
            print(f"*features dim before concatenate: {np.shape(features)}")
            features = np.concatenate(features, axis=0)
            print(f"*features dim after concatenate: {np.shape(features)}")
        except Exception as e:
            print(f"[Extracting features Exception] {e}")
        
        try:
            # save the featrue in .npy format
            os.makedirs(os.path.dirname(feature_path), exist_ok=True)
            np.save(feature_path, features)
        except Exception as e:
            print(f"[Saving exception] {e}")


        # save the featrue in .npy format
        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        np.save(feature_path, features)



if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(
        description='Extract ResNet feature out of SoccerNet Videos.')

    parser.add_argument('--soccernet_dirpath', type=str, default=DATASET_PATH,
                        help=f"Path for SoccerNet directory [default:{DATASET_PATH}]")
                        
    parser.add_argument('--overwrite', action="store_true",
                        help="Overwrite the features? [default:False]")
    parser.add_argument('--GPU', type=int, default=0,
                        help="ID of the GPU to use [default:0]")
    parser.add_argument('--verbose', action="store_true",
                        help="Print verbose? [default:False]")
    parser.add_argument('--game_ID', type=int, default=None,
                        help="ID of the game from which to extract features. If set to None, then loop over all games. [default:None]")

    # feature setup
    parser.add_argument('--back_end', type=str, default="our_yf_efficientnet",
                        help="Backend OUR_EfficientNet [default:our_yf_efficientnet]")
    parser.add_argument('--features', type=str, default="ResNET",
                        help="ResNET or R25D [default:ResNET]")
    parser.add_argument('--transform', type=str, default="crop",
                        help="crop or resize? [default:crop]")
    parser.add_argument('--video', type=str, default="LQ",
                        help="LQ or HQ? [default:LQ]")
    parser.add_argument('--grabber', type=str, default="opencv",
                        help="skvideo or opencv? [default:opencv]")
    parser.add_argument('--tmp_HQ_videos', type=str, default=None,
                        help="enter pawssword to download and store temporally the videos [default:None]")
    parser.add_argument('--FPS', type=float, default=2.0,
                        help="FPS for the features [default:2.0]")
    parser.add_argument('--split', type=str, default="all",
                        help="split of videos from soccernet [default:all]")

    args = parser.parse_args()
    print(args)

    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    myFeatureExtractor = FeatureExtractor(
        args.soccernet_dirpath, 
        feature=args.features,
        video=args.video,
        back_end=args.back_end, 
        # array=args.array,
        transform=args.transform,
        # preprocess=True,
        tmp_HQ_videos=args.tmp_HQ_videos,
        grabber=args.grabber,
        FPS=args.FPS,
        split=args.split)
    myFeatureExtractor.overwrite= args.overwrite

    # def extractGameIndex(self, index):
    if args.game_ID is None:
        myFeatureExtractor.extractAllGames()
    else:
        myFeatureExtractor.extractGameIndex(args.game_ID)
'''
python Features/ExtractResNET_TF2.py --back_end=efficientnet 
                                    --features=ResNET --video LQ --transform crop --verbose --split all --overwrite                                   
'''