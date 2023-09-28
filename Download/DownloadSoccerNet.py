import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(
    LocalDirectory="/workspace/mysocnet/.mnt/scratch/dataset")

mySoccerNetDownloader.password = input("Password for videos?:\n")

mySoccerNetDownloader.downloadGames(files=["1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy"], split=[
                                    "train", "valid", "test", "challenge"])  # download Features
