# Extract Features fro SoccerNet-v2

## Conda environment

``` bash
conda create -n SoccerNet-FeatureExtraction python=3.7
conda install cudnn cudatoolkit=10.1
pip install scikit-video tensorflow==2.3.0 imutils opencv-python==3.4.11.41
```

## Extract ResNET features for all 550 games (500 + 50 challenge)

```bash
python tools/ExtractResNET_TF2.py --soccernet_dirpath /path/to/SoccerNet/ --back_end=TF2 --features=ResNET --video LQ --transform crop --verbose --split all
```

## Reduce features for all 550 games (500 games to estimate PCA + 50 challenge games for inference)

```bash
python tools/ReduceFeaturesPCA.py --soccernet_dirpath /path/to/SoccerNet/
```