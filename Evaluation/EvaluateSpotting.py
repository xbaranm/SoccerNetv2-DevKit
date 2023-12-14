from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from SoccerNet.Evaluation.ActionSpotting import evaluate


DATASET_PATH =  '/workspace/mysocnet/.mnt/scratch/dataset'
SPLIT = 'test' # 'test/challenge'
METRIC = 'tight'        # 'loose'/'tight'

PREDICTIONS_PATH = f'/workspace/mysocnet/src/Task1-ActionSpotting/TemporallyAwarePooling/src/models/NetVLAD++/results_spotting_{SPLIT}.zip'
# CHALLENGE_PREDICTIONS_PATH = '/workspace/mysocnet/src/Task1-ActionSpotting/TemporallyAwarePooling/src/models/NetVLAD++/results_spotting_challenge.zip'


if __name__ == '__main__':

    # Load the arguments
    parser = ArgumentParser(description='Evaluation for Action Spotting', 
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--SoccerNet_path', required=False,
                        type=str, help='Path to the SoccerNet-V2 dataset folder (or zipped file) with labels', default=DATASET_PATH)
    parser.add_argument('--Predictions_path', required=False,
                        type=str, help='Path to the predictions folder (or zipped file) with prediction', default=PREDICTIONS_PATH)
    parser.add_argument('--Prediction_file', required=False, type=str,
                        help='Name of the prediction files as stored in folder (or zipped file) [None=try to infer it]', default=None)
    parser.add_argument('--split', required=False, type=str,
                        help='Set on which to evaluate the performances', default="test")
    parser.add_argument('--version', required=False, type=int,
                        help='Version of SoccerNet [1,2]', default=2)
    parser.add_argument('--metric', required=False, type=str,
                        help='tight (tight average mAP 2022) or loose (average mAP 2021)', default=METRIC)

    args = parser.parse_args()

    # a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown 
    results = evaluate(SoccerNet_path=args.SoccerNet_path, Predictions_path=args.Predictions_path,
                       split=args.split, version=args.version, prediction_file=args.Prediction_file, metric=args.metric)

    print(f"{METRIC} Average mAP: ", results["a_mAP"])
    print(f"{METRIC} Average mAP per class: ", results["a_mAP_per_class"])
    print(f"{METRIC} Average mAP visible: ", results["a_mAP_visible"])
    print(f"{METRIC} Average mAP visible per class: ", results["a_mAP_per_class_visible"])
    print(f"{METRIC} Average mAP unshown: ", results["a_mAP_unshown"])
    print(f"{METRIC} Average mAP unshown per class: ", results["a_mAP_per_class_unshown"])

# python tools/EvaluateSpotting.py --Predictions_path EvalAI/submission/results_spotting/ --SoccerNet_path /media/giancos/Football/SoccerNet/ --Prediction_file Predictions-v2.json
# python Evaluation/EvaluateSpotting.py --Predictions_path /home/giancos/git/SoccerNetv2/EvalAI/submission/results_spotting_test.zip --SoccerNet_path /home/giancos/git/SoccerNetv2/EvalAI/annotations/test_annotations_spotting.zip --Prediction_file Predictions-v2.json
