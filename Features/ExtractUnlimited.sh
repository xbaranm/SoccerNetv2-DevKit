for i in {0..100}
do
    echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
    echo "@ Starting extraction script number $i @"
    echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
    python ExtractResNET_TF2.py --back_end=efficientnet --features=ResNET --video LQ --transform crop --verbose --split all
done