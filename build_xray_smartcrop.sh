rm -r data/xray_smartcrop/train/
rm -r data/dxray_smartcrop/val/
rm -r data/xray_smartcrop/test/

mkdir data/xray_smartcrop/train
mkdir data/xray_smartcrop/train/0
mkdir data/xray_smartcrop/train/1
mkdir data/xray_smartcrop/train/2
mkdir data/xray_smartcrop/train/3
mkdir data/xray_smartcrop/train/4

mkdir data/xray_smartcrop/val
mkdir data/xray_smartcrop/val/0
mkdir data/xray_smartcrop/val/1
mkdir data/xray_smartcrop/val/2
mkdir data/xray_smartcrop/val/3
mkdir data/xray_smartcrop/val/4

mkdir data/xray_smartcrop/test
mkdir data/xray_smartcrop/test/0
mkdir data/xray_smartcrop/test/1
mkdir data/xray_smartcrop/test/2
mkdir data/xray_smartcrop/test/3
mkdir data/xray_smartcrop/test/4

source ~/.venv2/bin/activate
python3 build_xray_smartcrop.py $1 $2 $3
deactivate
