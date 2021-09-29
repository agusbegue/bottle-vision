# install packages
echo "Installing required packages..."
sudo apt update
sudo apt install python3-pip -y
sudo apt-get install python3-venv -y
sudo apt install unzip

# virtual environment
echo "Setting up virtual environment"
python3 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip

# libraries
echo "Installing required libraries"
pip3 install --ignore-installed --upgrade tensorflow==2.4.1 --no-cache-dir
sudo apt-get install -y python3-opencv
pip3 install -r requirements.txt

# download classification model
echo "Downloading classification model"
gdown https://drive.google.com/uc?id=1-G-y0Z2QmXCnUHMliYOAPThjyfbMFT1- -O models/classifier/classifier.zip
unzip models/classifier/classifier.zip -d models/classifier
mv models/classifier/model_v3 models/classifier/1
rm classifier.zip


#download object detection model
echo "Downloading object detection model"
# get models and support files for tf
git clone https://github.com/tensorflow/models.git models/models
# downloading and compiling protobuf libraries to configure models
sudo apt install protobuf-compiler -y
cd models/models/research
protoc object_detection/protos/*.proto --python_out=.
# install pycocoapi first
sudo apt-get install python3-dev gcc
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools ${PWD}/models/models/research/
# install dependencies
cp ${PWD}/models/models/research/object_detection/packages/tf2/setup.py ${PWD}/models/models/research/
python -m pip install ${PWD}/models/models/research
# mark directory as root
export PYTHONPATH="${PYTHONPATH}:${PWD}"
# download model and labels
wget -P models/od_model/ http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz
tar -xvzf models/od_model/efficientdet_d0_coco17_tpu-32.tar.gz -C models/od_model/
wget -P models/od_model/ https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt
# move model to version 1 of model (for tf serving)
mkdir models/od_model/efficientdet_d0_coco17_tpu-32/saved_model/1/
mv models/od_model/efficientdet_d0_coco17_tpu-32/saved_model/!(1) models/od_model/efficientdet_d0_coco17_tpu-32/saved_model/1/

# install docker and model serving
echo "Installing docker and tf model serving"
sudo apt install docker.io -y
sudo docker pull tensorflow/serving:latest-gpu
mkdir models/serving
sudo git clone https://github.com/tensorflow/serving models/serving/


