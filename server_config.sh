echo "Configuring server and and installing dependencies"
# install packages
sudo apt update
sudo apt install python3-pip
sudo apt-get install python3-venv
# virtual environment
mkdir bottle_cv && cd bottle_cv
python3 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip
# install tensorflow
pip3 install --ignore-installed --upgrade tensorflow==2.4.1 --no-cache-dir
# get models and support files for tf
git clone https://github.com/tensorflow/models.git
# downloading and compiling protobuf libraries to configure models
sudo apt install protobuf-compiler
cd models/research
protoc object_detection/protos/*.proto --python_out=.
# install pycocoapi first
sudo apt-get install python3-dev gcc
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools /home/ubuntu/bottle_cv/models/research/
# install dependencies
cd ../..
cp object_detection/packages/tf2/setup.py .
python -m pip install .
# mark directory as root
export PYTHONPATH="${PYTHONPATH}:/home/ubuntu/bottle_cv/"
# download model and labels
cd ../../
wget -P cv/od_model/ http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz
tar -xvzf cv/od_model/efficientdet_d0_coco17_tpu-32.tar.gz -C cv/od_model/
wget -P cv/od_model/ https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt
# copy object detection utils in source directory
#cp models/research/object_detection . -r (no se usa creo)
# move model to version 1 of model (for tf serving)
mkdir cv/od_model/efficientdet_d0_coco17_tpu-32/saved_model/1/
mv cv/od_model/efficientdet_d0_coco17_tpu-32/saved_model/!(1) cv/od_model/efficientdet_d0_coco17_tpu-32/saved_model/1/



