docker pull tensorflow/serving:latest-gpu

git clone https://github.com/tensorflow/serving

#docker run -p 8501:8501 --mount type=bind,source=/home/ubuntu/bottle_cv/models/efficientdet/efficientdet_d0_coco17_tpu-32/saved_model,target=/models/efficientdet -e MODEL_NAME=efficientdet -t tensorflow/serving:latest-gpu &
docker run -p 8501:8501 --mount type=bind,source=/home/ubuntu/bottle_cv/cv/od_model/efficientdet_d0_coco17_tpu-32/saved_model,target=/models/efficientdet -e MODEL_NAME=efficientdet -t tensorflow/serving:latest-gpu &



curl -F "image=@/home/ubuntu/bottle_cv/tests/coca512.png" -X POST http://127.0.0.1:8501/v1/models/efficientdet:predict -o /home/ubuntu/bottle_cv/data/prediction
