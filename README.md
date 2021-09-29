Serving of models
```bash
sudo docker run -p 8501:8501 --name=serving \
    --mount type=bind,source=${PWD}/models/od_model/efficientdet_d0_coco17_tpu-32/saved_model,target=/models/efficientdet \
    --mount type=bind,source=${PWD}/models/classifier,target=/models/classifier \
    --mount type=bind,source=${PWD}/models/models_serving.config,target=/models/models_serving.config \
    -t tensorflow/serving:latest \
    --model_config_file=/models/models_serving.config
```