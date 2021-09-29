# Bottle Vision

Bottle vision is a project developed to demo computer vision capabilities and applications. This case runs object detection to find bottles in an uploaded image, crops all bottles found, and finally classifies if each bottle has a cap or not.


### Visual flow

The flow is pretty simple, there is just a simple [Streamlit](https://streamlit.io/) UI for the user to upload the image that will be sent to the backend to run both models: object dection and cap classification

![alt text](https://github.com/agusbegue/bottle-vision/blob/master/data/screenshots/predictions.png?raw=true)

The model seems to be working pretty well, both in the test set and in these new images

![alt text](https://github.com/agusbegue/bottle-vision/blob/master/data/screenshots/bidon.png?raw=true)


### Models architecture

##### Object detection

The first model was [EfficientDet](https://ai.googleblog.com/2020/04/efficientdet-towards-scalable-and.html), developed by the Google Brain team, and it is one of the most advanced models for object detection. These range of models have great performance and are much lighter and faster than previous models. Luckily for us it was trained on the [COCO dataset](https://cocodataset.org/#home) and bottles are one of the 100 categories it was trained to detect, so we will just use it as-is.

##### Classification

After the bottles are detected and cropped, the second model has to classify if each bottle has a cap. For this section a custom model was developed in [Keras](https://keras.io/), not from scratch, but taking advantage of transfer learning. A pretrained model called [Xception](https://arxiv.org/abs/1610.02357) was used for extracting the features of the image. It was trained on the [ImageNet](https://www.image-net.org/) dataset, so it really learned to extract information from images and differentiate each of the 1000 classes. We will drop the final dense layers of the net for two reasons:

- Use features created in previous convolutional layers to learn this new task, instead of classifying into one of the 1000 classes
- Allow different size of input image. Using only the convolutional layers we have the liberty to input any new size we want, instead of the size of images the original model was trained for. This is important because with bottles the crops aren't squares, they are usually vertical, and this way we can set better dimentions than 1-1

So after these pretrained convolutional layers, we will add two dense layers that we will fine tune to be able to classify if the bottle has a cap or not. For this we need labeled images, so I went out and bought a bunch of different types of bottles, and took a lot of videos of these bottles in different settings (with and without cap, different filling levels, and different distances/angles/lightings). For each video the frames were extracted, the idea was to label each video instead of each image. After that the object detector was run on all images, cropped them and then fed them to the model to train. This whole pipeline was done in [Google Colab](https://colab.research.google.com/), in a notebook saved [here](https://colab.research.google.com/drive/19BKppxO-b2XF7X0LhDdnaHqAJ0IzC_eN?usp=sharing)

The classifier's architecture looks like this
![alt text](https://github.com/agusbegue/bottle-vision/blob/master/data/screenshots/model.png?raw=true)

##### Serving

Finally, both models are being served as an API using [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving), that allows you to have independent processes (the web and the serving), and just send the images to the different endpoints :

- /v1/models/efficientdet:predict for object detection
- /v1/models/classifier:predict for the classification


## How to use?

Clone the repository
```bash
git clone https://github.com/agusbegue/bottle-vision.git
```

Run the configuration and requirements
```bash
cd bottle-vision
sh setup.sh
```

Start serving the models
```bash
sudo docker run -p 8501:8501 --name=serving \
    --mount type=bind,source=${PWD}/models/od_model/efficientdet_d0_coco17_tpu-32/saved_model,target=/models/efficientdet \
    --mount type=bind,source=${PWD}/models/classifier,target=/models/classifier \
    --mount type=bind,source=${PWD}/models/models_serving.config,target=/models/models_serving.config \
    -t tensorflow/serving:latest \
    --model_config_file=/models/models_serving.config
```

On another terminal, start the Streamlit app
```bash
streamlit run src/main.py 8502
```


You will have your web running so you can access [localhost:8502](http://localhost:8502) and start using it!

There are some example images for you to try it [here](https://github.com/agusbegue/bottle-vision/blob/master/data/images)