# Deep Video Steganography: Hiding Videos in Plain Sight

A cnovolutional neural network for hiding videos inside other videos.It is implemented in keras/tensorflow using the concepts of deep learning, steganography and encryption.

## Background

**Steganography** is the practice of **concealing a secret message** within another, ordinary, message.The messages can be images, text, video, audio etc. In modern steganography, the goal is to **covertly communicate** a digital message.
The main aim of steganogrpahy is to prevent the detection of a hidden message. It is often combined with **cryptography** to improve the security of the hidden message.**Steganalysis** is the study of detecting messages hidden using steganography (breaking); this is analogous to cryptanalysis applied to cryptography.Steganography is used in **applications** like confidential communication, secret data storing, digital watermarking etc.

<p align="center">
  <img  src="https://media.springernature.com/original/springer-static/image/chp%3A10.1007%2F978-981-10-6872-0_26/MediaObjects/393723_1_En_26_Fig2_HTML.gif">
  <br>
  Basic Working Model
</p>

Steganography on images can be broadly  classified  as  **spatial  domain**  steganography  and **frequency  domain** steganography.In spatial domain, algorithms directly  manipulate the values (**least significant bits**) of some selected pixels.
In frequency domain, we change some **mid-frequency components** in the frequency domain.These heuristics are effective in the domains for which they are designed, but they are fundamentally **static** and therefore **easily detected**.We can evaluate a steganographic technique or algorithm by using **performance and qualtiy metrics** like capacity, secrecy, robustness, imperceptibility, speed, applicabilty etc.

Here we plan to extend the basic implementation from the paper 'Hiding images in plain sight: Deep steganography' to videos, i.e we will train a model for hiding videos within other videos using convolutional neural networks.Also, we will incorporate additional **block-shuffling** as an encryption method for **added security** and other **image enhancemnet** techniques for **improving the output quality**.

The implementaion will be done using **keras**, with tensorflow backend.Also, we will be using random images from  **imagenet**dataset for training the model.We will use **50000 images** (RGB-224x224) for taining and **7498 images** for validation.

## Dependencies

* Tensorflow(>=1.14.0), Python 3
* Keras(>=2.2.4)
* Opencv, PIL, Matplotlib

## Prerequisites

* Download training [data-set](https://github.com/dusty-nv/jetson-inference/blob/master/docs/imagenet-training.md#downloading-image-recognition-dataset)
* Download evalutation [data-set](https://www.ntu.edu.sg/home/asjfcai/Benchmark_Website/Semantic%20dataset100.zip)
* GPU with CUDA support


## Model Architecture

Our main goal is to **hide a full size (N*N RGB) color image** within another image
of the same size. Deep neural networks are simultaneously trained to create the **hiding and
revealing processes** and are designed to specifically work as a pair. The technique used is **image
compression through auto-encoding networks**.The trained system must learn to compress the
information from the secret image into the **least noticeable portions** of the cover image and then, it
must learn how to **extract and reconstruct** the same information from the encoded message, with
**minimum loss**.

Here is the basic **architecture** diagram

<p align="center">
  <img  src="steg_arch.png">
  <br>
  Basic Architecture
</p>

## Implementation

We train the hiding and reveal networks **simultaneously** in the form of an autoencoder, using keras.The model has two inputs corresponding to a pair of secret and cover image and two outputs corresponding to their inputs .Since we are using a **autoencoder** based architecture, the **labels** are same as their corresponding inputs.

The network consists of **three parts** viz. Perpare block, Hide block, Reveal block.In **prepare block**, we  transform the color-based pixels to more useful features for succinctly encoding the images. We then hide this transformed image inside the input cover image using the  **hide block**, to generate the container image.Finally, in the **reveal block** we decode the container image to produce the secret output.Therefore, the **training graph** has two inputs and two outputs.

**Training Graph**

![Screenshot](train_graph.png)

We use a **weighted L2 loss** function along with **Adam optimizer** for training the model.The model is trained for **100 epochs** suing a batch size of 8.

*Loss: L(c, c 0 , s, s 0 ) = ||c − c 0 || + β||s − s 0 ||*


Here c and s are the cover and secret images respectively, and β is how to weigh their reconstruction errors

To ensure that the networks do not simply encode the secret image in the **LSBs**, a small amount of **noise** is added to the output of the second network (e.g. into the generated container image) during training.

After the training, we split the trained model into two: **hide network and reveal network** (we remove noise layer).The **hide network** has two inputs corresponding to **secret and cover image** and one output corresponding to the **container image**. The reveal  network takes the **container image** as input and reveals(decodes) the **secret image** as output.

**Hide Network**

![Screenshot](models/hide.png)

**Reveal Network**

![Screenshot](models/reveal.png)

The hide network is used by the **sender**; while the reveal network is supposed to be used by the **receiver**.The receiver has access only to the container image.In additon to the normal steganographic hiding mechanism, we also **encrypyt(block shuffle)** our secret imagess for added security.Therefore , both the **sender and the receiver** shares a **symmetric key** for encrypting/decrypting the shuffled secret message. The encryption is performed by the sender on the **input secret image**; whereas the decryption is performed by the receiver on the **final decode image**.

Finally images/video are enhanced by applying **Non-local Means Denoising** algorithm and **Unsharp Masking**(USM) filter.

## How to run

1. Download the **imagenet dataset** and put them in **data** folder.
2. Select a **random subset** of images from the  imagenet dataset.
3. Resize all the images to **224*224(RGB)** size.
4. Split the dataset into **training and validation** subsets.

Also ensure the that **evalutaion** images(RGB:224x224) are stored in the directory **dataset/eval_data**.

### Directory structure:- 

dataset
├── eval_data
├── train_data
│   └── train
└── val_data
    └── validation

Configure the **filepaths and batch-size** in train.py, if needed.
After ensuring the data files are stored in the **desired directorires**, run the scripts in the **following order**.

```python
1. python train.py # Train the model on training dataset
2. python eval.py dataset/eval_data checkpoints/steg_model-06-0.03.hdf5 # Evaluate the model on evaluation dataset
3. python test.py test/testdata.npy checkpoints/steg_model-06-0.03.hdf5 # Test the model on test dataset
4. python split_model.py checkpoints/steg_model-06-0.03.hdf5 # Split the model into hide and reveal networks
```
* Use **image_hide.py & image_reveal.py** to test the models on **images**.
* Use **video_hide.py & video_reveal.py** to test the models on **videos**.
* Use **enhance.py** for enhancing the **output** secret video.

## Training graphs

Since we are using a **pretrained mobilentv2** as encoder for a head start, the training **quickly converges to 90% accuracy** within first couple of epochs. Also, here we use a flexible **learning rate schedule** (ReduceLROnPlateau) for training the model.

![Screenshot](losses.png)


## Demo

### Image Results

Sample results for a pair of input images - Secret & Cover

**Inputs**: Secret and Cover

![Screenshot](results/input.png)

**Outputs**: Secret and Cover

![Screenshot](results/output.png)

*Trian MSE: 0.03*, *Test MSE: 0.02*

### Video Results

Sample results for a pair of input videos - Secret & Cover

**Inputs**: Secret and Cover

<p align="left">
  <img  src="videos/secret_input.gif"  width="224" height="224">
  <img  src="videos/cover_input.gif" width="224" height="224">
</p>
<br>

**Outputs**: Secret and Cover
<p align="left">
  <img  src="videos/secret_outvid.gif" width="224" height="224">
  <img  src="videos/cover_outvid.gif" width="224" height="224">
</p>

## Key Insights and Drawbacks

1. Always start experimentation with **standard/pretrained** networks. Also try out **default/standard hyperparameter** settings before experimentation.
2. Make sure your **ground truth is correct/uncorrupted** and is in **desired format** before training (even standard dataset).
3. For **mobile devices**, make sure you use a **mobile-friendly architecture (like mobilenet)** for training and deployment.
4. Using **google colaboratory** along with google drive for training was **EASY & FUN**.It provides **high end GPU** (RAM also) for free.
5. Some of the mobile **optimization tools**(even TF) are still **experimental** (GPU deegate, NNAPI, FP16 etc.) and are buggy.They support only **limited operations and edge devices**.
6. Even **state-of-the art segmenation models**(deeplab-xception) seems to suffer from **false positives** (even at higher sizes), when we test them on a random image.
7. The **segmentaion maps** produced at this low resolution (128x128) have coarse or **sharp edges** (stair-case effect), especially when we resize them to higher resolution.
8. To tackle the problem of coarse edges, we apply a **blur filter** (also antialiasing at runtime) using **opencv** and perform **alpha blending** with the background image. Other approach was to **threshold** the blurred segmentation map with **smooth-step** function using **GLSL shaders**.
9. In android we can use **tensorflow-lite gpu-delegate** to speed up the inference.It was found that **flattening** the model output into a **rank 1 (or 2)** tensor helped us to reduce the **latency** due to **GPU-CPU** data transfer.Also this helped us to **post-process** the mask without looping over a multi-dimensional array.
10. Using **opencv (Android NEON)** for post-processing helped us to improve the **speed** of inference.But this comes at the cost of **additional memory** for opencv libraray in the application.
11. Still, there is a scope for improving the **latency** of inference by performing all the **postprocessing in the GPU**, without transfering the data to CPU. This can be acheived by using opengl shader storge buffers (**SSBO**). We can configure the GPU delegate to **accept input from SSBO** and also access model **output from GPU memory** for further processing (wihout CPU) and subsequent rendering.
12. The **difference** between the **input image frame rate and output mask generation frame rate** may lead to an output(rendering), where the segmentation **mask lags behind current frame**.This **stale mask** phenemena arises due to the model(plus post-processing) taking more than 40ms (corr. to 25 fps input) per frame (real-time video). The **solution** is to render then output image in accordance to the **mask generation fps** (depends on device capability) or **reduce the input frame rate**.
13. If your segmentaion-mask output contains **minor artifacts**, you can clean them up using **morphological operations** like **opening or closing**. However it can be slightly **expensive** if your output image size is **large**, especially if you perform them on every frame output.
14. If the background consists of **noise, clutter or objects like clothes, bags**  etc. the model **fails** miserably.
15. Even though the stand-alone **running time** of exported (tflite) model is  **low(around 100 ms)**,other operations like **pre/post-processing, data loading, data-transfer** etc. consumes **significant time** in a mobile device.


## TODO

* Port the code to **TF 2.0**
* Use a **bigger image** for training(224x224)
* Try **quantization-aware** training
* Train with **mixed precision** (FP16) 
* Optimize the model by performing weight **pruning**
* Improve **accuracy** & reduce **artifacts** at runtime
* Incroporate **depth** information and **boundary refinement** techniques

## Versioning

Version 1.0

## Authors

Anil Sathyan

## Acknowledgments
* https://www.tensorflow.org/model_optimization
* https://github.com/cainxx/image-segmenter-ios
* https://github.com/gallifilo/final-year-project
* https://github.com/lizhengwei1992/mobile_phone_human_matting
* https://machinethink.net/blog/mobilenet-v2/
*   [Deeplab Image Segmentation](https://colab.research.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb)
*   [Tensorflow - Image segmentation](https://www.tensorflow.org/beta/tutorials/images/segmentation)
*   [Hyperconnect - Tips for fast portrait segmentation](https://hyperconnect.github.io/2018/07/06/tips-for-building-fast-portrait-segmentation-network-with-tensorflow-lite.html)
* [Prismal Labs: Real-time Portrait Segmentation on Smartphones](https://blog.prismalabs.ai/real-time-portrait-segmentation-on-smartphones-39c84f1b9e66)
* [Keras Documentation](https://keras.io/)
* [Boundary-Aware Network for Fast and High-Accuracy Portrait Segmentation](https://arxiv.org/pdf/1901.03814.pdf)
* [Fast Deep Matting for Portrait Animation on Mobile Phone](https://arxiv.org/pdf/1707.08289.pdf)
