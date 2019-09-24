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

The implementaion will be done using **keras**, with tensorflow backend.Also, we will be using random images from **imagenet**dataset for training the model.We will use **50000 images** (RGB-224x224) for taining and **7498 images** for validation.

## Dependencies

* Tensorflow(>=1.14.0), Python 3
* Keras(>=2.2.4)
* Opencv, PIL, Matplotlib

## Prerequisites

* Download training [data-set](https://drive.google.com/file/d/1UBLzvcqvt_fin9Y-48I_-lWQYfYpt_6J/view?usp=sharing)
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

## How to run

Download the **dataset** from the above link and put them in **data** folder.
After ensuring the data files are stored in the **desired directorires**, run the scripts in the **following order**.

```python
1. python train.py # Train the model on data-set
2. python eval.py checkpoints/up_super_model-102-0.06.hdf5 # Evaluate the model on test-set
3. python export.py checkpoints/up_super_model-102-0.06.hdf5 # Export the model for deployment
4. python test.py test/four.jpeg # Test the model on a single image
5. python webcam.py test/beach.jpg # Run the model on webcam feed
```
You may also run the **Jupyter Notebook** (ipynb) in google colaboratory, after downloading the training dataset.

## Training graphs

Since we are using a **pretrained mobilentv2** as encoder for a head start, the training **quickly converges to 90% accuracy** within first couple of epochs. Also, here we use a flexible **learning rate schedule** (ReduceLROnPlateau) for training the model.


### Training Loss

![Screenshot](graphs/train_loss.png)

### Training Accuracy

![Screenshot](graphs/train_acc.png)

### Validation Loss

![Screenshot](graphs/val_loss.png)

### Validation Accuracy

![Screenshot](graphs/val_acc.png)


## Demo

### Result

Here the **inputs and outputs** are images of size **128x128**.
The **first row** represents the **input** and the **second row** shows the corresponding **cropped image** obtained by cropping the input image with the **mask output** of the model.

![Screenshot](result.png)

**Accuracy: 96%**

### Android Application

Real-time portrait video in android application

<p align="left">
  <img  src="android_portrait.gif">
</p>

(Shot on OnePlus 3 😉)

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
