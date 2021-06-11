# First Order Motion Model for Image Animation adding a Lip Sync Expert discriminator 

This repository contains the source code for the paper [First Order Motion Model for Image Animation](https://papers.nips.cc/paper/8935-first-order-motion-model-for-image-animation) modified to add a Lip Sync Expert discriminator.



### Installation

We support ```python3```. To install the dependencies run:
```
pip install -r requirements.txt
```

### YAML configs

There are two configuration (```config/AVSpeech.yaml```) and (```config/LRS2.yaml```) files one for each `dataset`. 


### Animation Demo
To run a demo, download checkpoint and run the following command:
```
python demo.py  --config config/dataset_name.yaml --driving_video path/to/driving --source_image path/to/source --checkpoint path/to/checkpoint --relative --adapt_scale
```
The result will be stored in ```result.mp4```.

The driving videos and source images should be cropped before it can be used in our method. To obtain some semi-automatic crop suggestions you can use ```python crop-video.py --inp some_youtube_video.mp4```. It will generate commands for crops using ffmpeg. In order to use the script, face-alligment library is needed:
```
git clone https://github.com/1adrianb/face-alignment
cd face-alignment
pip install -r requirements.txt
python setup.py install
```

### Colab Demo 
@graphemecluster prepared a gui-demo for the google-colab see: ```demo.ipynb```. To run press ```Open In Colab``` button.

For old demo, see ```old-demo.ipynb```.


### Training

To train a model on specific dataset run:
```
CUDA_VISIBLE_DEVICES=0 python run.py --config config/dataset_name.yaml --device_ids 0 --syncnet_checkpoint_path path/to/lipsync_expert/checkpoint/
```
The code will create a folder in the log directory (each run will create a time-stamped new directory).
Checkpoints will be saved to this folder.
To check the loss values during training see ```log.txt```.
You can also check training data reconstructions in the ```train-vis``` subfolder.
By default the batch size is tunned to run on 2 or 4 Titan-X gpu (appart from speed it does not make much difference). You can change the batch size in the train_params in corresponding ```.yaml``` file.

### Evaluation on video reconstruction

To evaluate the reconstruction performance run:
```
CUDA_VISIBLE_DEVICES=0 python run.py --config config/dataset_name.yaml --mode reconstruction --checkpoint path/to/checkpoint --syncnet_checkpoint_path path/to/lipsync_expert/checkpoint/
```
You will need to specify the path to the checkpoint,
the ```reconstruction``` subfolder will be created in the checkpoint folder.
The generated video will be stored to this folder, also generated videos will be stored in ```png``` subfolder in loss-less '.png' format for evaluation.
Instructions for computing metrics from the paper can be found: https://github.com/AliaksandrSiarohin/pose-evaluation.

### Pre-trained lipsync expert discriminator checkpoint
Lip sync_expert discriminator checkpoint can be found under following link: [Wav2Lip-github](https://github.com/Rudrabha/Wav2Lip) .

### Image animation

In order to animate videos run:
```
CUDA_VISIBLE_DEVICES=0 python run.py --config config/dataset_name.yaml --mode animate --checkpoint path/to/checkpoint --syncnet_checkpoint_path path/to/lipsync_expert/checkpoint/
```
You will need to specify the path to the checkpoint,
the ```animation``` subfolder will be created in the same folder as the checkpoint.
You can find the generated video there and its loss-less version in the ```png``` subfolder.
By default video from test set will be randomly paired, but you can specify the "source,driving" pairs in the corresponding ```.csv``` files. The path to this file should be specified in corresponding ```.yaml``` file in pairs_list setting.

There are 2 different ways of performing animation:
by using **absolute** keypoint locations or by using **relative** keypoint locations.

1) <i>Animation using absolute coordinates:</i> the animation is performed using the absolute postions of the driving video and appearance of the source image.
In this way there are no specific requirements for the driving video and source appearance that is used.
However this usually leads to poor performance since unrelevant details such as shape is transfered.

2) <i>Animation using relative coordinates:</i> from the driving video we first estimate the relative movement of each keypoint,
then we add this movement to the absolute position of keypoints in the source image.
This keypoint along with source image is used for animation. This usually leads to better performance, however this requires
that the object in the first frame of the video and in the source image have the same pose


### Datasets

1) **AVSpeech**. This dataset can be downloaded [here](https://looking-to-listen.github.io/avspeech/).

2) **LRS2**.  This dataset can be downloaded [here](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html).

### Training on your own dataset
1) Resize all the videos to the same size e.g 256x256, the videos can be in '.gif', '.mp4' or folder with images.
We recommend the later, for each video make a separate folder with all the frames in '.png' format. This format is loss-less, and it has better i/o performance.

2) Create a folder ```data/dataset_name``` with 2 subfolders ```train``` and ```test```, put training videos in the ```train``` and testing in the ```test```.

3) Create a config ```config/dataset_name.yaml```, in dataset_params specify the root dir the ```root_dir:  data/dataset_name```. Also adjust the number of epoch in train_params.