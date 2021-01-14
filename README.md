
## Image Classification with CNNs

This repo contains experiments comparing the accuracy of transfer-learning versus from-scratch learning and several different network architectures for image classification. 

I implemented several simple CNN architectures using increasing an number of feature maps at each layer plus dropouts. I also transfer learned from the convolutional bases of the Inception-ResNet V2 and VGG16 pre-trained architectures. I experimented with flattening versus global average pooling the final convolutional layers for the pre-trained models. I experimented with the depth and regularization of the final fully connected layers sitting on top of the pre-trained bases. I experimented with "fine-tune" training the last few layers of the pre-trained models using very low learning rates. I experimented with "warm-start" continuing training from the most promising models. Finally, I experimented with applying different image augmentations (stretching, rotating, cropping) to training data set.

### Methodology

This project is an attempt to practice *iterative*, *recorded*, and *reproducible* search for optimal hyper-parameters and architecture.

The [nbrun package](https://github.com/tritemio/nbrun) is used to execute a base experiment notebook with different combinations of parameters specifying the model architecture and other hyper-parameters. Each time an experiment is run a copy of the notebook is saved (as .ipynb and and .html) allowing reproducibility and later reference.

A logging framework is also defined which allows logging of metrics from each experiment as well as specifications of the data generators and models. Combined with saved model weights, the logged specifications allow for reconstruction of the model pipeline to predict on new data in a new python session without re-training the models.

A plot of the training loss and accuracy is also saved from each experiment.

**Example Training Plot** 

![](<figs/Train Test accuracy 2020-02-09_21h53m57s.png>)


### Project Structure

[code/Base Experiment.ipynb](<code/Base Experiment.ipynb>) -  the base notebook used to run experiments.  
[code/build_models.py](code/build_models.py) - this file defines all of the various model architectures tested.  
[code/build_data_gens.py](code/build_data_gens.py) - this file defines the the various data augmentation generators used.  
[code/saved experiments/](<code/saved experiments/>) - this directory stores the saved copies of each experiment notebook.  

### Data
Data is from this kaggle data set: https://www.kaggle.com/asdasdasasdas/garbage-classification

Data is images of five classes of recyclable material and one category of trash. 
A high degree of regularity in train images makes these models VERY poor at external generalization, i.e. while the models can accurately detect a image of paper from this dataset, they will do a poor job of identifying any random image of paper from the internet. That is because the images in the data set are all single items on a white background, under consistent lighting, at a uniform distance; random images from the internet will not have those same features.


### Leaderboard
See the top ten highest performing models.


*Note:* See [code/build_models.py](code/build_models.py) for the exact model configurations represented by the (non-timestamp) names in the "MODEL" column.  
*Note:* Names in the  "MODEL" column that are time stamps represent "warm-started" models which continued training from a previously trained configuration.  


<!--- This is copy/pasted from "misc/Top 10 models Log.html" --->
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>metrics_report.weighted avg.recall</th>
      <th>EPOCH</th>
      <th>MODEL</th>
      <th>run_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0.859756</td>
      <td>70.0</td>
      <td>VGG16 Fine-tuning</td>
      <td>2020-02-09_21h53m57s</td>
    </tr>
    <tr>
      <td>0.841463</td>
      <td>40.0</td>
      <td>2020-02-07_01h10m05s</td>
      <td>2020-02-09_14h25m27s</td>
    </tr>
    <tr>
      <td>0.814024</td>
      <td>40.0</td>
      <td>2020-02-07_01h36m15s</td>
      <td>2020-02-09_14h07m18s</td>
    </tr>
    <tr>
      <td>0.807927</td>
      <td>70.0</td>
      <td>2020-02-08_23h29m06s</td>
      <td>2020-02-09_15h50m36s</td>
    </tr>
    <tr>
      <td>0.786585</td>
      <td>100.0</td>
      <td>VGG16 Model</td>
      <td>2020-02-09_04h52m57s</td>
    </tr>
    <tr>
      <td>0.777439</td>
      <td>40.0</td>
      <td>2020-02-09_07h23m26s</td>
      <td>2020-02-09_17h27m39s</td>
    </tr>
    <tr>
      <td>0.777439</td>
      <td>100.0</td>
      <td>Inception-ResNet V2 finetuning final-module</td>
      <td>2020-02-09_07h23m26s</td>
    </tr>
    <tr>
      <td>0.774390</td>
      <td>NaN</td>
      <td>Lite Test</td>
      <td>2020-02-07_01h36m15s</td>
    </tr>
    <tr>
      <td>0.768293</td>
      <td>NaN</td>
      <td>Lite Test</td>
      <td>2020-02-07_01h10m05s</td>
    </tr>
    <tr>
      <td>0.746951</td>
      <td>70.0</td>
      <td>Inception-ResNet V2 w. Dropout Model</td>
      <td>2020-02-09_12h22m07s</td>
    </tr>
  </tbody>
</table>









