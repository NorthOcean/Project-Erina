<!--
 * @Author: ConghaoWong
 * @Date: 2019-12-20 09:37:18
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2020-09-09 19:19:06
 * @Description: file contentz
 -->

# Codes for BGM: Building a Dynamic Guidance Map without Visual Images for Trajectory Prediction

## The BGM Model

<div align='center'><img src="./figures/overview.png"></img></div>

BGM has shown a better prediction results on the ETH-UCY dataset than others.
Some visualized results are listed as follow.

<div align='center'>
<table>
    <thead>
        <tr>
            <th>conditions</th>
            <th>SR-LSTM (CVPR2019)</th>
            <th>Social GAN-P (K=20) (CVPR2018)</th>
            <th>Social STGCNN (K=20) (CVPR2020)</th>
            <th>BGM</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th>Meet (3:28 in UCY-zara1)</th>
            <th><img src="./figures/zara1_351_sr.jpg"></img></th>
            <th><img src="./figures/zara1_351_sgan.jpg"></img></th>
            <th><img src="./figures/zara1_351_stgcnn.jpg"></img></th>
            <th><img src="./figures/zara1_351_bgm.jpg"></img></th>
        </tr>
        <tr>
            <th>Follow (3:28 in UCY-zara1)</th>
            <th><img src="./figures/zara1_359_sr.jpg"></img></th>
            <th><img src="./figures/zara1_359_sgan.jpg"></img></th>
            <th><img src="./figures/zara1_359_stgcnn.jpg"></img></th>
            <th><img src="./figures/zara1_359_bgm.jpg"></img></th>
        </tr>
        <tr>
            <th>Meet (6:47 in ETH-hotel)</th>
            <th><img src="./figures/hotel_159_sr.jpg"></img></th>
            <th><img src="./figures/hotel_159_sgan.jpg"></img></th>
            <th><img src="./figures/hotel_159_stgcnn.jpg"></img></th>
            <th><img src="./figures/hotel_159_bgm.jpg"></img></th>
        </tr>
        <tr>
            <th>Follow (10:53 in ETH-hotel)</th>
            <th><img src="./figures/hotel_284_sr.jpg"></img></th>
            <th><img src="./figures/hotel_284_sgan.jpg"></img></th>
            <th><img src="./figures/hotel_284_stgcnn.jpg"></img></th>
            <th><img src="./figures/hotel_284_bgm.jpg"></img></th>
        </tr>
    </tbody>
</table>
<table>
    <tbody>
        <tr>
            <th><img src="./vis_pngs/obs.png"></img></th>
            <th>observations</th>
            <th><img src="./vis_pngs/gt.png"></img></th>
            <th>ground truths</th>
            <th><img src="./vis_pngs/pred.png"></img></th>
            <th>predictions</th>
        </tr>
    </tbody>
</table>
</div>

## Requirements
All codes are developed with Python 3 with NVIDIA TITAN X (Pascal) GPUs.
Detailed packages used can be seen in `./requirements.txt`.
You can install all of these packages by this command:

```bash
pip install -r ./requirements.txt
```

## Already Trained Models

You can also download our already trained models here(Available soon) to evaluate the BGM on ETH-UCY datasets.
The evaluation results should be as follows if there are no mistakes of configuration.

<div align='center'>
<table>
    <thead>
        <tr>
            <th>index</th>
            <th>dataset</th>
            <th>models</th>
            <th>download links</th>
            <th>BGM</th>
            <th>BGM w/o Social Module</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th>0</th>
            <th>eth</th>
            <th>20200905-202720NEW_500MRR3SSLSTMmap0</th>
            <th>available soon</th>
            <th>0.52/1.00</th>
            <th>0.50/1.00</th>
        </tr>
        <tr>
            <th>1</th>
            <th>hotel</th>
            <th>20200905-202722NEW_500MRR3SSLSTMmap1</th>
            <th>available soon</th>
            <th>0.25/0.48</th>
            <th>0.25/0.47</th>
        </tr>
        <tr>
            <th>2</th>
            <th>zara1</th>
            <th>20200905-202724NEW_500MRR3SSLSTMmap2</th>
            <th>available soon</th>
            <th>0.43/0.93</th>
            <th>0.41/0.91</th>
        </tr>
        <tr>
            <th>3</th>
            <th>zara2</th>
            <th>20200905-202726NEW_500MRR3SSLSTMmap3</th>
            <th>available soon</th>
            <th>0.34/0.73</th>
            <th>0.33/0.72</th>
        </tr>
        <tr>
            <th>4</th>
            <th>univ</th>
            <th>20200905-202727NEW_500MRR3SSLSTMmap4</th>
            <th>available soon</th>
            <th>0.48/1.03</th>
            <th>0.47/1.03</th>
        </tr>
    </tbody>
</table>
</div>


## Train New Models on ETH-UCY datasets

The original  ETH-UCY datasets (contains ETH-eth, ETH-hotel, UCY-zara1,2,3 and UCY-univ1,3,examples) are included in `./data/eth` and `./data/ucy` with the true position in `.csv` format.
To train a new model on these datasets, please use the commands above as an example:

```bash
python main.py \
    --test_set 0 \
    --epochs 500 \
    --batch_size 5000 \
    --lr 0.001 \
    --model_name example_model
```

More options can be seen below in section `Options`.
Your new model will be saved at `./logs/YOUR_TRAINING_TIME_YOUR_MODEL_NAME/`.
(For example `./logs/20200909-120030example_model`)

## Train New Models on Your Own Datasets

If you want to train BGM on your own dataset, please make your datas into several `.csv` files.
Their format are request as:

- The file should contain 4 columns.
- 



## Evaluate Models

### Prepare model and data

The fully saved model should contain at least (i) a model weight `YOUR_MODEL_NAME.h5` file and (ii) a training args `YOUR_MODEL_NAMEargs.npy` file.
If there is no test inputs file `YOUR_MODEL_NAMEtest.npy`, you shoud make your own test datas (it should be a `.csv` file) into the format BGM used by the following commands:

```bash
Available Soon
```

### Run the evaluation

You can run the evaluation of your model by the following commands:

```bash
python main.py \
    --load YOUR_MODEL_DIR/YOUR_MODEL_NAME \
    --draw_results 1
```

### Visualized Results

Set the evaluate arg `--draw_results` to `1` to enable saving visualized results.
For eth, hotel, zara1 and zara2 in ETH-UCY dataset, we have already calcutated their mapping parameters from real world positions to video-pixel positions in `./visual.py` as follows:

```python
...
class TrajVisual():
    def __init__(self, ...):
        self.video_path = ['path_for_set_0', ..., 'path_for_set_n']
        self.paras = [[sample_rate, frame_rate], ...]
        self.weights = [
            [Wx1, bx1, Wy1, by1],
            [np.array(H2), Wx2, bx2, Wy2, by2],
            ...
        ]
...
```

You need change the above video path list `self.video_path` to each video of your dataset.
If your dataset is the record with real world positions, you should calculate your transformation matrix `H` (optional) or the linear mapping weights `W` and `b` for each axis and write to `self.weights`.
Besides, you should also correct the frame rate of your videos and the sample rate of your dataset.
Default save path of these visualized results is `./YOUR_MODEL_DIR/VisualTrajs/`.

## Model Options

You can custom these options both on model compents and training or evaluation.
Note that all items that need a bool type of inputs should takes integer `0` and `1` instead of `False` and `True`. 

**Environment options**:

- `--gpu`:
(Optional) Choose which GPU the model training or evaluation on.
Parameter should be a positive integer.
- `--verbose`:
(Available Soon) Set if model gives output logs.
Default is `True`.

**Model options**:

- `--obs_frames`:
Length of historical trajectories.
Default is `8`.
- `--pred_frames`:
Length of predictions.
Default is `12`.
- `--dropout`:
The rate of dropout.
Default is `0.5`.
- `--calculate_social`:
Controls whether the model predict agent's neighbors trajectories.
Default is `False`.
It will be set to `True` automatically when `--sr_enable` has been set to `True`.
- `--sr_enable`:
Controls whether enable the Social Module.
Default is `False`.

**Social module options**:

- `--grid_shape_x`:
Grid width of the social energy map.
Default is `700`.
- `--grid_shape_y`:
Grid height of the social energy map.
Default is `700`.
- `--grid_length`:
Resolution of each grid in meters.
Default is `0.1`.
- `--avoid_size`:
Ideal distance to avoid collision with other people in grid size.
Default is `15`.
- `--interest_size`:
Ideal distance to be attrected by agent's original intention in grid size.
Default is `20`.
- `--max_refine`:
The maximum limit that social module could changes in meters.
Dufault is `0.8`.

**Guidance map options**:

Args with * represent those args have not been transferred to `main.py`.
Please change them manually when needed.

- `--window_size_map`*:
Resolution of the guidance map that shows the number of grids used to represent the reality 1 meter.
Default is `4`.
(Current in `./GridRefine.py`)
- `--local_half_size`*:
Half side length of the local guidance maps in grid size.
Default is `16`.
(Current in `./PrepareTrainData.py`)

**Training and evaluation options**:

- `--test_set`:
Test set of current model.
In ETH-UCY datasets, eth=`0`, hotel=`1`, zara1=`2`, zara2=`3` and univ=`4`.
When using your own datasets, please refer to the above sections.
Default is `2`.
- `--load`:
The model path to be evaluated.
Set `'null'` if you want to train new models.
Default is `'null'`.
- `--train_percent`:
Controls the percent of training set used to train the model.
If you have 5 datasets, you could use command `--train_percent 0.1 0.3 0.5 0.7 0.9` to controls the percent each dataset used.
You can also gives it one float number like `--train_percent 0.1` to set all training sets the same percent.
Default is `1.0`.
- `--reverse`:
Set whether use the reversation to strengthen training data.
Default is `False`.
- `--add_noise`:
Set whether use the additional noise data to strengthen training set.
Set `False` if you do not need this strengthen method, and set a positive integer `n` to add `n` times of noise data to the original training set.
Default is `False`.
- `--rotate`:
Set whether use the additional rotation data to strengthen training set.
Set `False` if you do not need this strengthen method, and set `n` means that the model will divide the 360 degrees into `n` parts, and add all other new degree's rotation trajectories except 0 degree to original training set.
Default is `False`.
- `--test`:
Set `True` to enable test during training.
It will be set to `True` automatically when `--load` is not `'null'`.
Default is `True`.
- `--start_test_percent`:
Controls when to start test during training.
Parameter should be a 0~1 float value.
Default is `0.0`.
It only works when `--test` is set to `True`.
- `--test_step`:
Controls the epoch step between two times of test during training.
Default is `3`.
- `--epochs`:
The number of epochs when training.
Default is `500`.
- `--lr`:
Learning rate.
Default is `0.001`.
- `--batch_size`:
Batch size.
Default is `500`.

**Save args**:

- `--model_name`:
The name of your new model.
Default is `'model'`.
- `--save_model`:
Controls if save the trained model after training.
Default is `True`.
- `--log_dir`:
The root dir where model saved.
Your model will be saved at `./logs/YOUR_TRAINING_TIME_YOUR_MODEL_NAME` when leave it `'null'`.
Default is `'null'`.
