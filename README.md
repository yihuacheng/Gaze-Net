# Gaze-Net
The Pytorch Implementation of "MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation". (updated in 2021/04/25)

This is the implemented version metioned in our survey **"Appearance-based Gaze Estimation With Deep Learning: A Review and Benchmark"**.
Please refer our paper or visit our benchmark website <a href="http://phi-ai.org/GazeHub/" target="_blank">*GazeHub*</a> for more information.
The performance of this version is reported in the website.

To know more detail about the method, please refer the origin paper.

We recommend you to use the data processing code provided in <a href="http://phi-ai.org/GazeHub/" target="_blank">*GazeHub*</a>.
You can use the processed dataset and this code for directly running.

## License
The code is under the license of [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/).


## Introduction
We provide the code for leave-one-person-out evaluation.

The project contains following files/folders.
- `model.py`, the model code.
- `train.py`, the entry for training.
- `test.py`, the entry for testing.
- `config.yaml`, this file is the config of the experiment. To run our code, **you should write your own** `config.yaml`. 
- `reader.py`, the code for reading data. You should use suit reader for different dataset.

## Getting Started
### Writing your own *config.yaml*

Normally, for training, you should change 
1. `train.save.save_path`, The model is saved in the `$save_path$/checkpoint/`.
2. `train.data.image`, This is the path of image.
3. `train.data.label`, This is the path of label.

For test, you should change 
1. `test.load.load_path`, it is usually the same as `train.save.save_path`. The test result is saved in `$load_path$/evaluation/`.
2. `test.data.image`, it is usually the same as `train.data.image`.
3. `test.data.label`, it is usually the same as `train.data.label`.
 
### Training

You can run
```
python train.py config.yaml 0
```
This means the code running with `config_mpii.yaml` and use the `0th` person as the test set.

You also can run
```
bash run.sh train.py config.yaml
```
This means the code will perform leave-one-person-out training automatically.   
`run.sh` performs iteration, you can change the iteration times in `run.sh` for different datasets, e.g., set the iteration times as `4` for four-fold validation.


### Testing
In leaveone folder, you can run
```
python test.py config.yaml 0
```
or
```
bash run.sh train.py config.yaml
```

### Result
After training or test, you can find the result from the `save_path` in `config_mpii.yaml`. 


## Citation
If you use our code, please cite:
```
@ARTICLE{Zhang_2017_tpami,
	author={X. {Zhang} and Y. {Sugano} and M. {Fritz} and A. {Bulling}},
	journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
	title={MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation},
	year={2019},
	volume={41},
	number={1},
	pages={162-175},
	doi={10.1109/TPAMI.2017.2778103},
	ISSN={1939-3539},
	month={Jan}
}


@inproceedings{Cheng2021Survey,
	title={Appearance-based Gaze Estimation With Deep Learning: A Review and Benchmark},
	author={Yihua Cheng, Haofei Wang, Yiwei Bao, Feng Lu},
	booktitle={arxiv}
	year={2021}
}
```
## Contact 
Please email any questions or comments to yihua_c@buaa.edu.cn.
