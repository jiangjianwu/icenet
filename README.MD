# ICENet (Indoor Cognition Extraction Network)

## 1. Installation of dependent libraries

```bash
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2.Dataset Download

Click on [Indoor Space Cognition Dataset](https://gluteducn-my.sharepoint.com/:u:/g/personal/fengbuxi_glut_edu_cn/ERHslxh4kiZCsy1wHvnN9JgBeiZfLp9cXhHUg0ouTaNerQ?e=Q9piKL) to download the dataset.


password: fengbuxi


## 3. Task-level configuration file usage

+ Configure global_config.py to add the new task's category mapping dictionary directly when adding a new task, and change the value of pretrained_task at the same time.

+ The base model of the last mission should be placed under ~/.torch/models/ folder, the system will read the base model automatically, the default name is mtcnet.pth, if you change the name, please synchronize to change the pth_path variable under train.py.

## 4.Train

```bash
python train.py
```

## 5.Eval

```bash
python eval.py
```

## 6.Demo

```bash
python demo.py
```

## Connection

    + Email: fengbuxi@glut.edu.cn

    + WebSite: https://www.jaingjianwu.cn/

## Citation

```
@article{ICENet,
  title={Scene-level spatial cognition information extraction method for indoor safety monitoring of elderly-assisting robots}, 
  author={Jianwu Jiang and Jiadi Chen and Jingwen Li and Qingyang Wang and Bo Song},
  journal={},
  year={2024}
}
```

## Acknowledgements

+ This model is constructed based on [PSPNet](https://arxiv.org/pdf/1612.01105).

+ This model is built on the open source framework [awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch.git).
