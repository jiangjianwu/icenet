# ICENet (Indoor Cognition Extraction Network)

## 1.安装依赖库

```bash
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2.数据集下载

点击此链接[Indoor Space Cognition Dataset](https://gluteducn-my.sharepoint.com/:u:/g/personal/fengbuxi_glut_edu_cn/ERHslxh4kiZCsy1wHvnN9JgBeiZfLp9cXhHUg0ouTaNerQ?e=Q9piKL)下载数据集。

密码：fengbuxi


## 3.任务级配置文件使用方法

+ 配置global_config.py，新增任务时直接添加新任务的类别映射字典，并同时改变pretrained_task的值

+ 上一轮任务的基础模型请放置于 ~/.torch/models/ 文件夹下，系统会自动读取基础模型，默认名称为mtcnet.pth，如改了名称请同步修改train.py 下的pth_path变量

## 4.训练
```bash
python train.py
```

## 5.验证
```bash
python eval.py
```

## 6.测试
```bash
python demo.py
```

## 联系方式

    + Email: fengbuxi@glut.edu.cn

    + WebSite: https://www.jaingjianwu.cn/

## 引用

```
@article{ICENet,
  title={Scene-level spatial cognition information extraction method for indoor safety monitoring of elderly-assisting robots}, 
  author={Jianwu Jiang and Jiadi Chen and Jingwen Li and Qingyang Wang and Bo Song},
  journal={Pending},
  year={2024}
}
```

## 鸣谢

+ 本模型基于[PSPNet](https://arxiv.org/pdf/1612.01105)构建。

+ 本模型基于开源架构[awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch.git)构建。
