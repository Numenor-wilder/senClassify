segment
=====

<!-- TOC -->

- [环境](#环境)
- [运行](#运行)
    - [1.训练分类器](#1训练分类器)
    - [2.切片并预测](#2切片并预测)
        - [2.1计算时间区间](#21计算时间区间)
        - [2.2计算重复区间并切分](#22计算重复区间并切分)
        - [2.3预处理数据(清洗，分词)](#23预处理数据清洗分词)
        - [2.4计算切片特征并预测标签](#24计算切片特征并预测标签)
        - [2.5寻找用户标签变化的切片](#25寻找用户标签变化的切片)

<!-- /TOC -->

# 环境

* python 3.7.12

安装python依赖：
```bash
pip install -r requirements.txt 
```

# 运行

## 1.训练分类器

## 2.切片并预测

### 2.1计算时间区间

```bash
python -m segment.get_range
```

* 数据文件：`data/new_users_wzh/*.json`，内容需要从新到旧

* 输出到： `output/user_date_range.csv`

`output/user_date_range.csv`：

```csv
id,date_min,date_max
1003193701819207680,2018-06-03,2021-08-08
4333013352,2021-05-19,2021-08-08
1317755599627509761,2020-11-10,2021-08-22
```

### 2.2计算重复区间并切分

```bash
python -m segment.segment
```

* 数据文件：`output/user_date_range.csv`
* 输出：`output/cuted/<切片id>/<用户id>.json`

### 2.3预处理数据(清洗，分词)

```bash
python -m segment.pretreatment
```

* 数据文件：`output/cuted/<切片id>/<用户id>.json`
* 输出：`output/fenci/tweets/<切片id>/<用户id>.txt`

### 2.4计算切片特征并预测标签

```bash
python -m segment.get_features
```

* 数据文件：`output/fenci/tweets/<切片id>/<用户id>.txt`
* 数据文件：`output/user_date_range.csv`
* 二分类特征文件：`data/training_merge.csv`
* 输出：`output/user_twoClasses_label_data.csv`

### 2.5寻找用户标签变化的切片

```bash
python -m segment.findChange
```

* 数据文件：`output/user_label_data.csv`
* 输出：`output/user_label_change.csv`