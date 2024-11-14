# 微表情识别/Micro-Expression-Recognition
微表情在揭示一个人的真实情感方面起着关键的作用。然而，鉴别微表情具有挑战性，因为它们持续时间短，面部动作微妙，而且是自发的；本实验通过将时空传输和双通道注意机制整合到三维DenseNet框架中，提出了一种用于自发性MER的新型时空3D-DAT-DenseNet；我们在三个常见的数据集上进行了广泛的实验。我们的最优框架在SMIC、CASME-II和SAMM上的总体准确率分别为90.85%、92.28%和83.82%。实验结果表明，所提出的3D-DAT-DenseNet在三种不同的自发微表情数据集上取得了良好的结果，并且优于许多最先进的方法。

Micro-expressions play a pivotal role in uncovering an individual's true emotions. However, discerning micro-expressions is challenging due to their fleeting duration, subtle facial movements, and spontaneous nature. This experiment introduces a novel spatiotemporal 3D-DAT-DenseNet for spontaneous micro-expression recognition (MER) by integrating spatiotemporal transfer and dual-channel attention mechanisms into a three-dimensional DenseNet framework. We conducted extensive experiments on three common datasets. Our optimal framework achieved overall accuracy rates of 90.85%, 92.28%, and 83.82% on SMIC, CASME-II, and SAMM, respectively. The experimental results demonstrate that the proposed 3D-DAT-DenseNet yields promising outcomes across three distinct spontaneous micro-expression datasets, outperforming many state-of-the-art methods.

# 总体概括/General overview
        1.训练部分：数据集预处理（已经处理完）-> 处理为numpy形式并划分为训练集、验证集 ->加载T3D模型 -> 训练得到hdf5权重和训练验证结果。  
        1.Training Phase: Dataset Preprocessing (already completed) -> Converted to NumPy format and split into training and validation sets -> Load the T3D model -> Train to obtain HDF5 weights and training/validation results.
        
        2.测试部分：测试集划分 -> 加载模型权重 ->得到测试结果。   
        2.Testing Phase: Split the test sets -> Load model weights -> Obtain test results.
        
# 环境安装/Environment deployment
        使用 pip 安装依赖：pip install -r requirements.txt
 
# 代码说明/Related code description
**在文件`me_train.py`中:/In the file `me_train.py`:**  

        1.第54行说明了数据参数形式,第84-194是处理数据集部分、分为训练集和验证集. 
        1. Line 54 describes the data parameter format, and lines 84-194 handle the dataset processing, which is split into training and validation sets.
        
        2.第210-211定义加载了预处理数据的位置,231-247模型加载,定义3D模型参数.第252-256定义了训练参数,第281-303训练结束,输出训练hdf5权重.
        2. Lines 210-211 define the location of loading preprocessed data, lines 231-247 load the model and define 3D model parameters. Lines 252-256 define the training parameters, and lines 281-303 conclude the training, outputting the trained HDF5 weights.
        
        3.第307-340分别进行了训练集和验证集划分、保存验证集、加载验证集、定义验证集的工作.
        3. Lines 307-340 perform the division of the training and validation sets, save the validation set, load the validation set, and define the work with the validation set.
        
        4.第381-424进行了模型训练、得到了训练过程准确率、loss曲线、混淆矩阵的plt结果.
        4. Lines 381-424 conduct model training, obtaining the training process accuracy, loss curves, and the plt results of the confusion matrix.
       
**在文件`me_test.py`中:/In the file `me_test.py`:**

        1.大体结构与前者相似，其中，71-73加载了3类数据集.
        1. The structure is similar to the former, with lines 71-73 loading three types of datasets.
        
        2.在213-233进行了模型数据加载,380-421输出测试结果.
        2. Lines 213-233 perform model data loading, and lines 380-421 output the test results.
       
**在文件`se_densenet_3d.py` 和 `se_3d.py`中:/In the files `se_densenet_3d.py` and `se_3d.py`:**

        1.主要功能是对模型的调用,模型参数的设定.
        1. The main functions are to call the model and set model parameters.

# 数据集说明/Datasets specification
 ***开源数据集相关链接：***    
 ***Associated datasets is available at***   
 
 ***数据集规范链接***：https://creativecommons.org/licenses/by-nc/4.0/  
 ***Canonical URL***:https://creativecommons.org/licenses/by-nc/4.0/

        一共使用了三种数据集：SMIC、CASME-II、SAMM。实验在三个标准微表达数据集上进行，包括中国科学院CASME-II、SAMM、和自发微表达语料库SMIC。  
        Experiments are performed on three standard micro-ex-pression datasets including Chinese Academy of Sciences Micro-expression-II (CASME-II), SAMM and the Spontaneous micro-expression corpus (SMIC).
        
        其中，自发微表达语料库(SMIC)数据集根据拍摄设备的不同分为高速(HS)、普通视觉相机(VIS)和近红外(NIR)三个部分。HS由一台100 FPS高速摄像机记录，包括164个微表情样本;VIS由一台25 FPS普通摄像机记录，包括71个微表情样本;NIR由一台25 FPS近红外摄像机记录，该摄像机还包含71个微表情样本。我们选择HS数据集进行积极、消极和惊喜样本的实验，仅仅是因为与NIS和VIS相比，样本数量更多。

# 评价指标/Evaluation index
        ACC（Accuracy）：是机器学习中最常用的分类性能度量之一，表示模型预测正确的样本所占总样本的比例。  
        ACC (Accuracy) : is one of the most commonly used classification performance measures in machine learning, representing the proportion of the total sample that the model predicts correctly.  
        
        F1：是精确率（Precision）和 召回率（Recall）的调和平均数，常用于衡量分类模型在 不平衡数据 上的表现。它特别适合评估模型在数据类别不平衡时的性能，因为它同时考虑了错误分类的正类和负类样本。  
        F1: is the harmonic average of Precision and Recall and is often used to measure the performance of a classification model on unbalanced data. It is particularly suitable for evaluating a model's performance when data classes are unbalanced, as it takes into account both misclassified positive and negative class samples.  
        
        此外还进行了多个性能、对比实验，详见论文.  
        In addition, a number of performance and comparison experiments were carried out, as detailed in the paper.
