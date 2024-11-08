# 微表情识别/Micro-Expression-Recognition
微表情在揭示一个人的真实情感方面起着关键的作用。然而，鉴别微表情具有挑战性，因为它们持续时间短，面部动作微妙，而且是自发的/Microexpressions play a key role in revealing a person's true emotions. However, identifying microexpressions is challenging because they are brief, involve subtle facial movements, and occur spontaneously.

# 总体概括/General overview
        训练部分：数据集预处理（已经处理完）-> 处理为numpy形式并划分为训练集、验证集 ->加载T3D模型 -> 训练得到hdf5权重和训练验证结果/Training Phase: Dataset Preprocessing (already completed) -> Converted to NumPy format and split into training and validation sets -> Load the T3D model -> Train to obtain HDF5 weights and training/validation results.
        
        测试部分：测试集划分 -> 加载模型权重 ->得到测试结果/Testing Phase: Split the test set -> Load model weights -> Obtain test results.
# 环境安装/Environment deployment
        使用 pip 安装依赖：pip install -r requirements.txt
# 简单说明/brief description
        在文件me_train.py中:/In the file `me_train.py`:
        1.第54行说明了数据参数形式,第84-194是处理数据集部分、分为训练集和验证集./1. Line 54 describes the data parameter format, and lines 84-194 handle the dataset processing, which is split into training and validation sets.
        2.第210-211定义加载了预处理数据的位置,231-247模型加载,定义3D模型参数.第252-256定义了训练参数,第281-303训练结束,输出训练hdf5权重./2. Lines 210-211 define the location of loading preprocessed data, lines 231-247 load the model and define 3D model parameters. Lines 252-256 define the training parameters, and lines 281-303 conclude the training, outputting the trained HDF5 weights.
        3.第307-340分别进行了训练集和验证集划分、保存验证集、加载验证集、定义验证集的工作./3. Lines 307-340 perform the division of the training and validation sets, save the validation set, load the validation set, and define the work with the validation set.
        4.第381-424进行了模型训练、得到了训练过程准确率、loss曲线、混淆矩阵的plt结果./4. Lines 381-424 conduct model training, obtaining the training process accuracy, loss curves, and the plt results of the confusion matrix.
       
        在文件me_test.py中:/In the file `me_test.py`:
        1.大体结构与前者相似，其中，71-73加载了3类数据集./1. The structure is similar to the former, with lines 71-73 loading three types of datasets.
        2.在213-233进行了模型数据加载,380-421输出测试结果./2. Lines 213-233 perform model data loading, and lines 380-421 output the test results.
       
        在文件se_densenet_3d.py\se_3d.py中:/In the files `se_densenet_3d.py` and `se_3d.py`:
        1.主要功能是对模型的调用,模型参数的设定./1. The main functions are to call the model and set model parameters.
      

# 评价指标/evaluation index
        进行了多个对比实验，详见论文./A number of comparative experiments were conducted, as detailed in the paper.
