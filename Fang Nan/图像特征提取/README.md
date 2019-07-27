# 图像特征提取

## [An Analysis of Single-Layer Networks in Unsupervised Feature Learning](http://proceedings.mlr.press/v15/coates11a/coates11a.pdf)

作者使用kmeans聚类使用滑窗操作对原图进行特征提取，得到feature map，再对feature map做sum pooling，得到的每一个图像对应的特征向量，在此基础上再做图像分类。

我比较懒，在这里 [Implementation for “An Analysis of Single-Layer Networks in Unsupervised Feature Learning”](https://github.com/shaonianruntu/MIL-Summer-School/tree/master/Implementation%20for%20%E2%80%9CAn%20Analysis%20of%20Single-Layer%20Networks%20in%20Unsupervised%20Feature%20Learning%E2%80%9D) 我对该论文进行了代码复现，并且在 notebook 文件中已经有对该论文复现过程的详细讲解，我就懒得再在这里写了。细节请移步，完。^.^

