# MaleWare Detection

## 脚本

- add_prefix.py: 给文件添加你前缀
- exe_to_bytes.py: exe 文件转字节码序列
- split_data.py: 划分数据集
- exe_to_img.py: exe 文件转图片


## 文章

- [准确率99%！基于深度学习的二进制恶意样本检测](http://www.hansight.com/blog-deepsense-virus-detection.html)
- [基于机器学习的高效恶意软件分类系统](http://blog.leanote.com/post/sixiao/%E8%AE%BA%E6%96%87)
- [镇守最后一道防线：三种逃逸沙盒技术分析](http://www.4hou.com/technology/3665.html)
- [分析恶意软件的加壳技巧](http://www.mottoin.com/103075.html)
- [专家教你利用深度学习检测恶意代码](https://zhuanlan.zhihu.com/p/32251097)
- [利用机器学习进行恶意代码分类](https://bindog.github.io/blog/2015/08/20/microsoft-malware-classification/)
- [超赞的恶意软件分析](https://github.com/sunnyelf/awesome-malware-analysis/blob/master/%E8%B6%85%E8%B5%9E%E7%9A%84%E6%81%B6%E6%84%8F%E8%BD%AF%E4%BB%B6%E5%88%86%E6%9E%90.md)
- [恶意软件分析大合集](https://github.com/rshipp/awesome-malware-analysis/blob/master/%E6%81%B6%E6%84%8F%E8%BD%AF%E4%BB%B6%E5%88%86%E6%9E%90%E5%A4%A7%E5%90%88%E9%9B%86.md)
- [EMBER](https://github.com/endgameinc/ember)


## 错误日志

- error.log: 记录了所有编程过程中出现了问题


## 实验结果

### 实验一
```
预处理：过滤并去除了大于 10 MB 的正常软件 
正常软件数量：8754
恶意软件数量：40848
数据划分：随机乱序后，正常软件 80% 划入训练集，20% 划入验证集，恶意软件 10000 个划入训练集，2000 个划入验证集
cnn_base_model.log
xception.log
resnet.log
inception.log
```

### 实验二
```
预处理：过滤并去除了大于 10 MB 的正常软件 
正常软件数量：8754
恶意软件数量：40848
恶意软件 / 正常软件：4.7 : 1
数据划分：随机乱序后，正常软件 80% 划入训练集，20% 划入验证集，恶意软件 80% 划入训练集， 20% 划入验证集
cnn_base_model_82.log
xception_82.log
resnet_82.log
inception_82.log
```


