# project_course：人脸检测识别门禁系统
本次课程项目的思路部分借鉴了：facenet、https://zhuanlan.zhihu.com/p/25025596?refer=shanren7

model文件由于体积问题未能上传，请自行下载后解压在项目文件夹（需要挂VPN）：下载地址：https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz

项目使用说明：
开发环境：tensorflow-gpu1.4.0+ python3.6 +opencv-python 3.4.1


***

下载好model文件并解压，且把本项目中压缩的new model.zip解压好后（照片太多只能压缩上传），打开ui.py

一共有三个功能，add a person（门禁系统增加用户）、sample（脸部采样）、entrance guard（门禁系统），分如下三步使用：

1.  首先使用sample！键盘输入采样人员的name（和已有的不一样），然后会弹出摄像头，脸部对准摄像头会进行大概40秒左右的采样（每10帧取一张，采样40张），采样结束会关闭摄像头

2.  这时候点add a person会弹出文件夹选择框，选择new model文件夹会发现刚刚输入的name已经创建好了一个文件夹，进入后在里面有一个detect_face文件夹，继续进入会找到采样的40张照片，选择40张照片并确定，系统会训练好该name的knn model（这时再查看name对应的文件夹会找到里面已经创建好的model）

3.  最后点击门禁识别entrance guard，并将脸对准摄像头，门禁系统会成功识别（如果不进行前两步会失败）。

***
