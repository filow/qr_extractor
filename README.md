# QR码提取器 qrcode_extractor
用python实现的基于OpenCV的QR码提取器，实现在图片中分析QR码的位置并将其提取出来以待进一步的处理。

## 版权 Copyright
本提取器是在 BHARATH P 使用C++编写的 opencv_qr 项目基础上做的重构和改编，提升了程序的稳定性和代码的可读性，相较原项目更能让人理解提取算法。

原项目的地址在：[bharathp666/opencv_qr](https://github.com/bharathp666/opencv_qr). 该项目采用[ZERO开源协议](http://dsynflo.blogspot.in/p/blog-page_16.html). 如果希望查看原项目作者对于QR码提取算法的介绍，请参见[这个地址](http://dsynflo.blogspot.jp/2014/10/opencv-qr-code-detection-and-extraction.html)

本项目基于[Apache协议](https://www.apache.org/licenses/LICENSE-2.0)开源，您可以毫无阻碍的在您的项目中使用本代码，但由于该算法的实现还有些粗糙，且没有经过大量的测试，所以非常不建议您在生产环境中使用。

## 运行环境
本提取器在Mac OS X Yosemite下测试可正常运行。在运行前，请安装OpenCV, python以及numpy和OpenCV对python的接口。如果您要使用视频扫码功能，需要保证您的电脑有可用的摄像头。

## 软件接口