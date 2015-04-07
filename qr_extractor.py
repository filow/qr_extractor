#coding: utf-8
import point
import numpy as np
import cv2

def extract(img):
  pass


# 解析图片中的二维码
def image(source):
  img = cv2.imread(source,1)
  extract(img)
  cv2.namedWindow('image',cv2.WINDOW_NORMAL)
  cv2.namedWindow('qr',cv2.WINDOW_NORMAL)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


# 从摄像头中读取二维码数据
def camera(width = 640, height = 480):
  cap = cv2.VideoCapture(0)
  cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)
  cv2.namedWindow('image',cv2.WINDOW_NORMAL)
  cv2.namedWindow('qr',cv2.WINDOW_NORMAL)
  while(1):
    k = cv2.waitKey(1) & 0xFF
    if k==27:
      break
    _, frame = cap.read()
    extract(frame)
  cv2.destroyAllWindows()
