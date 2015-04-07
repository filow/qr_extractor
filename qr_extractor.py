#coding: utf-8
import point
import numpy as np
import cv2
from os import path


def extract(img, qr_size=150):
  # 产生图片的灰度图形
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # 二维码提取的目标图片区域
  qr = np.zeros([qr_size,qr_size])
  # 将图片边缘化，留下所有边缘线
  edges = cv2.Canny(gray,104,255)
  # 找到图片中所有的边缘线，并保存到contours
  contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  # 计算边缘线圈的中心点
  mc = getContourCenters(contours)
  # 按特征寻找QR码的识别（边缘线嵌套5层以上）
  rect = findQrRect(contours, hierarchy)
  print rect
  cv2.imshow('image',edges)
  pass


# 寻找所有轨迹点的中心
def getContourCenters(contours):
  mc = []  # 所有轮廓的中心点
  for i in xrange(0,len(contours)):
    mu = cv2.moments(contours[i],False)
    if mu['m00'] < 0.01:
      mc.append([0,0])
    else:
      mc.append(point.new(mu['m10']/mu['m00'],mu['m01']/mu['m00']))
  return mc

def findQrRect(contours, hierarchy):
  rect = []
  hierarchy = hierarchy[0]
  # 遍历寻找五个框结合在一起的QR识别图形，
  for i in xrange(0,len(contours)):
    k=i
    c=0
    while hierarchy[k][2] != -1:
      k = hierarchy[k][2]
      c=c+1
    if hierarchy[k][2] != -1:
      c=c+1
    # 找到一个就注册一个标记，a'b'c轮流
    if c >= 5:
      rect.append(i)

  # 处理找到了超过三个方块的情况
  if len(rect)>3:
    rect = select_points(points,contours)
  return rect

# 从n个方块点中寻找3个可能的QR码识别方块，根据三个方块面积应该大致相等来确定
def select_points(points,contours):
  # points的长度
  p_len = len(points)
  # 每个轨迹的面积
  area = list(0 for n in xrange(p_len))
  for x in xrange(0,p_len):
    area[x] = cv2.contourArea(contours[points[x]])
  # 轨迹面积平均值
  avg = np.mean(area)

  # 从面积中选出3个与平均值最接近的值
  result = []
  while len(result)<3:
    closet_val = abs(avg-area[0])
    selected = 0
    for j in xrange(1,len(area)):
      val = abs(avg-area[j])
      if val < closet_val:
        closet_val = val
        selected = j
    result.append(points[selected])
    del area[selected]

  return result


# 解析图片中的二维码
def image(source):
  if not path.isfile(source):
    raise IOError("目标文件 <%s> 不存在！" % source)
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
