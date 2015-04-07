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

  # 仅当找到3个QR标志且面积都合适时，才进行下一步的识别操作
  if len(rect)==3 and canReconize(contours,rect):
    # 已经找到了3个QR标志，下面判断三个图形的位置
    orientation,top,bottom,right = findQrRectDirection(mc,rect)

    # 绘制三个识别框
    cv2.drawContours(img, [contours[top]], -1, (0,0,255),3)
    cv2.drawContours(img, [contours[bottom]], -1, (0,255,0),3)
    cv2.drawContours(img, [contours[right]], -1, (255,0,0),3)

    # 计算斜边的斜率
    slope,_ = point.lineSlope(mc[bottom],mc[right])
    # 获得三个识别框的四个角坐标
    L = getVertices(contours,top,slope,orientation)
    M = getVertices(contours,right,slope,orientation)
    O = getVertices(contours,bottom,slope,orientation)
    N = getIntersectionPoint([M[1],M[2],O[3],O[2]])

    # 显示识别点
    cv2.circle(img,(L[0][0],L[0][1]), 2, (0,255,0), 3)  # 绿色
    cv2.circle(img,(M[1][0],M[1][1]), 2, (0,0,255), 3)  # 红色
    cv2.circle(img,(O[3][0],O[3][1]), 2, (255,0,0), 3)  # 蓝色
    cv2.circle(img,(N[0],N[1]), 2, (21,32,155), 3)

    # 源区域和目标区域
    src = np.float32([L[0],M[1],N,O[3]])
    dst = np.float32([[0,0],[qr_size,0],[qr_size,qr_size],[0,qr_size]])
    if len(src)==4 and len(dst)==4:
      warp_matrix = cv2.getPerspectiveTransform(src, dst)
      qr = cv2.warpPerspective(gray,warp_matrix,(qr_size,qr_size))
      _,qr = cv2.threshold(qr, 104, 255, 0)
      cv2.imshow('qr',qr)
  cv2.imshow('image',img)

# 获取四个点形成的两条线的交点
def getIntersectionPoint(points):
  # 得到4个点的xy坐标
  x1,x2,x3,x4 = points[0][0],points[1][0],points[2][0],points[3][0]
  y1,y2,y3,y4 = points[0][1],points[1][1],points[2][1],points[3][1]
  
  x,y = 0.0, 0.0
  # 如果每条线上两个x都相等
  if x2==x1 and x3==x4:
    return [0,0]
  # 如果两个点的x值相等，代表这两个点所连的直线垂直坐标轴，交点的x坐标必然等于x
  if x2==x1:
    k = 1.0*(y4-y3)/(x4-x3)
    x = x1
    y = k*(x-x3)+y3
    return [x1,y3]
  elif x4==x3:
    k = 1.0*(y2-y1)/(x2-x1)
    x = x3
    y = k*(x-x1)+y1
    return [x3,y1]
  else:
    k1 = 1.0*(y2-y1)/(x2-x1)
    k2 = 1.0*(y4-y3)/(x4-x3)
    # k1 = k2时，没有交点
    if abs(k1-k2) < 0.01:
      return [0,0]
    x = (k1*x1-y1-k2*x3+y3) / (k1-k2)
    y = k1*(x-x1)+y1
  # print [x,y]
  return [int(abs(round(x))),int(abs(round(y)))]

def getVertices(contours, c_id, slope, orientation):
  x,y,w,h = cv2.boundingRect(contours[c_id])
  tl,br = (x,y),(x+w,y+h)
  A = [x,y]
  B = [x+w,y]
  C = [x+w,y+h]
  D = [x,y+h]
  W = [x+w/2, y]
  X = [x+w, y + h/2]
  Y = [x+w/2, y+h]
  Z = [x, y + h/2]
  dmax = [0.0,0.0,0.0,0.0]
  pd1,pd2 = 0.0,0.0
  M0,M1,M2,M3 = [0,0],[0,0],[0,0],[0,0]
  if abs(slope)>5:
    for i in xrange(0,len(contours[c_id])):
      pd1 = point.lineEquation(C,A,contours[c_id][i][0])
      pd2 = point.lineEquation(B,D,contours[c_id][i][0])
      if pd1>=0.0 and pd2>0.0:
        dmax[1],M1 = updateCorner(contours[c_id][i][0],W,dmax[1],M1)
      elif pd1 > 0.0 and pd2 <= 0.0:
        dmax[2],M2 = updateCorner(contours[c_id][i][0],X,dmax[2],M2)
      elif pd1 <= 0.0 and pd2 < 0.0:
        dmax[3],M3 = updateCorner(contours[c_id][i][0],Y,dmax[3],M3)
      elif pd1 < 0.0 and pd2 >= 0.0:
        dmax[0],M0 = updateCorner(contours[c_id][i][0],Z,dmax[0],M0)
      else:
        continue
  else:
    halfx = x+w/2
    halfy = y+h/2
    for i in xrange(0,len(contours[c_id])):
      temp_con = contours[c_id][i][0]
      tx,ty = temp_con[0],temp_con[1]
      
      # 左上角区域
      if tx<halfx and ty<=halfy:
        dmax[2],M0 = updateCorner(contours[c_id][i][0],C,dmax[2],M0)
      # 右上角
      elif tx>=halfx and ty<halfy:
        dmax[3],M1 = updateCorner(contours[c_id][i][0],D,dmax[3],M1)
      # 右下角
      elif tx>halfx and ty>=halfy:
        dmax[0],M2 = updateCorner(contours[c_id][i][0],A,dmax[0],M2)
      # 左下角
      elif tx<=halfx and ty>halfy:
        dmax[1],M3 = updateCorner(contours[c_id][i][0],B,dmax[1],M3)
  d = [M0,M1,M2,M3]

  if orientation == "North":
    return d
  elif orientation == "East":
    return [d[1],d[2],d[3],d[0]]
  elif orientation == "South":
    return [d[2],d[3],d[0],d[1]]
  elif orientation == "West":
    return [d[3],d[0],d[1],d[2]]
  else:
    return d

def updateCorner(p,ref,baseline,origin):
  temp_dist = point.distance(p,ref)
  if temp_dist > baseline:
    return temp_dist,p
  else:
    return baseline,origin

# 仅当三个识别图形的面积都大于limit时才认为可以识别
def canReconize(contours,rect,limit = 10):
  for x in xrange(1,len(rect)):
    if cv2.contourArea(contours[rect[x]]) < limit:
      return False
  return True
def findQrRectDirection(mc, rect):
  # 三条边的长度
  AB = point.distance(mc[rect[0]],mc[rect[1]])
  BC = point.distance(mc[rect[1]],mc[rect[2]])
  CA = point.distance(mc[rect[2]],mc[rect[0]])
  # 判断顶点
  outlier ,median1 ,median2 = 0,0,0
  if AB>BC and AB>CA:
    outlier ,median1 ,median2 = rect[2],rect[0],rect[1]
  elif CA>AB and  CA>BC:
    outlier ,median1 ,median2 = rect[1],rect[0],rect[2]
  elif BC > AB and BC > CA:
    outlier ,median1 ,median2 = rect[0],rect[1],rect[2]

  # 判断二维码方向
  p1,p2 = median1,median2
  orientation,top,bottom,right = "North",outlier,p1,p2

  # 顶点与中心点的相对位置
  pos = point.sub( mc[outlier],point.center(mc[p1],mc[p2]) )
  dx,dy = pos[0],pos[1]
  # 顶点在中点左上角
  if dx <= 0 and dy <= 0:
    orientation = "North"
  # 顶点在中点右上角
  elif dx > 0 and dy <= 0:
    bottom,right = p2,p1
    orientation = "East"
  # 顶点在中点右下角
  elif dx >= 0 and dy > 0:
    bottom,right = p2,p1
    orientation = "South"
  # 顶点在中点左下角
  elif dx < 0 and dy > 0:
    orientation = "West"

  return orientation,top,bottom,right


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
    rect = select_points(rect,contours)
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
