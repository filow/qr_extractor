# coding: utf-8
import math
# 提供两个点的处理函数库

# 新建一个点
def new(x,y):
  return [x,y]

# 两个点之间的距离
def distance(p1, p2):
  return math.sqrt(pow(abs(p1[0] - p2[0]),2) + pow(abs(p1[1] - p2[1]),2))

# 点1减去点2
def sub(p1, p2):
  return [p1[0]-p2[0],p1[1]-p2[1]]

# 取两个点的中点
def center(p1, p2):
  return [(p1[0]+p2[0])/2,(p1[1]+p2[1])/2]

# 获得在正常xy坐标系下的斜率（与cv中的坐标系算出的斜率是互为相反数的关系）
def lineSlope(p1, p2):
  dx = p1[0] - p2[0]
  dy = p1[1] - p2[1]
  if abs(dx) > 0.01:
    return -dy/dx,1
  else:
    return 0.0,0

def lineEquation(line1,line2,p):
  formula = (p[1]-line2[1])*(line1[0]-line2[0]) - (p[0]-line2[0])*(line1[1]-line2[1])
  a = line2[1]-line1[1]
  b = line1[0]-line2[0]
  if abs(a+b)<0.01:
    return 0
  dist = abs(formula)/math.sqrt(a*a + b*b)
  return dist