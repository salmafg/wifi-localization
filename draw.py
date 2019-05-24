import turtle
from random import randint

WIDTH, HEIGHT = 10, 10

def draw_trilateration(x1, y1, r1, x2, y2, r2, x3, y3, r3, l):
  
  myPen = turtle.Turtle()
  myPen.hideturtle()
  myPen.speed(0)
  myPen.width(2)
  
  window = turtle.Screen()
  window.bgcolor("#F0F0F0")
  window.title("Trilateration")
  scale = 50

  # Red center
  myPen.color("#ff5744")
  myPen.penup()
  myPen.goto(scale*x1-5, scale*y1)
  myPen.pendown()
  myPen.goto(scale*x1+5, scale*y1)
  myPen.penup()
  myPen.goto(scale*x1, scale*y1-5)
  myPen.pendown()
  myPen.goto(scale*x1, scale*y1+5)
  myPen.penup()
  
  # Red circle
  myPen.goto(scale*x1, scale*(y1-r1))
  myPen.pendown()
  myPen.circle(scale*r1)
  myPen.penup()
  
  # Green center
  myPen.color("#52bf54")
  myPen.penup()
  myPen.goto(scale*x2-5, scale*y2)
  myPen.pendown()
  myPen.goto(scale*x2+5, scale*y2)
  myPen.penup()
  myPen.goto(scale*x2, scale*y2-5)
  myPen.pendown()
  myPen.goto(scale*x2, scale*y2+5)
  myPen.penup()
  
  # Green circle
  myPen.goto(scale*x2, scale*(y2-r2))
  myPen.pendown()
  myPen.circle(scale*r2)
  myPen.penup()

  # Blue center
  myPen.color("#41befc")
  myPen.goto(scale*x3-5, scale*y3)
  myPen.pendown()
  myPen.goto(scale*x3+5, scale*y3)
  myPen.penup()
  myPen.goto(scale*x3, scale*y3-5)
  myPen.pendown()
  myPen.goto(scale*x3, scale*y3+5)
  myPen.penup()
  
  # Blue circle
  myPen.goto(scale*x3, scale*(y3-r3))
  myPen.pendown()
  myPen.circle(scale*r3)
  myPen.penup()

  # Localization
  myPen.color("#000000")
  myPen.goto(scale*l[0]-5, scale*l[1])
  myPen.pendown()
  myPen.goto(scale*l[0]+5, scale*l[1])
  myPen.penup()
  myPen.goto(scale*l[0], scale*l[1]-5)
  myPen.pendown()
  myPen.goto(scale*l[0], scale*l[1]+5)
  myPen.penup()
  
  window.exitonclick()
  window.update()