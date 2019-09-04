import turtle
from time import sleep

import numpy as np

WIDTH, HEIGHT = 10, 10
FONT = ('Arial', 8, 'normal')


def draw(el, l, p, r):

    myPen = turtle.Turtle()
    myPen.hideturtle()
    myPen.speed(0)
    myPen.width(3)

    window = turtle.Screen()
    window.clear()
    window.colormode(255)
    window.title("Trilateration")
    scale = 30

    for i in p:
        myPen.color(tuple(np.random.choice(range(256), size=3)))
        myPen.penup()
        myPen.goto(scale*p[i][0]-5, scale*p[i][1])
        myPen.pendown()
        myPen.goto(scale*p[i][0]+5, scale*p[i][1])
        myPen.penup()
        myPen.goto(scale*p[i][0], scale*p[i][1]-5)
        myPen.pendown()
        myPen.goto(scale*p[i][0], scale*p[i][1]+5)
        myPen.penup()

        myPen.write(i, True, font=FONT)
        myPen.penup()

        myPen.goto(scale*p[i][0], scale*(p[i][1]-r[i]))
        myPen.pendown()
        myPen.circle(scale*r[i])
        myPen.penup()


# Estimated localization
    myPen.color("#800080")
    myPen.goto(scale*el[0]-5, scale*el[1])
    myPen.pendown()
    myPen.goto(scale*el[0]+5, scale*el[1])
    myPen.penup()
    myPen.goto(scale*el[0], scale*el[1]-5)
    myPen.pendown()
    myPen.goto(scale*el[0], scale*el[1]+5)
    myPen.penup()

    myPen.write("x0", True, font=FONT)
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

    myPen.write("x1", True, font=FONT)
    myPen.penup()

#     window.exitonclick()
    sleep(2)
