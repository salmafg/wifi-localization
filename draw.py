import turtle
import random
import numpy as np

WIDTH, HEIGHT = 10, 10
FONT = ('Arial', 8, 'normal')


def draw(el, l, m):

    myPen = turtle.Turtle()
    myPen.hideturtle()
    myPen.speed(0)
    myPen.width(2)

    window = turtle.Screen()
    window.bgcolor("#F0F0F0")
    window.colormode(255)
    window.title("Trilateration")
    scale = 60

    for i in range(int(len(m)/2)):
        myPen.color(tuple(np.random.choice(range(256), size=3)))
        myPen.penup()
        myPen.goto(scale*m["P{0}".format(i+1)][0]-5, scale*m["P{0}".format(i+1)][1])
        myPen.pendown()
        myPen.goto(scale*m["P{0}".format(i+1)][0]+5, scale*m["P{0}".format(i+1)][1])
        myPen.penup()
        myPen.goto(scale*m["P{0}".format(i+1)][0], scale*m["P{0}".format(i+1)][1]-5)
        myPen.pendown()
        myPen.goto(scale*m["P{0}".format(i+1)][0], scale*m["P{0}".format(i+1)][1]+5)
        myPen.penup()

        myPen.write(str(m["P{0}".format(i+1)]), True, font=FONT)
        myPen.penup()

        myPen.goto(scale*m["P{0}".format(i+1)][0], scale*(m["P{0}".format(i+1)][1]-m["r{0}".format(i+1)]))
        myPen.pendown()
        myPen.circle(scale*m["r{0}".format(i+1)])
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

    window.exitonclick()
