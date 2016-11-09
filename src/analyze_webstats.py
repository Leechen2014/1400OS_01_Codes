#!/usr/bin/env python
# -*-coding:UTF-*-

'''
env python 2.7
    ubuntu 12.04
'''
import os
import scipy as sp
import matplotlib.pyplot as plt

##read data  step 1==>
data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")
data = sp.genfromtxt(os.path.join(data_dir, "web_traffic.tsv"), delimiter='\t')

colors = ['g', 'k', 'b', 'm', 'r']
linestyles = ['-', '-.', '--', '-.', '-']

###print data
print("x is ")  ##1D vector
print(data[:, 0])  ##1D vector
print("y is ")  ##1D vector
print(data[:, 1])  ##1D vector
##step 2 data pre-process

x = data[:, 0]
y = data[:, 1]

print("length x",len(x))
print("length y",len(y))

print("Number of invalid entries x: ", sp.sum(sp.sum(sp.isnan(x))))
print("Number of invalid entries y: ", sp.sum(sp.sum(sp.isnan(y))))
## flter error data
x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]


## plot data define
def plot_models(x, y, models, fname, mx=None, ymax=None, xmin=None):
    plt.clf()  # clear current figure

    print('length of x :', len(x))
    print('length of y :', len(y))

    plt.scatter(x, y)  # make a scatter plit of x y , where x and y are sequnce like obj of the same length
    plt.title("Web traffic over the last month")  # set a title of the current axes

    plt.xlabel("Time")
    plt.ylabel("Hits/hours")

    plt.xticks([w * 7 * 24 for w in range(10)], ['week %i' % w for w in range(10)]) #get or set x-limites of current tick localtion and lables


    if models:
        if mx is None:
            mx = sp.linspace(0, x[-1], 1000) # start=0,end=x[-1],number=1000 mx is a array ,element is range[start,end]
        for model, style, color, in zip(models, linestyles, colors):
            plt.plot(mx, model(mx), linestyle=style, linewidth=2, c=color)# plot point
        """
          place a legend , loc is location
        """
        plt.legend(["d=%i" % m.order for m in models], loc="upper left")

    plt.autoscale(tight=True)
    plt.ylim(ymin=0)

    if ymax:
        plt.ylim(ymax=ymax)
    if xmin:
        plt.xlim(xmin=xmin)

    plt.grid(True, linestyle='-', color='0.75') # turn the axes grids on or off
    plt.savefig(fname) # save fig by name
    ##end function

## first look at the data
plot_models(x, y, None, os.path.join("..", "1400_01_0.png"))

# create and plot models
fp1, res, rank, sv, rcond = sp.polyfit(x, y, 1, full=True)#last squares polynormaial fit
print("model parameters: %s" % fp1)
print("Error of the model : ", res)
f1 = sp.poly1d(fp1)
f2 = sp.poly1d(sp.polyfit(x, y, 2)) #degree =2
f3 = sp.poly1d(sp.polyfit(x, y, 3))
f10 =sp.poly1d(sp.polyfit(x, y, 10))
f100=sp.poly1d(sp.polyfit(x, y, 100))
#  plot img
plot_models(x, y, [f1], os.path.join("..", "1400_01_02.png"))
plot_models(x, y, [f1,f2], os.path.join("..", "1400_01_03.png"))
plot_models(x, y, [f1,f2,f3,f10,f100] , os.path.join("..","1400_01_04.png"))

## fit and plot a model using the konwledge about inflection point
inflection = 3.5 * 7 * 24
xa = x[:inflection]
ya = y[:inflection]
xb = x[inflection:]
yb = y[inflection:]

fa = sp.poly1d(sp.polyfit(xa, ya, 1))
fb = sp.poly1d(sp.polyfit(xb, yb, 1))

plot_models(x, y, [fa, fb], os.path.join("..","1400_01_05.png"))

def error(f, x, y):
    return sp.sum((f(x)-y)**2)

print("Error for the complete data set : ")

for f in [f1,f2,f3,f10,f100]:
    print("Error d=%i %f "%(f.order, error(f, x, y)))

print("Error for only the time after inflection point ")
for f in [f1,f2,f3,f10,f100]:
    print("Error d=%i %f" % (f.order, error(f, xb, yb)))

print("Error inflection=%f " % (error(fa,xa,ya)+error(fb,xb,yb)))

## extraplating into future
plot_models(
    x,y,[f1,f2,f3,f10,f100],os.path.join("..","1400_01_06.png"),
    mx=sp.linspace(0*7*24 , 6*7*24, 100),
    ymax=10000, xmin=0*7*24
)


##separating training form testing data
frac=0.3
split_idx=int(frac*len(xb))
shuffled = sp.random.permutation(list(range(len(xb))))
test = sorted(shuffled[:split_idx])
train = sorted(shuffled[split_idx:])
fbt1 = sp.poly1d(sp.polyfit(xb[train],yb[train],1))
fbt2 = sp.poly1d(sp.polyfit(xb[train],yb[train],2))
fbt3 = sp.poly1d(sp.polyfit(xb[train],yb[train],3))
fbt10 = sp.poly1d(sp.polyfit(xb[train],yb[train],10))
fbt100 = sp.poly1d(sp.polyfit(xb[train],yb[train],100))

print("Trained only on data after inflection point ")
for f in [fbt1, fbt2, fbt3, fbt10, fbt100]:
    print("Error d=%i : %f"%(f.order,error(f,xb[test],yb[test])))

plot_models(
    x,y,[fbt1,fbt2,fbt3,fbt10,fbt100],os.path.join("..","1400_01_08.png"),
    mx=sp.linspace(0*7*24, 6*7*24,100),
    ymax=10000,xmin=0*7*24)

from scipy.optimize import fsolve
print(fbt2)
reached_max = fsolve(fbt2 -100000,800)/(7*24)
print("100 ,000 hits/hour expected at week %f " %reached_max[0])






