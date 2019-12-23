import numpy as np
import matplotlib.pyplot as plt
import math

def lagranz(x, y, t):
    z = 0
    for j in range(len(y)):
        p1 = 1
        p2 = 1
        for i in range(len(x)):
            if i == j:
                p1 = p1 * 1
                p2 = p2 * 1
            else:
                p1 = p1 * (t - x[i])
                p2 = p2 * (x[j] - x[i])
        z = z + y[j] * p1 / p2
    return z


def ChebyshevNodes(a,b,n):
    nodes = []
    for i in range(n):
        node = (a+b)/2 + ((a-b)/2)*math.cos((2*i + 1)*math.pi/(2*n + 2))
        nodes.append(node)
    return nodes


#without count of values
def standartDeviationV1(y_real,y_target):
    s = 0
    for i in range(len(y_real)):
        s += (y_real[i] - y_target[i]) ** 2
    return s

def standartDeviationV2(y_real,y_target):
    s = 0
    for i in range(len(y_real)):
        s += (y_real[i] - y_target[i]) ** 2
    return math.sqrt(s/len(y_real))


def f(x):
    if (x < -2):
        return 29/3 + (4*x)/3
    if (x >= -2 and x < 2):
        return 18/4 - (5*x)/4
    if (x >= 2):
        return 4/6 + (4*x)/6

#Variant 1
a = -5
b = 8
n = int(input("Enter number of nodes: "))
#equal distribution
h = abs(b - a)/n

y_real = [f(i) for i in np.arange(a,b,0.1,dtype = float)]

x_equal_dist = np.arange(a,b,h,dtype = float)
y_equal_dist = [f(i) for i in x_equal_dist]


x_equal_dist_largranz = np.arange(a,b,0.1,dtype = float)
y_equal_dist_lagranz = [lagranz(x_equal_dist,y_equal_dist,i) for i in x_equal_dist_largranz]



#Chebyshev nodes

x_chebyshev_nodes = ChebyshevNodes(a,b,n)
y_chebyshev = [f(i) for i in x_chebyshev_nodes]

x_chebyshev_lagranz = np.arange(a,b,0.1,dtype = float)
y_chebyshev_lagranz = [lagranz(x_chebyshev_nodes,y_chebyshev,i) for i in x_chebyshev_lagranz]




#Standart deviation graph
figdev,stddevplot = plt.subplots()
stddevplot.set_title("Standart deviation")

n_arr = []
#equal distribution
stddev_eq = []
#Chebyshev nodes
stddev_chebyshev = []
for i in range(2,13):
    #step
    h_dev = abs(b-a)/i
    #x array with from a to b with step h_dev
    x_dev = np.arange(a,b,h_dev,dtype = float)
    y_dev = [f(j) for j in x_dev]

    #lagranz's polinom with equal distribution
    x_real = np.arange(a,b,0.1,dtype = float)
    y_dev_lagranz = [lagranz(x_dev,y_dev,k) for k in x_real]

    #Chebyshev nodes
    x_dev_chebyshev = ChebyshevNodes(a,b,i)
    y_dev_chebysev = [f(j) for j in x_dev_chebyshev]

    #Lagranz's polinom with Chebyshev nodes
    y_dev_chebysev_lagranz = [lagranz(x_dev_chebyshev,y_dev_chebysev,k) for k in x_real]

    #Relation deviation from n
    n_arr.append(i)
    stddev_eq.append(standartDeviationV2(y_real,y_dev_lagranz))

    stddev_chebyshev.append(standartDeviationV2(y_real,y_dev_chebysev_lagranz))

stddevplot.plot(n_arr,stddev_eq,'r',color="red",label="Equal distribute")
stddevplot.plot(n_arr,stddev_chebyshev,'b',color="blue",label="Chebyshev nodes")
stddevplot.legend()

fig,lgplot = plt.subplots()
lgplot.set_title("Lagranz's polinom")

#Lagranz's polinom with equal distribution
lgplot.plot(x_equal_dist_largranz,y_equal_dist_lagranz,'r',x_equal_dist,y_equal_dist,'ro',label= "Equal distribute")

#Lagranz's polinom with Chebyshev nodes
lgplot.plot(x_chebyshev_lagranz,y_chebyshev_lagranz,'b',x_chebyshev_nodes,y_chebyshev,'bs',label="Chebyshev nodes")

#real
lgplot.plot(x_equal_dist_largranz,y_real,'g',label="Real function")
lgplot.legend()

plt.grid(True)
plt.show()