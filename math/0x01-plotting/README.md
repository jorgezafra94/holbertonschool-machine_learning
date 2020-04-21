# PLOTTING

plot works for see some data sets in a graphic way
with this tool we are going to visualize all the data in a graphic
and in that way understand what we have to do with that and also
see its bevahiour

if you want to configure your windows 10, with Vagrant and VirtualBox
please check this [blog](https://medium.com/@jcook0017/how-to-enable-x11-forwarding-in-windows-10-on-a-vagrant-virtual-box-running-ubuntu-d5a7b3\
4363f)

## MATPLOTLIB
this is a module that help us to realice the graphics of some data sets
the main graphics forms are:
- Plot
- Scatter
- Historiam
- Bars

### Line Styles
some line styles for graphics
- `'-'` solid line
- `'--'` dashed line
- `'-.'` dash-dot line
- `':'` dotted line

### Markers
some markers for graphics
- `'.'` point
- `'o'` circle
- `'^'` up triangle
- `'v` down triangle
- `'+'` plus symbol
- `'x'` x symbol
- `'*'` star symbol
- `'s'` square
- `'d'` diamond

### Colors
somo colors for graphics
- `'b'` blue
- `'g'` green
- `'r'` red
- `'c'` cyan
- `'m'` magenta
- `'y'` yellow
- `'k'` black
- `'w'` white

## TASKS
### 0-PLOT
In this task we are goint to use:
- `plt.plot()` to graphic
- `plt.xlim()` to limit the x axis
- `plt.show()` to print the graphic
we have to use this

```
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

# your code here
```
In order to get this graphic
![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/math/0x01-plotting/graphics/picture0.png)

### 1-SCATTER
In this task we are goint to use:
- `plt.scatter()` to graphic
- `plt.xlabel and plt.ylabel` to give a title to the axis
- `plt.title()` to give a title to the graphic
- `plt.show()` to print the graphic
we have to use this

```
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180

# your code here
```
In order to get this graphic
![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/math/0x01-plotting/graphics/picture1.png)
### 2-plot with scale
In this task we are goint to use:
- `plt.plot()` to graphic
- `plt.xlim()` to limit the x axis
- `plt.xlabel() and plt.ylabel()` to give a name to axis
- `plt.yscale()` to give a format to scale in y axis
- `plt.title()` to give a title to the graphic
- `plt.show()` to print the graphic

some scales are:
- linear
- log
- symlog
- logit

we have to use this


```
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

# your code here
```
In order to get this graphic
![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/math/0x01-plotting/graphics/picture2.png)
### 3-plotting 2 graphics
In this task we are goint to use:
- `plt.plot()` to graphic
- `plt.xlim() and plt.ylim()` to limit the axis
- `plt.xlabel() and plt.ylabel()` to give a name to axis
- `plt.legend()` to print a legend inside the graphic, it works with the labels of each plots
- `plt.title()` to give a title to the graphic
- `plt.show()` to print the graphic
we have to use this

```
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

# your code here
```
In order to get this graphic
![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/math/0x01-plotting/graphics/picture3.png)
### 4-HISTORIAM
In this task we are goint to use:
- `plt.hist()` to graphic
- `plt.xlim() and plt.ylim()` to limit the axis
- `plt.xlabel() and plt.ylabel()` to give a name to axis
- `plt.xticks()` to create the ticks in x axis
- `plt.title()` to give a title to the graphic
- `plt.show()` to print the graphic
we have to use this

```
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# your code here
```
In order to get this graphic
![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/math/0x01-plotting/graphics/picture4.png)
### 5-subplots
In this task we are goint to use:
- `fig = plt.figure()` to create a figura that contains all the subplots
- `ax1 = plt.subplot2grid()` to create the different subplots, first we have to define the space and then in wich position the new subplot is located
- `ax1.plot() or ax1.scatter() or ax1.hist()` to create subplot graphic
- `ax1.set_title()` set a title to subplot
- `ax1.set_xlabel() and ax1.set_ylabel()`
to set a label in x and y for subplot graphic
- `ax1.set_yscale()` to set scale in a axis of the subplot
- `ax1.set_xlim() and ax1.set_ylim()` to set limits in subplot
- `ax1.legends()` to create a legend in the subplot
- `ax1.set_xticks()` to put ticks in axis of subplot
- `fig.suptitle()` title of complete figure
- `fig.subplots_adjust()` to set distances between subplots inside figure
- `plt.show()` print graphics

we have to use this

```
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# your code here
```
In order to get this graphic
![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/math/0x01-plotting/graphics/picture5.png)
### 6-BARS
In this task we are goint to use:
- `plt.bart()` to graphic
- `plt.ylabel()` to label axis
- `plt.yticks()` to create ticks in y axis
- `plt.title()` to create a title for the graphic
- `plt.legend()` to create a legend in the graphic
- `plt.show()` to print the graphic
we have to use this

```
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

# your code here
```
In order to get this graphic
![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/math/0x01-plotting/graphics/picture6.png)
### 100-colorbar
In this task we are goint to use:
- `plt.scatter()` to graphic
- `plt.xlabel() and plt.ylabel()` to label the axis
- `plt.title()` to set a title
- `plt.colorbar()` to create a colorbar with colors of scatter
- `plt.show()` to print the graphic
we have to use this

```
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

x = np.random.randn(2000) * 10
y = np.random.randn(2000) * 10
z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

# your code here
```
In order to get this graphic
![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/math/0x01-plotting/graphics/picture100.png)
### 101- 3D graphic
In this task we are goint to use:
- `figure = plt.figure()` to create a figure that contains subplots
- `ax = fig.add_subplot(111, projection= '3d')` to create 3D graphic
- `ax.set_xlabel() and ax.set_ylabel() and ax.zlabel()` to label all axis
- `ax.scatter()` to graphic
- `plt.title()` to set a title
- `plt.show()` to print the graphic

here we have to change the color map and we did it only putting cmap = plt.cm.plasma, with this we can use array of values for set colors
we have to use this

```
#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

# your code here
```
In order to get this graphic
![](https://github.com/jorgezafra94/holbertonschool-machine_learning/blob/master/math/0x01-plotting/graphics/picture101.png)
