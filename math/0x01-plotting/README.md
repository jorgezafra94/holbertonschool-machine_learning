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
