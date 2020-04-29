# Probability
In this module we learned:
- CDF 
- PMF, PDF
- Combinations, Permutations
- distributtions as Poisson, Normal, Exponential, Binomial
- mean, meadian, mode
- variance , standard Deviation

### Some Ecuations
| `PA`  |  `Possible Order` |
| ------------ | ------------ |
|  `#success / # total` | `n!`  |

| `Permutation`  | `Combination`  |
| ------------ | ------------ |
|   `(n!)/(n-x)!` |  `(n!)/(x! * (n-x!))` |

### PMF PDF CDF
- `CDF = sum of all PMF() from 0 to x if distribution is discrete` 
- `CDF = integral of PDF() evaulated from 0 to x if distribution is continuous` 
- `PMF = evaluate point in function of distribution`
- `PDF = evaluate point in function of distribution`

### Mean Mode and Median
![](https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/Visualisation_mode_median_mean.svg/150px-Visualisation_mode_median_mean.svg.png)
mean -- balance point
median -- divide the graph in two areas each one of 50%
mode -- peaks
# Tasks
In these tasks we are goint to build our own Binomial, Poisson, Normal and Exponential Distributions
# Poisson Distribution
- `lambtha = (sum(x)/total of elements)`
- `e = 2.7182818285`
- ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/75b15a85051528f8722d2b676a557f6c22bf85c4) - PMF
## 0-Initialize Poisson
Create Class constructor `def __init__(self, data=None, lambtha=1.)`
we have to use:

```
#!/usr/bin/env python3

import numpy as np
Poisson = __import__('poisson').Poisson

np.random.seed(0)
data = np.random.poisson(5., 100).tolist()
p1 = Poisson(data)
print('Lambtha:', p1.lambtha)

p2 = Poisson(lambtha=5)
print('Lambtha:', p2.lambtha)
```
and as result:
```
Lambtha: 4.84
Lambtha: 5.0
```
## 1-Poisson PMF 
create Instance method `def pmf(self, k):`

we have to use:

```
#!/usr/bin/env python3

import numpy as np
Poisson = __import__('poisson').Poisson

np.random.seed(0)
data = np.random.poisson(5., 100).tolist()
p1 = Poisson(data)
print('P(9):', p1.pmf(9))

p2 = Poisson(lambtha=5)
print('P(9):', p2.pmf(9))
```
and as result:
```
P(9): 0.03175849616802446
P(9): 0.036265577412911795
```
## 2-Poisson CDF 
Create Instance method `def cdf(self, k):`

we have to use:

```
#!/usr/bin/env python3

import numpy as np
Poisson = __import__('poisson').Poisson

np.random.seed(0)
data = np.random.poisson(5., 100).tolist()
p1 = Poisson(data)
print('F(9):', p1.cdf(9))

p2 = Poisson(lambtha=5)
print('F(9):', p2.cdf(9))
```
and as result:
```
F(9): 0.9736102067423525
F(9): 0.9681719426208609
```
# Exponential Distribution
- `lambtha = 1/((sum(x)/total of elements))`
- `e = 2.7182818285`
- ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/5196af323ae9f482da3f01a3c6f62c7b80cbc5b4) - PDF
- ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/6cc79b9dcb7538bec33cfc1dcd3c45b770ca2386) - CDF
## 3-Initialize Exponential
Create Class constructor `def __init__(self, data=None, lambtha=1.)`
we have to use:

```
#!/usr/bin/env python3

import numpy as np
Exponential = __import__('exponential').Exponential

np.random.seed(0)
data = np.random.exponential(0.5, 100).tolist()
e1 = Exponential(data)
print('Lambtha:', e1.lambtha)

e2 = Exponential(lambtha=2)
print('Lambtha:', e2.lambtha)
```
and as result:
```
Lambtha: 2.1771114730906937
Lambtha: 2.0
```
## 4-Exponential PDF 
create Instance method `def pdf(self, x):`

we have to use:

```
#!/usr/bin/env python3

import numpy as np
Exponential = __import__('exponential').Exponential

np.random.seed(0)
data = np.random.exponential(0.5, 100).tolist()
e1 = Exponential(data)
print('f(1):', e1.pdf(1))

e2 = Exponential(lambtha=2)
print('f(1):', e2.pdf(1))
```
and as result:
```
f(1): 0.24681591903431568
f(1): 0.2706705664650693
```
## 5-Exponential CDF 
Create Instance method `def cdf(self, x):`

we have to use:

```
#!/usr/bin/env python3

import numpy as np
Exponential = __import__('exponential').Exponential

np.random.seed(0)
data = np.random.exponential(0.5, 100).tolist()
e1 = Exponential(data)
print('F(1):', e1.cdf(1))

e2 = Exponential(lambtha=2)
print('F(1):', e2.cdf(1))
```
and as result:
```
F(1): 0.886631473819791
F(1): 0.8646647167674654
```

# Normal Distribution
- `lambtha = 1/((sum(x)/total of elements))`
- `e = 2.7182818285`
- `pi = 3.1415926536`
- `erf = (2*pi) * (x - ((x**3)/3) + ((x**5)/10) - ((x**7)/42) + ((x**9)/216))`
- ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/e91cc633835136bead4b126c0b4b3ec1e3bcc960) - PDF
- ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/187f33664b79492eedf4406c66d67f9fe5f524ea) - CDF
## 6-Initialize Normal
Create Class constructor `def __init__(self, data=None, mean=0., stddev=1.):`
we have to use:

```
#!/usr/bin/env python3

import numpy as np
Normal = __import__('normal').Normal

np.random.seed(0)
data = np.random.normal(70, 10, 100).tolist()
n1 = Normal(data)
print('Mean:', n1.mean, ', Stddev:', n1.stddev)

n2 = Normal(mean=70, stddev=10)
print('Mean:', n2.mean, ', Stddev:', n2.stddev)
```
and as result:
```
Mean: 70.59808015534485 , Stddev: 10.078822447165797
Mean: 70.0 , Stddev: 10.0
```
## 7-Normalize Normal
create Instance method `def z_score(self, x):` 
create also Instance method `def x_value(self, z):`

we have to use: 
```
#!/usr/bin/env python3

import numpy as np
Normal = __import__('normal').Normal

np.random.seed(0)
data = np.random.normal(70, 10, 100).tolist()
n1 = Normal(data)
print('Z(90):', n1.z_score(90))
print('X(2):', n1.x_value(2))

n2 = Normal(mean=70, stddev=10)
print()
print('Z(90):', n2.z_score(90))
print('X(2):', n2.x_value(2))
```
as result 
```
Z(90): 1.9250185174272068
X(2): 90.75572504967644

Z(90): 2.0
X(2): 90.0
```
## 8-Normal PDF 
create Instance method `def pdf(self, x):`

we have to use:

```
#!/usr/bin/env python3

import numpy as np
Normal = __import__('normal').Normal

np.random.seed(0)
data = np.random.normal(70, 10, 100).tolist()
n1 = Normal(data)
print('PSI(90):', n1.pdf(90))

n2 = Normal(mean=70, stddev=10)
print('PSI(90):', n2.pdf(90))
```
and as result:
```
PSI(90): 0.006206096804434349
PSI(90): 0.005399096651147344
```
## 9-Normal CDF 
Create Instance method `def cdf(self, x):`

we have to use:

```
#!/usr/bin/env python3

import numpy as np
Normal = __import__('normal').Normal

np.random.seed(0)
data = np.random.normal(70, 10, 100).tolist()
n1 = Normal(data)
print('PHI(90):', n1.cdf(90))

n2 = Normal(mean=70, stddev=10)
print('PHI(90):', n2.cdf(90))
```
and as result:
```
PHI(90): 0.982902011086006
PHI(90): 0.9922398930667251
```
# Binomial Distribution
- MEAN = ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/3d6eb41e0e5e136f594b1a703d2f371d9a5e0c27) 
- MEAN = `sum(x)/ total elements`
- Variance = ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/bf5b4b6e591c413e746d1ba867277e99b9b083b9)
- Variance = `(sum(((x - MEAN) ** 2)) / Total element`
- P = `MEAN/ n` with a fixed n

- ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/20edfc22372742d64909cf7c7f97593bade88338) - PMF

## 10-Initialize Exponential
Create Class constructor `def __init__(self, data=None, n=1, p=0.5):`
we have to use:

```
#!/usr/bin/env python3

import numpy as np
Binomial = __import__('binomial').Binomial

np.random.seed(0)
data = np.random.binomial(50, 0.6, 100).tolist()
b1 = Binomial(data)
print('n:', b1.n, "p:", b1.p)

b2 = Binomial(n=50, p=0.6)
print('n:', b2.n, "p:", b2.p)
```
and as result:
```
n: 50 p: 0.606
n: 50 p: 0.6
```
## 11-Binomial PMF
create Instance method `def pmf(self, k):`

we have to use:

```
#!/usr/bin/env python3

import numpy as np
Binomial = __import__('binomial').Binomial

np.random.seed(0)
data = np.random.binomial(50, 0.6, 100).tolist()
b1 = Binomial(data)
print('P(30):', b1.pmf(30))

b2 = Binomial(n=50, p=0.6)
print('P(30):', b2.pmf(30))
```
and as result:
```
P(30): 0.11412829839570347
P(30): 0.114558552829524
```
## 12-Binomial CDF 
Create Instance method `def cdf(self, k):`

we have to use:

```
#!/usr/bin/env python3

import numpy as np
Binomial = __import__('binomial').Binomial

np.random.seed(0)
data = np.random.binomial(50, 0.6, 100).tolist()
b1 = Binomial(data)
print('F(30):', b1.cdf(30))

b2 = Binomial(n=50, p=0.6)
print('F(30):', b2.cdf(30))
```
and as result:
```
F(30): 0.5189392017296368
F(30): 0.5535236207894576
```
