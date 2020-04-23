# CALCULUS
As we already know, Machine Learning is based on Linear Algebra, Calculus and Probability mixed with code lines.

So here in this module we are going to learn Pee and Seegma Notation, Derivative and Integrals

### SEEGMA
Seegma is used to represent consecutive additions, from one number to another, putting these values through a function.

![](http://latex.codecogs.com/gif.latex?\sum_{i=2}^{5}&space;i)

in the case of the above picture we have a function `f(i) = i` a initial value `i=2` and a top value where the addition is going to stop, here `5`. So the result of the seegma notation in this case will be:
`2 + 3 + 4 + 5`

### PEE
Pee notation is used to represent consecutive products, it works the same like Seegma notation

![](http://latex.codecogs.com/gif.latex?\prod_{i&space;=&space;1}^{m}&space;i)

in the case of the above picture we have a function `f(i) = i` a initial value `i=1` and a top value where the addition is going to stop, here `m`. So the result of the seegma notation in this case will be:
`1 * 2 * 3 * 4 * 5 ..... * m`

### DERIVATIVE
The derivative of a function of a single variable at a chosen input value, when it exists, is the slope of the tangent line to the graph of the function at that point. The tangent line is the best linear approximation of the function near that input value.

![](http://latex.codecogs.com/gif.latex?\frac{dy}{dx}) ![](http://latex.codecogs.com/gif.latex?y&space;=&space;x^4&space;+&space;3x^3&space;-&space;5x&space;+&space;1)

the derivate of `x^4 = 4 * x ^(4 - 1) = 4(x^3)`

Some classic derivatives: [common derivatives](https://www.coastal.edu/media/academics/universitycollege/mathcenter/handouts/calculus/deranint.PDF)

<b>Rules of derivate of more than one function:</b>
f and g are functions
- `d/dx (f + g) = d/dx(f) + d/dx(g)`
- `d/dx (f * g) = (d/dx(f) * g) + (f * d/dx(g))`
- `d/dx (f / g) = ((d/dx(f) * g) - (f * d/dx(g)))/(g^2)`

#### Chain Rule
The chain rule is simple, it is just the external derivative product the internal one
`d/dx (ln(x^2))` derivada externa es `d/du ln(u)` where `u = x ^ 2` product  
`d/dx (x ^ 2)` so the result should be:
`d/dx (ln(x^2)) = (1/(x^2)) * (2x)` ----- external product internal

#### Partial derivatives
It is used when we want to derivate a function that has more than one variable, so in a partial derivative the others variables work as constants.

`d/ dx (e^((x^2)*y)) = (e^((x^2) * y)) * (2*x*y)`

### INTEGRAL
can be interpreted informally as the signed area of the region in the xy-plane that is bounded by the graph of f, the x-axis and the vertical lines x = a and x = b. The area above the x-axis adds to the total and that below the x-axis subtracts from the total.
also know as the Anti-derivative

![](http://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/6/ada047ad4cbee23dfed8.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200423%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200423T125623Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=f7ccdf9a2a9dce058f76cb80d2c48091969d599ebba126ec946da17eb45eed7d)

the integral is `((x ^ (3 + 1)) * (1/(3 + 1))) = (x^4 / 4)`

#### Definite Integral
![](http://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/6/b94ec3cf3ae61acd0275.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200423%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200423T125624Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=6e74f14dc15b7ad9e4f0a4af6a70b72c6ebae288f255e32a90f7230fe2892d67)

it is `integral evaluated in 3 - integral evaluated in 0` in this case

# TASKS
 ## 0-8 Answers
 resolve manually some seegma, peee and derivative functions
 ## 9 - Seegma Program
 create a program that works as seegma without loops
 using as function `i ^ 2`

 ```
 #!/usr/bin/env python3

summation_i_squared = __import__('9-sum_total').summation_i_squared

n = 5
print(summation_i_squared(n))
 ```

 the result should be
 
 ```
./9-main.py 
55 
```
## 10- Derivate Program
the idea is to create a program that can resolve the derivate of a polynomial using lists
each position of the input array means the exponential of the polynomial in this case we are passing as input `(x^3) + 3x + 5` = `[5, 3, 0, 1]`
```
#!/usr/bin/env python3

poly_derivative = __import__('10-matisse').poly_derivative

poly = [5, 3, 0, 1]
print(poly_derivative(poly))
```
the result should be:
```
[3, 0, 3]
```
 ## 11-16 Answers
 some answers of Integral problems
 ## 17- Integral Program
 the idea is to create a program that can resolve the integral of a polynomial using lists
each position of the input array means the exponential of the polynomial for example if we pass as input `(x^3) + 3x + 5` = `[5, 3, 0, 1]`
```
#!/usr/bin/env python3

poly_integral = __import__('17-integrate').poly_integral

poly = [5, 3, 0, 1]
print(poly_integral(poly))
```
the result should be:
if the float is a whole number we have to pass it to integer
```
[0, 5, 1.5, 0, 0.25]
```
