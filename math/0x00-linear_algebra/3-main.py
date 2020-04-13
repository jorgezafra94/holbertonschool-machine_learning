#!/usr/bin/env python3

matrix_transpose = __import__('3-flip_me_over').matrix_transpose

mat1 = [[1, 2], [3, 4]]
print(mat1)
print(matrix_transpose(mat1))
mat2 = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]
print(mat2)
print(matrix_transpose(mat2))
mat3 = [1, 2, 3, 4]
print(mat3)
print(matrix_transpose(mat3))
mat4 = [[1], [2], [3], [4]]
print(mat4)
print(matrix_transpose(mat4))
mat5 = [[1]]
print(mat5)
print(matrix_transpose(mat5))
mat6 = [1]
print(mat6)
print(matrix_transpose(mat6))
mat7 = [[]]
print(mat7)
print(matrix_transpose(mat7))
