import numpy as np

def gaussian_elimination(A, B):
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float).reshape(-1, 1)
    
    n = A.shape[1]
    m = A.shape[0]

    # Copy A and B to oldA and oldB
    oldA = np.array(A)
    oldB = np.array(B)

    # Perform Gaussian elimination
    for i in range(min(n, m) - 1):
        if A[i, i] == 0:
            # Find the maximum element in the column
            k = np.argmax(np.abs(A[i + 1:, i])) + i + 1

            # Swap rows i and k in matrix A and vector B
            A[[i, k], :] = A[[k, i], :]
            B[[i, k], :] = B[[k, i], :]

        for j in range(i + 1, m):
            if A[i, i] == 0:
                continue  # Skip the division if A[i, i] is still zero
            c = A[j, i] / A[i, i]
            A[j, i:] -= c * A[i, i:]
            B[j] -= c * B[i]

        # Back-substitution
        x = np.zeros(n)
        for i in range(min(n, m) - 1, -1, -1):
         if A[i, i] == 0:
            continue  # Skip the division if A[i, i] is still zero
         x[i] = (B[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]



    return oldA, oldB, A, B, x

# Example usage
rowsA = int(input("Enter the number of rows of matrix A: "))
colsA = int(input("Enter the number of columns of matrix A: "))

A = []
print("Enter matrix A elements:")
for i in range(rowsA):
    row = [float(input(f"element [{i + 1}][{j + 1}]: ")) for j in range(colsA)]
    A.append(row)

sizeB = rowsA
B = [float(input(f"Element {i + 1}: ")) for i in range(sizeB)]

if sizeB != rowsA:
    print("Error: The number of rows in matrix A must be equal to the size of vector B.")
else:
    oldA, oldB, newA, newB, solution = gaussian_elimination(A, B)

    print("\nMatrices before elimination are:")
    print(oldA)
    print(oldB)

    print("\nMatrices after elimination are:")
    print(newA)
    print(newB)

    print("\nThe solution of the system is:")
    print("x =", solution)
