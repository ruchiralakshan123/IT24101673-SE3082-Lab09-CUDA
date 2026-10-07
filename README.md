# SE3082 Lab 09 - Introduction to CUDA

**Student name:** Senarath D.M.G.R.L
**Student ID:** IT24101673

This repository contains the CUDA practical exercises completed for **SE3082 -
Parallel Computing Lab 09**. The exercises demonstrate how to write GPU
kernels, allocate and transfer memory between the host (CPU) and device (GPU),
configure CUDA grids and blocks, and verify GPU results on the CPU.

The programs were tested in **Google Colab** using an NVIDIA **Tesla T4 GPU**.
The original Colab notebook is included as
[`SE3082_Lab09_IT24101673.ipynb`](./SE3082_Lab09_IT24101673.ipynb).

## Learning outcomes

This lab covers:

- Compiling and running a basic CUDA program.
- Launching kernels with different grid and block configurations.
- Using `cudaMalloc`, `cudaMemcpy`, and `cudaFree`.
- Mapping CUDA thread and block indices to array and matrix elements.
- Handling large vectors and two-dimensional data.
- Verifying device results against the expected host-side calculation.
- Understanding the maximum number of threads supported by a block.

## Project structure

| Folder/file | Description |
| --- | --- |
| [`Exercise 1/`](./Exercise%201/) | Basic GPU kernel that prints a message from four GPU threads. |
| [`Exercise 2/`](./Exercise%202/) | Adds two integers using a single GPU thread. |
| [`Exercise 3/`](./Exercise%203/) | Adds two 512-element vectors using 512 blocks with one thread per block. |
| [`Exercise 4/`](./Exercise%204/) | Adds two 512-element vectors using one block containing 512 threads. |
| [`Exercise 5/`](./Exercise%205/) | Multiplies two 10,000,000-element vectors using 512 threads per block. |
| [`Exercise 6/`](./Exercise%206/) | Performs element-wise multiplication of two 10,000 x 10,000 integer matrices using a 2D grid. |
| `*.cu` files | CUDA C++ source files containing the host code and GPU kernels. |
| `*_output.txt` files | Captured output from the corresponding CUDA programs. |

## Exercise summary

### Exercise 1 - Hello GPU

`hello.cu` launches one block with four threads:

```text
hello<<<1, 4>>>();
```

Each thread prints its `threadIdx.x`, producing messages for threads 0 to 3.
This introduces the CUDA kernel launch syntax and GPU thread indexing.

### Exercise 2 - Adding two numbers

`ex2_add.cu` allocates device memory for two input integers and one output
integer. The values 2 and 7 are copied to the GPU, added by a single-thread
kernel, and copied back to the CPU. The recorded result is:

```text
Result is 9
```

### Exercise 3 - Vector addition with blocks

`ex3_blocks.cu` adds 512-element vectors using **512 blocks and one thread per
block**. The kernel uses `blockIdx.x` to select the vector element:

```text
c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
```

The output contains all 512 additions and reports:

```text
Verification: 0 mismatches out of 512
```

### Exercise 4 - Vector addition with threads

`ex4_threads.cu` performs the same 512 additions using **one block with 512
threads**. This time, `threadIdx.x` selects the vector element:

```text
c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
```

The valid 512-thread configuration completed successfully with zero
mismatches. The file `ex4_2048_failure_output.txt` records an additional
experiment using 2048 threads in one block. That launch failed with
`invalid argument`, because a CUDA block cannot contain that many threads on
the tested GPU. The failure produced 2048 mismatches because the kernel did
not generate valid output for that launch.

### Exercise 5 - Large vector multiplication

`ex5_vecmul.cu` multiplies two vectors containing **10,000,000 integers**.
Each thread calculates its global index using:

```text
i = blockIdx.x * blockDim.x + threadIdx.x
```

The program uses 512 threads per block and calculates the required number of
blocks with ceiling division. The recorded launch and verification results
are:

```text
Launching 19532 blocks x 512 threads
Verification: 0 mismatches out of 10000000
```

Only the final 1,000 results are printed to keep the output manageable, while
all 10,000,000 elements are verified.

### Exercise 6 - 2D matrix element-wise multiplication

`ex6_matrix.cu` multiplies corresponding elements of two matrices:

```text
C[row][col] = A[row][col] * B[row][col]
```

This is **element-wise multiplication**, not conventional matrix
multiplication. The program processes two 10,000 x 10,000 matrices using
two-dimensional thread blocks of `32 x 16`. The recorded grid is `313 x 625`
blocks and the complete verification reports:

```text
Verification: 0 mismatches out of 100000000
```

Only the final 1,000 matrix elements are printed; all 100,000,000 elements
are checked.

## Requirements

- NVIDIA GPU with CUDA support.
- CUDA Toolkit and the `nvcc` compiler.
- A compatible NVIDIA driver.
- Sufficient GPU and host memory for the large allocations in Exercises 5 and
  6.

Google Colab can be used instead of a local CUDA installation by enabling a
GPU runtime and running the cells in the included notebook.

## Running the exercises

From the relevant exercise directory, compile a CUDA source file with:

```bash
nvcc hello.cu -o hello
```

Replace `hello.cu` and `hello` with the source file and executable name for
the selected exercise. Run the compiled program with:

```bash
./hello
```

For example, on Linux or Google Colab:

```bash
cd "Exercise 3"
nvcc ex3_blocks.cu -o ex3_blocks
./ex3_blocks
```

On Windows, run the generated executable from PowerShell:

```powershell
cd "Exercise 3"
nvcc ex3_blocks.cu -o ex3_blocks.exe
.\ex3_blocks.exe
```

The source files use standard CUDA runtime functions and do not require
additional third-party libraries.

## Verification and notes

Each vector or matrix program compares the GPU output with the expected
calculation and reports the number of mismatches. A result of
`0 mismatches` indicates that all processed elements matched the expected
values.

The input values are generated with `rand() % 100`, so the displayed random
values can differ between runs. The verification result, grid configuration,
and program behavior should remain consistent on a compatible CUDA device.