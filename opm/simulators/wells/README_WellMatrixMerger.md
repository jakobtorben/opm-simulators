# WellMatrixMerger

The `WellMatrixMerger` class is designed to merge BCR matrices from multiple wells into consolidated matrices, preserving the well and perforation numbering. It can be used with both standard wells and multi-segment wells.

## Overview

In OPM, wells are represented using a system of matrices B, C, and D that form a Schur complement system:

```
[A C^T] [x      ] = [res      ]
[B  D ] [x_well]   [res_well]
```

Where:
- A is the reservoir matrix
- B and C are coupling matrices between wells and reservoir
- D is the well matrix
- x is the reservoir solution vector
- x_well is the well solution vector

The `WellMatrixMerger` class allows you to merge the B, C, and D matrices from multiple wells into three large matrices, which can then be used for various purposes such as:

- Applying the Schur complement to a vector: y -= C^T * (D^-1 * (B * x))
- Analyzing the well-reservoir coupling
- Implementing specialized preconditioners
- Visualizing the well-reservoir system

## Usage

### Basic Usage

```cpp
#include <opm/simulators/wells/WellMatrixMerger.hpp>

// Define constants
const int numEq = 3;       // Number of equations per grid cell
const int numWellEq = 2;   // Number of equations per well

// Create the matrix merger
Opm::WellMatrixMerger<double, numWellEq, numEq> merger;

// Add wells to the merger
merger.addWell(B1, C1, D1, cells1, 0, "Well1");
merger.addWell(B2, C2, D2, cells2, 1, "Well2");

// Finalize the merger
merger.finalize();

// Get the merged matrices
const auto& mergedB = merger.getMergedB();
const auto& mergedC = merger.getMergedC();
const auto& mergedD = merger.getMergedD();

// Apply the merged matrices to a vector
merger.apply(x, y);  // Computes y -= C^T * (D^-1 * (B * x))
```

### Template Parameters

- `Scalar`: The scalar type (typically `double`)
- `numWellEq`: Number of equations per well
- `numEq`: Number of equations per grid cell

### Key Methods

- `addWell(B, C, D, wellCells, wellIndex, wellName)`: Add a well's matrices to the merger
- `finalize()`: Finalize the matrix merging process (must be called after adding all wells)
- `getMergedB()`, `getMergedC()`, `getMergedD()`: Get the merged matrices
- `apply(x, y)`: Apply the merged matrices to a vector (y -= C^T * (D^-1 * (B * x)))
- `getWellIndexForCell(cellIndex)`: Get the well index for a given cell
- `getWellName(wellIndex)`: Get the well name for a given well index
- `clear()`: Clear all data and reset the merger

## Example

See `examples/well_matrix_merger_example.cpp` for a complete example of how to use the `WellMatrixMerger` class.

## Testing

The `WellMatrixMerger` class is tested in `tests/test_wellmatrixmerger.cpp`. The tests verify that:

1. The merged matrices correctly preserve the original matrix entries
2. The `apply` method correctly computes y -= C^T * (D^-1 * (B * x))
3. The well-cell mapping is correctly maintained

## Implementation Details

The `WellMatrixMerger` class uses a two-phase approach to merge the matrices:

1. In the first phase, it collects all the well matrices and cell information
2. In the second phase (triggered by `finalize()`), it constructs the merged matrices

This approach allows for efficient memory allocation and matrix construction.

### DUNE BCRSMatrix API

The implementation uses the DUNE BCRSMatrix API for sparse matrix operations:

1. Matrices are created with a specified build mode (`row_wise`)
2. The sparsity pattern is defined using `createbegin()` and `createend()` iterators
3. Matrix entries are added to the sparsity pattern using `row.insert(col)`
4. Matrix values are set using the `matrix[row][col] = value` syntax

For matrix-vector multiplication, the implementation uses a careful approach that respects the sparsity pattern:

1. For the B matrix multiplication (B * x), we iterate through each non-zero block and use `blockMatrix.umv(x_block, result_block)` to accumulate the result
2. For the D matrix inversion and multiplication, we use block-level operations with `blockMatrix.mv(vector_block, result_block)`
3. For the C^T multiplication, we manually compute the transpose operation by iterating through the non-zero blocks

This approach ensures that we only access matrix elements that exist in the sparsity pattern, avoiding errors with compressed array access. It follows the recommended DUNE practices for working with sparse matrices and ensures efficient and correct matrix operations.

## Limitations

- The current implementation assumes that the D matrix is invertible
- For multi-segment wells, the D matrix can be more complex, and a direct inversion might not be appropriate
- The `apply` method uses a simplified approach for inverting D; in practice, you might want to use a linear solver

## Future Improvements

- Add support for more sophisticated D matrix inversion methods
- Add support for parallel computation
- Add visualization capabilities for the merged matrices
- Add methods for extracting submatrices for specific wells or perforations 