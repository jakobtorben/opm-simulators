/*
  Copyright 2023 OPM Contributors

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <config.h>

#include <opm/simulators/wells/WellMatrixMerger.hpp>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>

#include <iostream>
#include <vector>
#include <iomanip>

// Helper function to create a simple well matrix
template<class Matrix, class Block>
void createWellMatrix(Matrix& matrix, int numRows, int numCols, const std::vector<int>& colIndices, const Block& blockValue)
{
    matrix.setSize(numRows, numCols);
    matrix.setBuildMode(Matrix::row_wise);
    
    // Define the sparsity pattern
    for (auto row = matrix.createbegin(); row != matrix.createend(); ++row) {
        for (const auto& col : colIndices) {
            row.insert(col);
        }
    }
    
    // Set the values
    for (int row = 0; row < numRows; ++row) {
        for (const auto& col : colIndices) {
            matrix[row][col] = blockValue;
        }
    }
}

// Helper function to print a matrix
template<class Matrix>
void printMatrix(const std::string& name, const Matrix& matrix)
{
    std::cout << "Matrix " << name << " (" << matrix.N() << " x " << matrix.M() << "):" << std::endl;
    
    for (auto row = matrix.begin(); row != matrix.end(); ++row) {
        std::cout << "  Row " << row.index() << ": ";
        for (auto col = row->begin(); col != row->end(); ++col) {
            std::cout << "Col " << col.index() << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Helper function to check if two matrices are equal
template<class Matrix>
bool matricesEqual(const Matrix& a, const Matrix& b, double tolerance = 1e-10)
{
    // Check dimensions
    if (a.N() != b.N() || a.M() != b.M()) {
        return false;
    }
    
    // Check each row
    for (size_t rowIdx = 0; rowIdx < a.N(); ++rowIdx) {
        auto rowA = a[rowIdx];
        
        // Check number of non-zeros in the row
        if (rowA.size() != b[rowIdx].size()) {
            return false;
        }
        
        // Check each column in the row
        for (auto colA = rowA.begin(); colA != rowA.end(); ++colA) {
            int colIdx = colA.index();
            
            // Check if the column exists in the second matrix
            if (colIdx >= b.M() || b[rowIdx].find(colIdx) == b[rowIdx].end()) {
                return false;
            }
            
            // Check if the blocks are approximately equal
            const auto& blockA = *colA;
            const auto& blockB = b[rowIdx][colIdx];
            
            for (size_t i = 0; i < blockA.N(); ++i) {
                for (size_t j = 0; j < blockA.M(); ++j) {
                    if (std::abs(blockA[i][j] - blockB[i][j]) > tolerance) {
                        return false;
                    }
                }
            }
        }
    }
    
    return true;
}

// Helper function to merge matrices using Dune functionality - simplified version
template<class Matrix>
void mergeMatrices(const std::vector<Matrix>& matrices, Matrix& mergedMatrix)
{
    // Count total rows and find maximum column index
    int totalRows = 0;
    int maxColIndex = 0;
    
    for (const auto& matrix : matrices) {
        totalRows += matrix.N();
        maxColIndex = std::max(maxColIndex, static_cast<int>(matrix.M()) - 1);
    }
    
    // Set up the merged matrix
    mergedMatrix.setSize(totalRows, maxColIndex + 1);
    mergedMatrix.setBuildMode(Matrix::row_wise);
    
    // Define sparsity pattern and copy values in one pass
    int rowOffset = 0;
    for (const auto& matrix : matrices) {
        for (size_t i = 0; i < matrix.N(); ++i) {
            // Create row in merged matrix
            auto mergedRow = mergedMatrix.createbegin();
            std::advance(mergedRow, rowOffset + i);
            
            // Add entries from original matrix
            auto row = matrix[i];
            for (auto it = row.begin(); it != row.end(); ++it) {
                mergedRow.insert(it.index());
            }
        }
        
        // Copy values
        for (size_t i = 0; i < matrix.N(); ++i) {
            auto row = matrix[i];
            for (auto it = row.begin(); it != row.end(); ++it) {
                mergedMatrix[rowOffset + i][it.index()] = *it;
            }
        }
        
        rowOffset += matrix.N();
    }
}

int main()
{
    // Define constants for the example
    const int numEq = 3;       // Number of equations per grid cell
    const int numWellEq = 2;   // Number of equations per well
    
    // Define matrix and vector types
    using DiagMatrixBlockWellType = Dune::FieldMatrix<double, numWellEq, numWellEq>;
    using DiagMatWell = Dune::BCRSMatrix<DiagMatrixBlockWellType>;
    
    using OffDiagMatrixBlockWellType = Dune::FieldMatrix<double, numWellEq, numEq>;
    using OffDiagMatWell = Dune::BCRSMatrix<OffDiagMatrixBlockWellType>;
    
    using VectorBlockWellType = Dune::FieldVector<double, numWellEq>;
    using BVectorWell = Dune::BlockVector<VectorBlockWellType>;
    
    using VectorBlockType = Dune::FieldVector<double, numEq>;
    using BVector = Dune::BlockVector<VectorBlockType>;
    
    std::cout << "=== Well Matrix Merger Example ===" << std::endl << std::endl;
    
    // Create the matrix merger
    Opm::WellMatrixMerger<double, numWellEq, numEq> merger;
    
    std::cout << "Created WellMatrixMerger instance" << std::endl << std::endl;
    
    // Create test data for two wells
    
    // Well 1
    OffDiagMatWell B1, C1;
    DiagMatWell D1;
    
    std::vector<int> cells1 = {10, 20, 30}; // Cell indices for well 1
    
    // Create B1 matrix
    OffDiagMatrixBlockWellType B1Block;
    for (int i = 0; i < numWellEq; ++i) {
        for (int j = 0; j < numEq; ++j) {
            B1Block[i][j] = 0.1 * (i + 1) * (j + 1);
        }
    }
    createWellMatrix(B1, 1, cells1.size(), {0, 1, 2}, B1Block);
    
    // Create C1 matrix
    OffDiagMatrixBlockWellType C1Block;
    for (int i = 0; i < numWellEq; ++i) {
        for (int j = 0; j < numEq; ++j) {
            C1Block[i][j] = 0.2 * (i + 1) * (j + 1);
        }
    }
    createWellMatrix(C1, 1, cells1.size(), {0, 1, 2}, C1Block);
    
    // Create D1 matrix
    DiagMatrixBlockWellType D1Block;
    for (int i = 0; i < numWellEq; ++i) {
        for (int j = 0; j < numWellEq; ++j) {
            D1Block[i][j] = (i == j) ? 1.0 : 0.1;
        }
    }
    createWellMatrix(D1, 1, 1, {0}, D1Block);
    
    // Well 2
    OffDiagMatWell B2, C2;
    DiagMatWell D2;
    
    std::vector<int> cells2 = {40, 50}; // Cell indices for well 2
    
    // Create B2 matrix
    OffDiagMatrixBlockWellType B2Block;
    for (int i = 0; i < numWellEq; ++i) {
        for (int j = 0; j < numEq; ++j) {
            B2Block[i][j] = 0.3 * (i + 1) * (j + 1);
        }
    }
    createWellMatrix(B2, 1, cells2.size(), {0, 1}, B2Block);
    
    // Create C2 matrix
    OffDiagMatrixBlockWellType C2Block;
    for (int i = 0; i < numWellEq; ++i) {
        for (int j = 0; j < numEq; ++j) {
            C2Block[i][j] = 0.4 * (i + 1) * (j + 1);
        }
    }
    createWellMatrix(C2, 1, cells2.size(), {0, 1}, C2Block);
    
    // Create D2 matrix
    DiagMatrixBlockWellType D2Block;
    for (int i = 0; i < numWellEq; ++i) {
        for (int j = 0; j < numWellEq; ++j) {
            D2Block[i][j] = (i == j) ? 2.0 : 0.2;
        }
    }
    createWellMatrix(D2, 1, 1, {0}, D2Block);
    
    std::cout << "Created matrices for two wells:" << std::endl;
    printMatrix("B1", B1);
    printMatrix("C1", C1);
    printMatrix("D1", D1);
    printMatrix("B2", B2);
    printMatrix("C2", C2);
    printMatrix("D2", D2);
    
    // Add wells to the merger
    std::cout << "Adding wells to the merger..." << std::endl;
    merger.addWell(B1, C1, D1, cells1, 0, "Well1");
    merger.addWell(B2, C2, D2, cells2, 1, "Well2");
    
    // Finalize the merger
    std::cout << "Finalizing the merger..." << std::endl;
    merger.finalize();
    
    // Get the merged matrices
    const auto& mergedB = merger.getMergedB();
    const auto& mergedC = merger.getMergedC();
    const auto& mergedD = merger.getMergedD();
    
    std::cout << "Merged matrices:" << std::endl;
    printMatrix("mergedB", mergedB);
    printMatrix("mergedC", mergedC);
    printMatrix("mergedD", mergedD);
    
    // Demonstrate direct matrix merging using Dune functionality
    std::cout << "\nDemonstrating direct matrix merging using Dune functionality..." << std::endl;
    
    // Create vectors of matrices
    std::vector<OffDiagMatWell> bMatrices = {B1, B2};
    std::vector<OffDiagMatWell> cMatrices = {C1, C2};
    std::vector<DiagMatWell> dMatrices = {D1, D2};
    
    // Create merged matrices
    OffDiagMatWell directMergedB, directMergedC;
    DiagMatWell directMergedD;
    
    // Get the number of wells
    const size_t totalWells = bMatrices.size();
    
    // Merge B matrices
    mergeMatrices(bMatrices, directMergedB);
    
    // Merge C matrices
    mergeMatrices(cMatrices, directMergedC);
    
    // For D matrix, we need to create a special merged matrix with diagonal blocks
    directMergedD.setSize(totalWells, totalWells);
    directMergedD.setBuildMode(DiagMatWell::row_wise);
    
    // Define sparsity pattern for D matrix (diagonal blocks only)
    for (size_t i = 0; i < totalWells; ++i) {
        auto row = directMergedD.createbegin();
        std::advance(row, i);
        row.insert(i);  // Only diagonal entries
    }
    
    // Set values for D matrix
    for (size_t i = 0; i < dMatrices.size(); ++i) {
        if (dMatrices[i].N() > 0 && dMatrices[i][0].size() > 0) {
            directMergedD[i][i] = dMatrices[i][0][0];
        }
    }
    
    std::cout << "Directly merged matrices:" << std::endl;
    printMatrix("directMergedB", directMergedB);
    printMatrix("directMergedC", directMergedC);
    printMatrix("directMergedD", directMergedD);
    
    // Verify that the matrices are the same
    bool bEqual = matricesEqual(mergedB, directMergedB);
    bool cEqual = matricesEqual(mergedC, directMergedC);
    bool dEqual = matricesEqual(mergedD, directMergedD);
    
    std::cout << "\nVerifying that the matrices are the same:" << std::endl;
    std::cout << "B matrices equal: " << (bEqual ? "Yes" : "No") << std::endl;
    std::cout << "C matrices equal: " << (cEqual ? "Yes" : "No") << std::endl;
    std::cout << "D matrices equal: " << (dEqual ? "Yes" : "No") << std::endl;
    
    // Create test vectors
    BVector x(5); // 5 cells
    BVector y(5);
    
    // Initialize x with some values
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < numEq; ++j) {
            x[i][j] = 0.1 * (i + 1) * (j + 1);
        }
    }
    
    // Initialize y with ones
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < numEq; ++j) {
            y[i][j] = 1.0;
        }
    }
    
    std::cout << "Initial vectors:" << std::endl;
    std::cout << "x = [";
    for (int i = 0; i < 5; ++i) {
        std::cout << "(";
        for (int j = 0; j < numEq; ++j) {
            std::cout << std::fixed << std::setprecision(2) << x[i][j];
            if (j < numEq - 1) std::cout << ", ";
        }
        std::cout << ")";
        if (i < 4) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "y = [";
    for (int i = 0; i < 5; ++i) {
        std::cout << "(";
        for (int j = 0; j < numEq; ++j) {
            std::cout << std::fixed << std::setprecision(2) << y[i][j];
            if (j < numEq - 1) std::cout << ", ";
        }
        std::cout << ")";
        if (i < 4) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // Apply the merged matrices
    std::cout << "Applying the merged matrices..." << std::endl;
    merger.apply(x, y);
    
    std::cout << "Result after applying merged matrices:" << std::endl;
    std::cout << "y = [";
    for (int i = 0; i < 5; ++i) {
        std::cout << "(";
        for (int j = 0; j < numEq; ++j) {
            std::cout << std::fixed << std::setprecision(4) << y[i][j];
            if (j < numEq - 1) std::cout << ", ";
        }
        std::cout << ")";
        if (i < 4) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // Map cell indices to vector indices for verification
    std::map<int, int> cellToVecIdx = {
        {10, 0}, {20, 1}, {30, 2}, {40, 3}, {50, 4}
    };
    
    // Check well indices for cells
    std::cout << std::endl << "Well indices for cells:" << std::endl;
    for (const auto& cell : cells1) {
        std::cout << "Cell " << cell << " belongs to well " << merger.getWellIndexForCell(cell) 
                  << " (" << merger.getWellName(merger.getWellIndexForCell(cell)) << ")" << std::endl;
    }
    for (const auto& cell : cells2) {
        std::cout << "Cell " << cell << " belongs to well " << merger.getWellIndexForCell(cell)
                  << " (" << merger.getWellName(merger.getWellIndexForCell(cell)) << ")" << std::endl;
    }
    std::cout << "Cell 60 belongs to well " << merger.getWellIndexForCell(60) << " (not perforated)" << std::endl;
    
    std::cout << std::endl << "Example completed successfully!" << std::endl;
    
    return 0;
} } 
