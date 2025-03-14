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

#define BOOST_TEST_MODULE WellMatrixMergerTest

#include <boost/test/unit_test.hpp>

#include <opm/simulators/wells/WellMatrixMerger.hpp>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>

#include <iostream>
#include <vector>
#include <cmath>

// Helper function to create a simple well matrix
template<class Matrix, class Block>
void createWellMatrix(Matrix& matrix, int numRows, int numCols, const std::vector<int>& colIndices, const Block& blockValue)
{
    matrix.setSize(numRows, numCols);
    matrix.setBuildMode(Matrix::random);
    
    // Define the sparsity pattern
    for (int row = 0; row < numRows; ++row) {
        matrix.setrowsize(row, colIndices.size());
    }
    matrix.endrowsizes();
    
    for (int row = 0; row < numRows; ++row) {
        for (const auto& col : colIndices) {
            matrix.addindex(row, col);
        }
    }
    matrix.endindices();
    
    // Set the values
    for (int row = 0; row < numRows; ++row) {
        for (const auto& col : colIndices) {
            matrix[row][col] = blockValue;
        }
    }
}

// Helper function to check if two matrices are approximately equal
template<class Matrix>
bool matricesEqual(const Matrix& a, const Matrix& b, double tolerance = 1e-10)
{
    // For the merged matrices, we only care about the non-zero entries
    // that correspond to the original matrix
    
    // Check each row of the original matrix
    for (auto rowA = a.begin(); rowA != a.end(); ++rowA) {
        int rowIdx = rowA - a.begin();
        
        // Check if the row exists in the merged matrix
        if (rowIdx >= b.N()) {
            return false;
        }
        
        // Check each column in the row
        for (auto colA = rowA->begin(); colA != rowA->end(); ++colA) {
            int colIdx = colA.index();
            
            // Check if the column exists in the merged matrix
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

BOOST_AUTO_TEST_CASE(BasicMergeTest)
{
    // Define constants for the test
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
    
    // Create the matrix merger
    Opm::WellMatrixMerger<double, numWellEq, numEq> merger;
    
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
    
    // Add wells to the merger
    merger.addWell(B1, C1, D1, cells1, 0, "Well1");
    merger.addWell(B2, C2, D2, cells2, 1, "Well2");
    
    // Finalize the merger
    merger.finalize();
    
    // Check the number of wells and perforations
    BOOST_CHECK_EQUAL(merger.getNumWells(), 2);
    BOOST_CHECK_EQUAL(merger.getNumPerforations(), 5);
    
    // Check well indices for cells
    BOOST_CHECK_EQUAL(merger.getWellIndexForCell(10), 0);
    BOOST_CHECK_EQUAL(merger.getWellIndexForCell(20), 0);
    BOOST_CHECK_EQUAL(merger.getWellIndexForCell(30), 0);
    BOOST_CHECK_EQUAL(merger.getWellIndexForCell(40), 1);
    BOOST_CHECK_EQUAL(merger.getWellIndexForCell(50), 1);
    BOOST_CHECK_EQUAL(merger.getWellIndexForCell(60), -1); // Not perforated
    
    // Check well names
    BOOST_CHECK_EQUAL(merger.getWellName(0), "Well1");
    BOOST_CHECK_EQUAL(merger.getWellName(1), "Well2");
    
    // Get the merged matrices
    const auto& mergedB = merger.getMergedB();
    const auto& mergedC = merger.getMergedC();
    const auto& mergedD = merger.getMergedD();
    
    // Check matrix dimensions
    BOOST_CHECK_EQUAL(mergedB.N(), 2); // 2 wells
    BOOST_CHECK_EQUAL(mergedC.N(), 2); // 2 wells
    BOOST_CHECK_EQUAL(mergedD.N(), 2); // 2 wells
    
    // Check that the merged matrices contain the correct values
    
    // Check B matrix for well 1
    for (auto colB = mergedB[0].begin(); colB != mergedB[0].end(); ++colB) {
        int cellIdx = colB.index();
        BOOST_CHECK(cellIdx == 10 || cellIdx == 20 || cellIdx == 30);
        
        // Check block values
        for (int i = 0; i < numWellEq; ++i) {
            for (int j = 0; j < numEq; ++j) {
                BOOST_CHECK_CLOSE((*colB)[i][j], 0.1 * (i + 1) * (j + 1), 1e-10);
            }
        }
    }
    
    // Check B matrix for well 2
    for (auto colB = mergedB[1].begin(); colB != mergedB[1].end(); ++colB) {
        int cellIdx = colB.index();
        BOOST_CHECK(cellIdx == 40 || cellIdx == 50);
        
        // Check block values
        for (int i = 0; i < numWellEq; ++i) {
            for (int j = 0; j < numEq; ++j) {
                BOOST_CHECK_CLOSE((*colB)[i][j], 0.3 * (i + 1) * (j + 1), 1e-10);
            }
        }
    }
    
    // Check C matrix for well 1
    for (auto colC = mergedC[0].begin(); colC != mergedC[0].end(); ++colC) {
        int cellIdx = colC.index();
        BOOST_CHECK(cellIdx == 10 || cellIdx == 20 || cellIdx == 30);
        
        // Check block values
        for (int i = 0; i < numWellEq; ++i) {
            for (int j = 0; j < numEq; ++j) {
                BOOST_CHECK_CLOSE((*colC)[i][j], 0.2 * (i + 1) * (j + 1), 1e-10);
            }
        }
    }
    
    // Check C matrix for well 2
    for (auto colC = mergedC[1].begin(); colC != mergedC[1].end(); ++colC) {
        int cellIdx = colC.index();
        BOOST_CHECK(cellIdx == 40 || cellIdx == 50);
        
        // Check block values
        for (int i = 0; i < numWellEq; ++i) {
            for (int j = 0; j < numEq; ++j) {
                BOOST_CHECK_CLOSE((*colC)[i][j], 0.4 * (i + 1) * (j + 1), 1e-10);
            }
        }
    }
    
    // Check D matrix
    for (int i = 0; i < numWellEq; ++i) {
        for (int j = 0; j < numWellEq; ++j) {
            BOOST_CHECK_CLOSE(mergedD[0][0][i][j], (i == j) ? 1.0 : 0.1, 1e-10);
            BOOST_CHECK_CLOSE(mergedD[1][1][i][j], (i == j) ? 2.0 : 0.2, 1e-10);
        }
    }
    
    // Test the apply method
    
    // Create test vectors
    BVector x(5); // 5 cells
    BVector y(5);
    BVector expected(5);
    
    // Initialize x with some values
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < numEq; ++j) {
            x[i][j] = 0.1 * (i + 1) * (j + 1);
        }
    }
    
    // Initialize y and expected with the same values
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < numEq; ++j) {
            y[i][j] = 1.0;
            expected[i][j] = 1.0;
        }
    }
    
    // Map cell indices to vector indices
    std::map<int, int> cellToVecIdx = {
        {10, 0}, {20, 1}, {30, 2}, {40, 3}, {50, 4}
    };
    
    // Compute expected result manually for well 1
    BVectorWell Bx1(1), invDBx1(1);
    Bx1[0] = 0.0;
    
    // Compute B1 * x
    for (auto colB = B1[0].begin(); colB != B1[0].end(); ++colB) {
        int cellIdx = cells1[colB.index()];
        int vecIdx = cellToVecIdx[cellIdx];
        
        for (int i = 0; i < numWellEq; ++i) {
            for (int j = 0; j < numEq; ++j) {
                Bx1[0][i] += (*colB)[i][j] * x[vecIdx][j];
            }
        }
    }
    
    // Compute D1^-1 * B1 * x
    invDBx1[0] = 0.0;
    DiagMatrixBlockWellType invD1;
    for (int i = 0; i < numWellEq; ++i) {
        for (int j = 0; j < numWellEq; ++j) {
            invD1[i][j] = (i == j) ? 1.0 / D1[0][0][i][j] : 0.0;
        }
    }
    
    for (int i = 0; i < numWellEq; ++i) {
        for (int j = 0; j < numWellEq; ++j) {
            invDBx1[0][i] += invD1[i][j] * Bx1[0][j];
        }
    }
    
    // Compute expected -= C1^T * invDBx1
    for (auto colC = C1[0].begin(); colC != C1[0].end(); ++colC) {
        int cellIdx = cells1[colC.index()];
        int vecIdx = cellToVecIdx[cellIdx];
        
        for (int i = 0; i < numEq; ++i) {
            for (int j = 0; j < numWellEq; ++j) {
                expected[vecIdx][i] -= (*colC)[j][i] * invDBx1[0][j];
            }
        }
    }
    
    // Compute expected result manually for well 2
    BVectorWell Bx2(1), invDBx2(1);
    Bx2[0] = 0.0;
    
    // Compute B2 * x
    for (auto colB = B2[0].begin(); colB != B2[0].end(); ++colB) {
        int cellIdx = cells2[colB.index()];
        int vecIdx = cellToVecIdx[cellIdx];
        
        for (int i = 0; i < numWellEq; ++i) {
            for (int j = 0; j < numEq; ++j) {
                Bx2[0][i] += (*colB)[i][j] * x[vecIdx][j];
            }
        }
    }
    
    // Compute D2^-1 * B2 * x
    invDBx2[0] = 0.0;
    DiagMatrixBlockWellType invD2;
    for (int i = 0; i < numWellEq; ++i) {
        for (int j = 0; j < numWellEq; ++j) {
            invD2[i][j] = (i == j) ? 1.0 / D2[0][0][i][j] : 0.0;
        }
    }
    
    for (int i = 0; i < numWellEq; ++i) {
        for (int j = 0; j < numWellEq; ++j) {
            invDBx2[0][i] += invD2[i][j] * Bx2[0][j];
        }
    }
    
    // Compute expected -= C2^T * invDBx2
    for (auto colC = C2[0].begin(); colC != C2[0].end(); ++colC) {
        int cellIdx = cells2[colC.index()];
        int vecIdx = cellToVecIdx[cellIdx];
        
        for (int i = 0; i < numEq; ++i) {
            for (int j = 0; j < numWellEq; ++j) {
                expected[vecIdx][i] -= (*colC)[j][i] * invDBx2[0][j];
            }
        }
    }
    
    // Apply the merged matrices
    merger.apply(x, y);
    
    // Check that y matches the expected result
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < numEq; ++j) {
            BOOST_CHECK_CLOSE(y[i][j], expected[i][j], 1e-10);
        }
    }
    
    // Test the clear method
    merger.clear();
    
    BOOST_CHECK_EQUAL(merger.getNumWells(), 0);
    BOOST_CHECK_EQUAL(merger.getNumPerforations(), 0);
}

BOOST_AUTO_TEST_CASE(MultisegmentWellTest)
{
    // Define constants for the test
    const int numEq = 3;       // Number of equations per grid cell
    const int numWellEq = 4;   // Number of equations per well segment
    
    // Define matrix and vector types
    using DiagMatrixBlockWellType = Dune::FieldMatrix<double, numWellEq, numWellEq>;
    using DiagMatWell = Dune::BCRSMatrix<DiagMatrixBlockWellType>;
    
    using OffDiagMatrixBlockWellType = Dune::FieldMatrix<double, numWellEq, numEq>;
    using OffDiagMatWell = Dune::BCRSMatrix<OffDiagMatrixBlockWellType>;
    
    // Create the matrix merger
    Opm::WellMatrixMerger<double, numWellEq, numEq> merger;
    
    // Create test data for a multi-segment well
    
    // Multi-segment well with 3 segments
    OffDiagMatWell B, C;
    DiagMatWell D;
    
    std::vector<int> cells = {10, 20, 30}; // Cell indices for the well
    
    // Create B matrix (3 segments x 3 perforations)
    B.setSize(3, 3);
    B.setBuildMode(Dune::BCRSMatrix<OffDiagMatrixBlockWellType>::random);
    B.setrowsize(0, 1);
    B.setrowsize(1, 1);
    B.setrowsize(2, 1);
    B.endrowsizes();
    
    B.addindex(0, 0);
    B.addindex(1, 1);
    B.addindex(2, 2);
    B.endindices();
    
    OffDiagMatrixBlockWellType BBlock;
    for (int i = 0; i < numWellEq; ++i) {
        for (int j = 0; j < numEq; ++j) {
            BBlock[i][j] = 0.1 * (i + 1) * (j + 1);
        }
    }
    
    B[0][0] = BBlock;
    B[1][1] = BBlock;
    B[2][2] = BBlock;
    
    // Create C matrix (3 segments x 3 perforations)
    C.setSize(3, 3);
    C.setBuildMode(Dune::BCRSMatrix<OffDiagMatrixBlockWellType>::random);
    C.setrowsize(0, 1); // Each segment connects to one perforation
    C.setrowsize(1, 1);
    C.setrowsize(2, 1);
    C.endrowsizes();
    
    C.addindex(0, 0);
    C.addindex(1, 1);
    C.addindex(2, 2);
    C.endindices();
    
    OffDiagMatrixBlockWellType CBlock;
    for (int i = 0; i < numWellEq; ++i) {
        for (int j = 0; j < numEq; ++j) {
            CBlock[i][j] = 0.2 * (i + 1) * (j + 1);
        }
    }
    
    C[0][0] = CBlock;
    C[1][1] = CBlock;
    C[2][2] = CBlock;
    
    // Create D matrix (3x3 block matrix for segments)
    D.setSize(3, 3);
    D.setBuildMode(Dune::BCRSMatrix<DiagMatrixBlockWellType>::random);
    D.setrowsize(0, 3); // Each segment potentially connects to all segments
    D.setrowsize(1, 3);
    D.setrowsize(2, 3);
    D.endrowsizes();
    
    D.addindex(0, 0);
    D.addindex(0, 1);
    D.addindex(0, 2);
    D.addindex(1, 0);
    D.addindex(1, 1);
    D.addindex(1, 2);
    D.addindex(2, 0);
    D.addindex(2, 1);
    D.addindex(2, 2);
    D.endindices();
    
    DiagMatrixBlockWellType DBlock;
    for (int i = 0; i < numWellEq; ++i) {
        for (int j = 0; j < numWellEq; ++j) {
            DBlock[i][j] = (i == j) ? 1.0 : 0.1;
        }
    }
    
    // Fill D matrix with connections between segments
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            D[i][j] = DBlock;
            
            // Scale off-diagonal blocks
            if (i != j) {
                for (int k = 0; k < numWellEq; ++k) {
                    for (int l = 0; l < numWellEq; ++l) {
                        D[i][j][k][l] *= 0.5;
                    }
                }
            }
        }
    }
    
    // Add the multi-segment well to the merger
    merger.addWell(B, C, D, cells, 0, "MSWell");
    
    // Finalize the merger
    merger.finalize();
    
    // Check the number of wells and perforations
    BOOST_CHECK_EQUAL(merger.getNumWells(), 1);
    BOOST_CHECK_EQUAL(merger.getNumPerforations(), 3);
    
    // Check well indices for cells
    BOOST_CHECK_EQUAL(merger.getWellIndexForCell(10), 0);
    BOOST_CHECK_EQUAL(merger.getWellIndexForCell(20), 0);
    BOOST_CHECK_EQUAL(merger.getWellIndexForCell(30), 0);
    
    // Get the merged matrices
    const auto& mergedB = merger.getMergedB();
    const auto& mergedC = merger.getMergedC();
    const auto& mergedD = merger.getMergedD();
    
    // Check that the merged matrices match the original matrices
    BOOST_CHECK(matricesEqual(B, mergedB));
    BOOST_CHECK(matricesEqual(C, mergedC));
    BOOST_CHECK(matricesEqual(D, mergedD));
} 