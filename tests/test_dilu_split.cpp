/*
  Copyright 2024 SINTEF AS
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
#define BOOST_TEST_MODULE TestDILUSplit

#include <config.h>
#include <opm/simulators/linalg/DILU.hpp>
#include <opm/simulators/linalg/MultithreadDILU.hpp>

#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>
#include <dune/common/fmatrix.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/preconditioners.hh>

using NumericTypes = boost::mpl::list<double, float>;

BOOST_AUTO_TEST_CASE_TEMPLATE(CompareDILUAndMultithreadDILU, T, NumericTypes)
{
    /*
        Test that both DILU implementations produce the same results
        for a simple 3x3 matrix with block size 2x2.

                 A
        | | 3  1| | 2  1| | 0  0| |
        | | 2  1| | 1  3| | 0  0| |
        |                         |
        | | 1  0| | 3  1| | 1  0| |
        | | 0  1| | 1  3| | 0  1| |
        |                         |
        | | 0  0| | 2  0| | 3  1| |
        | | 0  0| | 0  2| | 1  4| |
    */

    const int N = 3;
    constexpr int bz = 2;
    const int nonZeroes = 6;
    using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<T, bz, bz>>;
    using Vector = Dune::BlockVector<Dune::FieldVector<T, bz>>;

    Matrix A(N, N, nonZeroes, Matrix::row_wise);
    for (auto row = A.createbegin(); row != A.createend(); ++row) {
        if (row.index() == 0) {
            row.insert(0);
            row.insert(1);
        } else if (row.index() == 1) {
            row.insert(0);
            row.insert(1);
            row.insert(2);
        } else if (row.index() == 2) {
            row.insert(1);
            row.insert(2);
        }
    }

    // First row
    A[0][0][0][0] = 3.0;
    A[0][0][0][1] = 1.0;
    A[0][0][1][0] = 2.0;
    A[0][0][1][1] = 1.0;

    A[0][1][0][0] = 2.0;
    A[0][1][0][1] = 1.0;
    A[0][1][1][0] = 1.0;
    A[0][1][1][1] = 3.0;

    // Second row
    A[1][0][0][0] = 1.0;
    A[1][0][1][1] = 1.0;

    A[1][1][0][0] = 3.0;
    A[1][1][0][1] = 1.0;
    A[1][1][1][0] = 1.0;
    A[1][1][1][1] = 3.0;

    A[1][2][0][0] = 1.0;
    A[1][2][1][1] = 1.0;

    // Third row
    A[2][1][0][0] = 2.0;
    A[2][1][1][1] = 2.0;

    A[2][2][0][0] = 3.0;
    A[2][2][0][1] = 1.0;
    A[2][2][1][0] = 1.0;
    A[2][2][1][1] = 4.0;

    // Create input and output vectors
    Vector x1(N), x2(N), b(N);
    
    // Initialize b with some values
    b[0][0] = 2.0;
    b[0][1] = 3.0;
    b[1][0] = 1.0;
    b[1][1] = 4.0;
    b[2][0] = 3.0;
    b[2][1] = 2.0;

    // Create both preconditioners
    Dune::DILU<Matrix, Vector, Vector> serial_dilu(A);
    Dune::MultithreadDILU<Matrix, Vector, Vector> mt_dilu(A);

    // Test 1: Compare diagonal matrices
    auto serial_Dinv = serial_dilu.getDiagonal();
    auto mt_Dinv = mt_dilu.getDiagonal();
    
    BOOST_CHECK_EQUAL(serial_Dinv.size(), mt_Dinv.size());
    
    for (std::size_t i = 0; i < serial_Dinv.size(); ++i) {
        for (int r = 0; r < bz; ++r) {
            for (int c = 0; c < bz; ++c) {
                BOOST_CHECK_CLOSE(serial_Dinv[i][r][c], mt_Dinv[i][r][c], 1e-5);
            }
        }
    }

    // Test 2: Compare apply results
    serial_dilu.apply(x1, b);
    mt_dilu.apply(x2, b);
    
    for (std::size_t i = 0; i < N; ++i) {
        for (int r = 0; r < bz; ++r) {
            BOOST_CHECK_CLOSE(x1[i][r], x2[i][r], 1e-5);
        }
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(CompareDILUAndMultithreadDILULarger, T, NumericTypes)
{
    /*
        Test with a larger sparse matrix (10x10)
    */
    const int N = 10;
    constexpr int bz = 3;
    const int nonZeroes = 50; // Upper bound on number of non-zeros
    using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<T, bz, bz>>;
    using Vector = Dune::BlockVector<Dune::FieldVector<T, bz>>;

    // Create a matrix with banded structure (tridiagonal plus some other connections)
    Matrix A(N, N, nonZeroes, Matrix::row_wise);
    for (auto row = A.createbegin(); row != A.createend(); ++row) {
        // Always include diagonal
        row.insert(row.index());
        
        // Lower diagonal
        if (row.index() > 0) {
            row.insert(row.index() - 1);
        }
        
        // Upper diagonal
        if (row.index() < N - 1) {
            row.insert(row.index() + 1);
        }
        
        // Some random longer connections
        if (row.index() % 3 == 0 && row.index() + 3 < N) {
            row.insert(row.index() + 3);
        }
        if (row.index() % 4 == 0 && row.index() - 2 >= 0) {
            row.insert(row.index() - 2);
        }
    }

    // Fill the matrix with values
    for (int i = 0; i < N; ++i) {
        // Diagonal blocks are identity matrices scaled by some factor
        for (int r = 0; r < bz; ++r) {
            for (int c = 0; c < bz; ++c) {
                if (r == c) {
                    A[i][i][r][c] = 5.0 + i*0.1; // Diagonal element
                } else {
                    A[i][i][r][c] = 0.1; // Off-diagonal element
                }
            }
        }
        
        // Off-diagonal blocks
        for (auto col = A[i].begin(); col != A[i].end(); ++col) {
            int j = col.index();
            if (i != j) {
                for (int r = 0; r < bz; ++r) {
                    for (int c = 0; c < bz; ++c) {
                        A[i][j][r][c] = 0.1 * (i+1) * (j+1) / (10.0 + std::abs(i-j)); // Some arbitrary value
                    }
                }
            }
        }
    }

    // Create input and output vectors
    Vector x1(N), x2(N), b(N);
    
    // Initialize b with some values
    for (int i = 0; i < N; ++i) {
        for (int r = 0; r < bz; ++r) {
            b[i][r] = 1.0 + i*0.5 + r*0.25;
        }
    }

    // Create both preconditioners
    Dune::DILU<Matrix, Vector, Vector> serial_dilu(A);
    Dune::MultithreadDILU<Matrix, Vector, Vector> mt_dilu(A);

    // Test 1: Compare diagonal matrices
    auto serial_Dinv = serial_dilu.getDiagonal();
    auto mt_Dinv = mt_dilu.getDiagonal();
    
    BOOST_CHECK_EQUAL(serial_Dinv.size(), mt_Dinv.size());
    
    for (std::size_t i = 0; i < serial_Dinv.size(); ++i) {
        for (int r = 0; r < bz; ++r) {
            for (int c = 0; c < bz; ++c) {
                BOOST_CHECK_CLOSE(serial_Dinv[i][r][c], mt_Dinv[i][r][c], 1e-5);
            }
        }
    }

    // Test 2: Compare apply results
    serial_dilu.apply(x1, b);
    mt_dilu.apply(x2, b);
    
    for (std::size_t i = 0; i < N; ++i) {
        for (int r = 0; r < bz; ++r) {
            BOOST_CHECK_CLOSE(x1[i][r], x2[i][r], 1e-5);
        }
    }
} 