/*
  Copyright 2026 SINTEF AS

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

#define BOOST_TEST_MODULE TestDraugrAMGPreconditioner
#define BOOST_TEST_NO_MAIN

#include <boost/test/unit_test.hpp>

#include <opm/simulators/linalg/DraugrAMGPreconditioner.hpp>
#include <opm/simulators/linalg/PropertyTree.hpp>

#include <dune/common/fmatrix.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/paamg/amg.hh>
#include <dune/istl/solvers.hh>

#include <numeric>
#include <vector>

using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>>;
using Vector = Dune::BlockVector<Dune::FieldVector<double, 1>>;

namespace {

/**
 * @brief Setup a 2D Laplace operator matrix for testing.
 *
 * Creates an N x N grid discretization of the 2D Laplace operator with
 * standard 5-point stencil (diagonal = 4, neighbors = -1).
 */
void setupLaplace2d(int N, Matrix& mat)
{
    const int nonZeroes = N * N * 5;
    mat.setBuildMode(Matrix::row_wise);
    mat.setSize(N * N, N * N, nonZeroes);

    for (auto row = mat.createbegin(); row != mat.createend(); ++row) {
        const int i = row.index();
        int x = i % N;
        int y = i / N;

        row.insert(i);
        if (x > 0)
            row.insert(i - 1);
        if (x < N - 1)
            row.insert(i + 1);
        if (y > 0)
            row.insert(i - N);
        if (y < N - 1)
            row.insert(i + N);
    }

    for (auto row = mat.begin(); row != mat.end(); ++row) {
        const int i = row.index();
        int x = i % N;
        int y = i / N;

        (*row)[i] = 4.0;
        if (x > 0)
            (*row)[i - 1] = -1.0;
        if (x < N - 1)
            (*row)[i + 1] = -1.0;
        if (y > 0)
            (*row)[i - N] = -1.0;
        if (y < N - 1)
            (*row)[i + N] = -1.0;
    }
}

} // anonymous namespace

BOOST_AUTO_TEST_CASE(TestDraugrAMGPreconditioner_Solve)
{
    constexpr int N = 50; // 50x50 grid
    const int ndof = N * N;

    // Create matrix
    Matrix matrix;
    setupLaplace2d(N, matrix);

    // Create vectors
    Vector x(ndof), b(ndof);
    x = 0.0;
    b = 1.0;

    // Create operator
    using Operator = Dune::MatrixAdapter<Matrix, Vector, Vector>;
    Operator op(matrix);

    // Set up Draugr AMG parameters (string-based JSON config)
    Opm::PropertyTree prm;
    prm.put("coarsening", "hmis");
    prm.put("smoother", "l1_colored_gs");
    prm.put("interpolation", "extended_i");
    prm.put("strength", "absolute");
    prm.put("cycle", "v");
    prm.put("theta", 0.5);
    prm.put("max_levels", 15);
    prm.put("max_coarse_size", 50);
    prm.put("pre_smoothing_steps", 1);
    prm.put("post_smoothing_steps", 1);
    prm.put("verbose", 0);

    // Create preconditioner
    auto prec = std::make_shared<
        Opm::Draugr::DraugrAMGPreconditioner<Matrix, Vector, Vector, Dune::Amg::SequentialInformation>>(
        matrix, prm, Dune::Amg::SequentialInformation());

    // Create solver
    double reduction = 1e-8;
    int maxit = 300;
    int verbosity = 0;
    Dune::LoopSolver<Vector> solver(op, *prec, reduction, maxit, verbosity);

    // Solve
    Dune::InverseOperatorResult res;
    solver.apply(x, b, res);

    // Check convergence
    BOOST_CHECK(res.converged);
    BOOST_CHECK_LT(res.reduction, 1e-8);
}

BOOST_AUTO_TEST_CASE(TestDraugrAMGPreconditioner_Update)
{
    constexpr int N = 30; // 30x30 grid
    const int ndof = N * N;

    // Create matrix
    Matrix matrix;
    setupLaplace2d(N, matrix);

    // Setup preconditioner (string-based JSON config)
    Opm::PropertyTree prm;
    prm.put("coarsening", "hmis");
    prm.put("smoother", "l1_colored_gs");
    prm.put("interpolation", "extended_i");
    prm.put("theta", 0.5);
    prm.put("pre_smoothing_steps", 1);
    prm.put("post_smoothing_steps", 1);
    prm.put("verbose", 0);

    auto prec = std::make_shared<
        Opm::Draugr::DraugrAMGPreconditioner<Matrix, Vector, Vector, Dune::Amg::SequentialInformation>>(
        matrix, prm, Dune::Amg::SequentialInformation());

    // Perturb diagonal values slightly
    for (auto row = matrix.begin(); row != matrix.end(); ++row) {
        (*row)[row.index()] = 4.0 + 0.1 * (row.index() % 5);
    }

    // Update the preconditioner with new matrix values
    prec->update();

    // Solve with updated preconditioner
    using Operator = Dune::MatrixAdapter<Matrix, Vector, Vector>;
    Operator op(matrix);

    Vector x(ndof), b(ndof);
    x = 0.0;
    b = 1.0;

    Dune::LoopSolver<Vector> solver(op, *prec, 1e-8, 300, 0);

    Dune::InverseOperatorResult res;
    solver.apply(x, b, res);

    BOOST_CHECK(res.converged);
    BOOST_CHECK_LT(res.reduction, 1e-8);
}

BOOST_AUTO_TEST_CASE(TestDraugrAMGPreconditioner_DefaultConfig)
{
    constexpr int N = 30;
    const int ndof = N * N;

    Matrix matrix;
    setupLaplace2d(N, matrix);

    // Use default parameters (empty property tree)
    Opm::PropertyTree prm;

    auto prec = std::make_shared<
        Opm::Draugr::DraugrAMGPreconditioner<Matrix, Vector, Vector, Dune::Amg::SequentialInformation>>(
        matrix, prm, Dune::Amg::SequentialInformation());

    using Operator = Dune::MatrixAdapter<Matrix, Vector, Vector>;
    Operator op(matrix);

    Vector x(ndof), b(ndof);
    x = 0.0;
    b = 1.0;

    Dune::LoopSolver<Vector> solver(op, *prec, 1e-8, 300, 0);

    Dune::InverseOperatorResult res;
    solver.apply(x, b, res);

    BOOST_CHECK(res.converged);
    BOOST_CHECK_LT(res.reduction, 1e-8);
}

bool init_unit_test_func()
{
    return true;
}

int main(int argc, char** argv)
{
    int result = boost::unit_test::unit_test_main(&init_unit_test_func, argc, argv);
    return result;
}
