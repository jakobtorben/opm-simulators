#include <config.h>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>

#include <opm/simulators/linalg/WellMatrixMerger.hpp>
#include <opm/simulators/linalg/TailoredBlockPreconditioner.hpp>
#include <opm/simulators/linalg/PropertyTree.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <memory>

// Define block sizes
constexpr int numResDofs = 3;  // Number of equations per reservoir block
constexpr int numWellDofs = 4; // Number of equations per well block

// Define matrix and vector types
using MatrixBlockType = Dune::FieldMatrix<double, numResDofs, numResDofs>;
using WellBlockType = Dune::FieldMatrix<double, numWellDofs, numWellDofs>;
using RRMatrix = Dune::BCRSMatrix<MatrixBlockType>;
using WRMatrix = Dune::BCRSMatrix<Dune::FieldMatrix<double, numWellDofs, numResDofs>>;
using RWMatrix = Dune::BCRSMatrix<Dune::FieldMatrix<double, numResDofs, numWellDofs>>;
using WWMatrix = Dune::BCRSMatrix<WellBlockType>;

using RVector = Dune::BlockVector<Dune::FieldVector<double, numResDofs>>;
using WVector = Dune::BlockVector<Dune::FieldVector<double, numWellDofs>>;

using WellMerger = Opm::WellMatrixMerger<MatrixBlockType, WellBlockType>;
using SystemMatrix = typename WellMerger::SystemMatrix;
using SystemVector = typename WellMerger::SystemVector;

// Define indices for accessing blocks
constexpr auto _0 = Dune::Indices::_0;
constexpr auto _1 = Dune::Indices::_1;

// Helper functions
template<class Matrix, class Block>
void createMatrix(Matrix& matrix, int numRows, int numCols, const std::vector<int>& colIndices, const Block& blockValue)
{
    matrix.setSize(numRows, numCols);
    matrix.setBuildMode(Matrix::random);
    
    // Set up sparsity pattern
    for (int row = 0; row < numRows; ++row) {
        for (const auto& col : colIndices) {
            if (col < numCols) {
                matrix.entry(row, col);
            }
        }
    }
    
    // Set values
    matrix = 0.0;
    for (int row = 0; row < numRows; ++row) {
        for (const auto& col : colIndices) {
            if (col < numCols) {
                matrix[row][col] = blockValue;
            }
        }
    }
}

template<class Matrix>
void printMatrix(const std::string& name, const Matrix& matrix)
{
    std::cout << name << " (" << matrix.N() << " x " << matrix.M() << "):" << std::endl;
    for (int row = 0; row < std::min(5, (int)matrix.N()); ++row) {
        std::cout << "  Row " << row << ": ";
        for (auto it = matrix[row].begin(); it != matrix[row].end(); ++it) {
            std::cout << "(" << row << "," << it.index() << ") ";
        }
        std::cout << std::endl;
    }
    if (matrix.N() > 5) {
        std::cout << "  ... (" << matrix.N() - 5 << " more rows)" << std::endl;
    }
    std::cout << std::endl;
}

template<class Vector>
void printVector(const std::string& name, const Vector& vector)
{
    std::cout << name << " (" << vector.size() << "):" << std::endl;
    for (int i = 0; i < std::min(5, (int)vector.size()); ++i) {
        std::cout << "  " << i << ": " << vector[i] << std::endl;
    }
    if (vector.size() > 5) {
        std::cout << "  ... (" << vector.size() - 5 << " more entries)" << std::endl;
    }
    std::cout << std::endl;
}

template<int ROWS, int COLS>
Dune::FieldMatrix<double, ROWS, COLS> makeBlock(const std::function<double(int, int)>& valueFn) {
    Dune::FieldMatrix<double, ROWS, COLS> block;
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            block[i][j] = valueFn(i, j);
        }
    }
    return block;
}

template<int SIZE>
Dune::FieldMatrix<double, SIZE, SIZE> makeDiagonalBlock(double diagValue, double offDiagValue = 0.0) {
    return makeBlock<SIZE, SIZE>([=](int i, int j) {
        return (i == j) ? diagValue : offDiagValue;
    });
}

template<int ROWS, int COLS>
Dune::FieldMatrix<double, ROWS, COLS> makeScaledBlock(double scale) {
    return makeBlock<ROWS, COLS>([=](int i, int j) {
        return scale * (i + 1) * (j + 1);
    });
}

template<class Matrix, class Block>
void createDenseMatrix(Matrix& matrix, int numRows, int numCols, const Block& blockValue)
{
    std::vector<int> allColumns(numCols);
    for (int i = 0; i < numCols; ++i) {
        allColumns[i] = i;
    }
    createMatrix(matrix, numRows, numCols, allColumns, blockValue);
}

// Example of how to use the WellMatrixMerger in OPM
void demonstrateOpmIntegration()
{
    std::cout << "\n=== How to Use WellMatrixMerger in OPM ===\n" << std::endl;
    
    std::cout << "1. In your OPM code, you need to modify the ISTLSolver class to use the WellMatrixMerger approach:" << std::endl;
    std::cout << "   - Add a new option to FlowLinearSolverParameters for well handling method" << std::endl;
    std::cout << "   - Modify the solve method to use WellMatrixMergerIntegration when this option is selected" << std::endl;
    
    std::cout << "\n2. Example code for ISTLSolver::solve():" << std::endl;
    std::cout << "```cpp" << std::endl;
    std::cout << "bool ISTLSolver<TypeTag>::solve(Vector& x)" << std::endl;
    std::cout << "{" << std::endl;
    std::cout << "    // Check if we should use the well matrix merger approach" << std::endl;
    std::cout << "    if (parameters_.well_handling_method_ == WellHandlingMethod::MATRIX_MERGER) {" << std::endl;
    std::cout << "        // Create the well matrix merger integration" << std::endl;
    std::cout << "        WellMatrixMergerIntegration<TypeTag> wellMergerIntegration(simulator_);" << std::endl;
    std::cout << "        wellMergerIntegration.initialize(*wellOperator_);" << std::endl;
    std::cout << "        " << std::endl;
    std::cout << "        // Solve the system" << std::endl;
    std::cout << "        bool converged = wellMergerIntegration.solve(getMatrix(), x, *rhs_);" << std::endl;
    std::cout << "        " << std::endl;
    std::cout << "        // Set convergence status" << std::endl;
    std::cout << "        converged_ = converged;" << std::endl;
    std::cout << "        " << std::endl;
    std::cout << "        return converged_;" << std::endl;
    std::cout << "    }" << std::endl;
    std::cout << "    " << std::endl;
    std::cout << "    // Use the traditional approach" << std::endl;
    std::cout << "    // ..." << std::endl;
    std::cout << "}" << std::endl;
    std::cout << "```" << std::endl;
    
    std::cout << "\n3. To use this in a simulation, run flow with the new option:" << std::endl;
    std::cout << "   $ flow --well-handling-method=matrix-merger CASE.DATA" << std::endl;
    
    std::cout << "\n4. Benefits of this approach:" << std::endl;
    std::cout << "   - Better handling of well equations in the linear solver" << std::endl;
    std::cout << "   - Improved convergence for cases with many wells" << std::endl;
    std::cout << "   - Ability to use specialized preconditioners for reservoir and well blocks" << std::endl;
    std::cout << "   - More efficient solution of the coupled system" << std::endl;
}

int main()
{
    std::cout << "=== Matrix Merger Solver Example ===" << std::endl << std::endl;
    
    // Create matrices for a simple reservoir system
    RRMatrix A;
    
    // Create matrices for two wells
    WRMatrix B1, B2;
    RWMatrix C1, C2;
    WWMatrix D1, D2;
    
    // Cell indices for the wells
    std::vector<int> cells1 = {0, 1, 2}; // Cells for well 1
    std::vector<int> cells2 = {3, 4};    // Cells for well 2

    
    // Create reservoir matrix A (diagonal blocks)
    auto A_block = makeDiagonalBlock<numResDofs>(2.0);
    createDenseMatrix(A, 5, 5, A_block);

    // Create B1 matrix (well 1 to reservoir)
    auto B1_block = makeScaledBlock<numWellDofs, numResDofs>(0.1);
    createMatrix(B1, 1, 5, cells1, B1_block);
    
    // Create C1 matrix (reservoir to well 1)
    auto C1_block = makeScaledBlock<numResDofs, numWellDofs>(0.2);
    createMatrix(C1, 5, 1, {0}, C1_block);
    
    // Create D1 matrix (well 1 to well 1)
    auto D1_block = makeDiagonalBlock<numWellDofs>(1.5, 0.1);
    createMatrix(D1, 1, 1, {0}, D1_block);
    
    // Create B2 matrix (well 2 to reservoir)
    auto B2_block = makeScaledBlock<numWellDofs, numResDofs>(0.3);
    createMatrix(B2, 1, 5, cells2, B2_block);
    
    // Create C2 matrix (reservoir to well 2)
    auto C2_block = makeScaledBlock<numResDofs, numWellDofs>(0.4);
    createMatrix(C2, 5, 1, {0}, C2_block);
    
    // Create D2 matrix (well 2 to well 2)
    auto D2_block = makeDiagonalBlock<numWellDofs>(1.8, 0.2);
    createMatrix(D2, 1, 1, {0}, D2_block);
    
    // Print the matrices
    printMatrix("A", A);
    printMatrix("B1", B1);
    printMatrix("C1", C1);
    printMatrix("D1", D1);
    printMatrix("B2", B2);
    printMatrix("C2", C2);
    printMatrix("D2", D2);
    
    // Create a well matrix merger
    WellMerger merger;
    
    // Add the wells to the merger
    merger.addWell(B1, C1, D1, cells1, 0, "WELL1");
    merger.addWell(B2, C2, D2, cells2, 1, "WELL2");
    
    // Finalize the merger to create the merged matrices
    merger.finalize();
    
    // Create the system matrix
    auto S = merger.createSystemMatrix(A);
    
    // Print the system matrix
    std::cout << "\nSystem Matrix S:" << std::endl;
    std::cout << "S[0][0] (A):" << std::endl;
    printMatrix("S[0][0]", S[_0][_0]);
    std::cout << "S[0][1] (C):" << std::endl;
    printMatrix("S[0][1]", S[_0][_1]);
    std::cout << "S[1][0] (B):" << std::endl;
    printMatrix("S[1][0]", S[_1][_0]);
    std::cout << "S[1][1] (D):" << std::endl;
    printMatrix("S[1][1]", S[_1][_1]);
    
    // Create vectors for the system
    SystemVector x, b;
    
    // Initialize the vectors
    x[_0].resize(5);
    x[_1].resize(2);
    b[_0].resize(5);
    b[_1].resize(2);
    
    // Set initial values
    for (int i = 0; i < 5; ++i) {
        x[_0][i] = 0.0;
        b[_0][i] = 1.0;
    }
    
    for (int i = 0; i < 2; ++i) {
        x[_1][i] = 0.0;
        b[_1][i] = 1.0;
    }
    
    // Create a linear operator for the system matrix
    const Dune::MatrixAdapter<SystemMatrix, SystemVector, SystemVector> S_linop(S);
    
    // Create a preconditioner for the system using TailoredBlockPreconditioner
    Opm::PropertyTree params;
    params.put("solver", "bicgstab");
    params.put("tol", 1e-6);
    params.put("maxiter", 100);
    params.put("verbosity", 1);
    
    // Set reservoir preconditioner parameters
    params.put("reservoir_preconditioner.type", "ilu0");
    params.put("reservoir_preconditioner.relaxation", 0.9);
    params.put("reservoir_preconditioner.fill_level", 0);
    
    // Set well preconditioner parameters
    params.put("well_preconditioner.type", "diagonal");
    
    // Create the preconditioner
    Opm::TailoredBlockPreconditioner<MatrixBlockType, WellBlockType> precond(S, params);
    
    // Create a solver for the system
    Dune::BiCGSTABSolver<SystemVector> solver(S_linop, precond, 1e-6, 100, 1);
    
    // Solve the system
    Dune::InverseOperatorResult result;
    solver.apply(x, b, result);
    
    // Print the solution
    std::cout << "\nSolution:" << std::endl;
    printVector("x[0]", x[_0]);
    printVector("x[1]", x[_1]);
    
    // Print solver results
    std::cout << "\nSolver Results:" << std::endl;
    std::cout << "  Converged: " << (result.converged ? "Yes" : "No") << std::endl;
    std::cout << "  Iterations: " << result.iterations << std::endl;
    std::cout << "  Reduction: " << result.reduction << std::endl;
    
    // Demonstrate how to use this in OPM
    demonstrateOpmIntegration();
    
    return 0;
}
