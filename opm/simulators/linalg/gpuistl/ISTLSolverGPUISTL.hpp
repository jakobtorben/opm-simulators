/*
  Copyright 2025 Equinor ASA

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

#ifndef OPM_ISTLSOLVERGPUISTL_HEADER_INCLUDED
#define OPM_ISTLSOLVERGPUISTL_HEADER_INCLUDED

#include "dune/istl/operators.hh"
#include "opm/simulators/linalg/gpuistl/GpuVector.hpp"
#include <opm/simulators/linalg/AbstractISTLSolver.hpp>
#include <opm/simulators/linalg/ISTLSolver.hpp>

#include <opm/simulators/linalg/FlexibleSolver.hpp>
// TODO: These should be removed so that we avoid the compilation burden.
//       That is, we should instantiate them in the cpp file instead.
#include <opm/simulators/linalg/PreconditionerFactory_impl.hpp>
#include <opm/simulators/linalg/FlexibleSolver_impl.hpp>

#include <opm/simulators/linalg/gpuistl/GpuSparseMatrix.hpp>


namespace Opm::gpuistl
{
template <class TypeTag>
class ISTLSolverGPUISTL : public AbstractISTLSolver<TypeTag>
{
public:
    using SparseMatrixAdapter = GetPropType<TypeTag, Properties::SparseMatrixAdapter>;
    using Vector = GetPropType<TypeTag, Properties::GlobalEqVector>;
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;
    using Matrix = typename SparseMatrixAdapter::IstlMatrix;
    constexpr static std::size_t pressureIndex = GetPropType<TypeTag, Properties::Indices>::pressureSwitchIdx;



    using real_type = typename Vector::field_type;
    using XGPU = GpuVector<real_type>;
    using GPUOperatorType = Dune::MatrixAdapter<GpuSparseMatrix<real_type>, XGPU, XGPU>;
    using AbstractGPUOperator = Dune::LinearOperator<XGPU, XGPU>;
    using SolverType = Dune::FlexibleSolver<GPUOperatorType>;

#if HAVE_MPI
    using CommunicationType = Dune::OwnerOverlapCopyCommunication<int, int>;
#else
    using CommunicationType = Dune::Communication<int>;
#endif

    /// Construct a system solver.
    /// \param[in] simulator   The opm-models simulator object
    /// \param[in] parameters  Explicit parameters for solver setup, do not
    ///                        read them from command line parameters.
    /// \param[in] forceSerial If true, will set up a serial linear solver only,
    ///                        local to the current rank, instead of creating a
    ///                        parallel (MPI distributed) linear solver.
    ISTLSolverGPUISTL(const Simulator& simulator,
                      [[maybe_unused]] const FlowLinearSolverParameters& parameters,
                      [[maybe_unused]] bool forceSerial = false)
        : m_parameters(parameters)
    {
        // TODO: Is there a nicer way of reading the parameters?
        // TODO: We already read them in the runtime option proxy, so we could just
        //       pass them to the constructor here. Though, then we would lose the
        //       common constructor signature.
        m_parameters.init(simulator.vanguard().eclState().getSimulationConfig().useCPR());
        m_propertyTree = setupPropertyTree(parameters,
                                           Parameters::IsSet<Parameters::LinearSolverMaxIter>(),
                                           Parameters::IsSet<Parameters::LinearSolverReduction>());
    }

    /// Construct a system solver.
    /// \param[in] simulator   The opm-models simulator object
    explicit ISTLSolverGPUISTL(const Simulator& simulator)
        : ISTLSolverGPUISTL(simulator, FlowLinearSolverParameters(), false)
    {
    }



    void eraseMatrix() override
    {
        // Nothing, this is the same as the ISTLSolver
    }

    void setActiveSolver(int num) override
    {
        if (num != 0) {
            OPM_THROW(std::logic_error, "Only one solver available for the GPU backend.");
        }
    }

    int numAvailableSolvers() const override
    {
        return 1;
    }

    void prepare(const SparseMatrixAdapter& M, Vector& b) override
    {
        prepare(M.istlMatrix(), b);
    }

    void prepare(const Matrix& M, Vector& b) override
    {
        try {
            updateMatrix(M);
            updateRhs(b);
        }
        OPM_CATCH_AND_RETHROW_AS_CRITICAL_ERROR("This is likely due to a faulty linear solver JSON specification. "
                                                "Check for errors related to missing nodes.");
    }

    void setResidual(Vector& b) override
    {
        // Should be handled in prepare() instead.
    }

    void getResidual(Vector& b) const override
    {
        if (!m_rhs) {
            OPM_THROW(std::runtime_error, "m_rhs not initialized, prepare(matrix, rhs); needs to be called");
        }
        m_rhs->copyToHost(b);
    }

    void setMatrix(const SparseMatrixAdapter& M) override
    {
        // Should be handled in prepare() instead.
    }

    bool solve(Vector& x) override
    {
        if (!m_matrix) {
            OPM_THROW(std::runtime_error, "m_matrix not initialized, prepare(matrix, rhs); needs to be called");
        }
        if (!m_rhs) {
            OPM_THROW(std::runtime_error, "m_rhs not initialized, prepare(matrix, rhs); needs to be called");
        }
        if (!m_gpuFlexibleSolver) {
            OPM_THROW(std::runtime_error,
                      "m_gpuFlexibleSolver not initialized, prepare(matrix, rhs); needs to be called");
        }

        ++m_solveCount;

        if (m_x) {
            m_x = std::make_unique<GpuVector<real_type>>(x);
        } else {
            m_x->copyFromHost(x);
        }

        // TODO: Write matrix to disk if needed
        Dune::InverseOperatorResult result;

        m_gpuFlexibleSolver->apply(*m_x, *m_rhs, result);
        m_lastSeenIterations = result.iterations;

        return result.converged;
    }

    int iterations() const override
    {
        return m_lastSeenIterations;
    }

    const CommunicationType* comm() const override
    {
        // TODO: Implement this if needed
        return nullptr;
    }

    int getSolveCount() const override
    {
        return m_solveCount;
    }

private:
    void updateMatrix(const Matrix& M)
    {
        if (!m_matrix) {
            m_matrix.reset(new auto(GpuSparseMatrix<real_type>::fromMatrix(M)));
            m_gpuOperator = std::make_unique<GPUOperatorType>(*m_matrix);
            std::function<XGPU()> weightsCalculator = {};
            m_gpuFlexibleSolver = std::make_unique<SolverType>(
                *m_gpuOperator, m_propertyTree, weightsCalculator, pressureIndex);
            
        } else {
            m_matrix->updateNonzeroValues(M);
        }

        m_gpuFlexibleSolver->preconditioner().update();
    }

    void updateRhs(const Vector& b)
    {
        if (!m_rhs) {
            m_rhs.reset(new GpuVector<real_type>(b));
        } else {
            m_rhs->copyFromHost(b);
        }
    }

    std::unique_ptr<GpuSparseMatrix<real_type>> m_matrix;

    std::unique_ptr<GPUOperatorType> m_gpuOperator;

    std::unique_ptr<SolverType> m_gpuFlexibleSolver;

    std::unique_ptr<GpuVector<real_type>> m_rhs;
    std::unique_ptr<GpuVector<real_type>> m_x;

    FlowLinearSolverParameters m_parameters;
    PropertyTree m_propertyTree;

    int m_lastSeenIterations = 0;
    int m_solveCount = 0;
};
} // namespace Opm::gpuistl

#endif
