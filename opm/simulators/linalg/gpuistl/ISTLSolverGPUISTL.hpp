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

#include <opm/simulators/linalg/AbstractISTLSolver.hpp>
#include <opm/simulators/linalg/ISTLSolver.hpp>

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
                      const FlowLinearSolverParameters& parameters,
                      bool forceSerial = false)
        : cpuSolver_(simulator, parameters, forceSerial)
    {
    }

    /// Construct a system solver.
    /// \param[in] simulator   The opm-models simulator object
    explicit ISTLSolverGPUISTL(const Simulator& simulator)
        : cpuSolver_(simulator)
    {
    }


    void eraseMatrix() override
    {
        cpuSolver_->eraseMatrix();
    }

    void setActiveSolver(int num) override
    {
        cpuSolver_->setActiveSolver(num);
    }

    int numAvailableSolvers() const override
    {
        return cpuSolver_->numAvailableSolvers();
    }

    void prepare(const SparseMatrixAdapter& M, Vector& b) override
    {
        cpuSolver_->prepare(M, b);
    }

    void prepare(const Matrix& M, Vector& b) override
    {
        cpuSolver_->prepare(M, b);
    }

    void setResidual(Vector& b) override
    {
        cpuSolver_->setResidual(b);
    }

    void getResidual(Vector& b) const override
    {
        cpuSolver_->getResidual(b);
    }

    void setMatrix(const SparseMatrixAdapter& M) override
    {
        cpuSolver_->setMatrix(M);
    }

    bool solve(Vector& x) override
    {
        return cpuSolver_->solve(x);
    }

    int iterations() const override
    {
        return cpuSolver_->iterations();
    }

    const CommunicationType* comm() const override
    {
        return cpuSolver_->comm();
    }

    int getSolveCount() const override
    {
        return cpuSolver_->getSolveCount();
    }

private:
    ISTLSolver<TypeTag> cpuSolver_;
};
} // namespace Opm::gpuistl

#endif
