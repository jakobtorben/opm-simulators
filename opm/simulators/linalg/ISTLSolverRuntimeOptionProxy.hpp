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

#ifndef OPM_ISTLSOLVERRUNTIMEOPTIONPROXY_HEADER_INCLUDED
#define OPM_ISTLSOLVERRUNTIMEOPTIONPROXY_HEADER_INCLUDED

#include "opm/simulators/linalg/setupPropertyTree.hpp"
#include <opm/simulators/linalg/AbstractISTLSolver.hpp>
#include <opm/simulators/linalg/ISTLSolver.hpp>

namespace Opm
{
template <class TypeTag>
class ISTLSolverRuntimeOptionProxy : public AbstractISTLSolver<TypeTag>
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


    static void registerParameters()
    {
        FlowLinearSolverParameters::registerParameters();
    }

    /// Construct a system solver.
    /// \param[in] simulator   The opm-models simulator object
    /// \param[in] parameters  Explicit parameters for solver setup, do not
    ///                        read them from command line parameters.
    /// \param[in] forceSerial If true, will set up a serial linear solver only,
    ///                        local to the current rank, instead of creating a
    ///                        parallel (MPI distributed) linear solver.
    ISTLSolverRuntimeOptionProxy(const Simulator& simulator,
                                 const FlowLinearSolverParameters& parameters,
                                 bool forceSerial = false)
    {
        createSolver(simulator, parameters, forceSerial);
    }

    /// Construct a system solver.
    /// \param[in] simulator   The opm-models simulator object
    explicit ISTLSolverRuntimeOptionProxy(const Simulator& simulator)
    {
        createSolver(simulator);
    }


    void eraseMatrix() override
    {
        istlSolver_->eraseMatrix();
    }

    void setActiveSolver(int num) override
    {
        istlSolver_->setActiveSolver(num);
    }

    int numAvailableSolvers() const override
    {
        return istlSolver_->numAvailableSolvers();
    }

    void prepare(const SparseMatrixAdapter& M, Vector& b) override
    {
        istlSolver_->prepare(M, b);
    }

    void prepare(const Matrix& M, Vector& b) override
    {
        istlSolver_->prepare(M, b);
    }

    void setResidual(Vector& b) override
    {
        istlSolver_->setResidual(b);
    }

    void getResidual(Vector& b) const override
    {
        istlSolver_->getResidual(b);
    }

    void setMatrix(const SparseMatrixAdapter& M) override
    {
        istlSolver_->setMatrix(M);
    }

    bool solve(Vector& x) override
    {
        return istlSolver_->solve(x);
    }

    int iterations() const override
    {
        return istlSolver_->iterations();
    }

    const CommunicationType* comm() const override
    {
        return istlSolver_->comm();
    }

    int getSolveCount() const override
    {
        return istlSolver_->getSolveCount();
    }

private:
    std::unique_ptr<AbstractISTLSolver<TypeTag>> istlSolver_;

    template <class... Args>
    void createSolver(const Simulator& simulator, Args&&... args)
    {
        // TODO: We need to use the parameters sent in in the constructor
        // instead of the default ones.
        FlowLinearSolverParameters p;
        p.init(simulator.vanguard().eclState().getSimulationConfig().useCPR());
        const auto prm = setupPropertyTree(p,
                                           Parameters::IsSet<Parameters::LinearSolverMaxIter>(),
                                           Parameters::IsSet<Parameters::LinearSolverReduction>());
        const auto backend = prm.get<std::string>("backend", "cpu");
        
        istlSolver_ = std::make_unique<ISTLSolver<TypeTag>>(simulator, std::forward<Args>(args)...);
    }
};
} // namespace Opm

#endif
