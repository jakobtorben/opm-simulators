/*
  Copyright 2022-2023 SINTEF AS

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
#ifndef OPM_SOLVERADAPTER_HPP
#define OPM_SOLVERADAPTER_HPP

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/paamg/pinfo.hh>
#include <dune/istl/schwarz.hh>
#include <dune/istl/solver.hh>

#include <opm/common/ErrorMacros.hpp>

#include <opm/simulators/linalg/WellOperators.hpp>
#include <opm/simulators/linalg/gpuistl/GpuBlockPreconditioner.hpp>
#include <opm/simulators/linalg/gpuistl/GpuSparseMatrix.hpp>
#include <opm/simulators/linalg/gpuistl/GpuVector.hpp>
#include <opm/simulators/linalg/gpuistl/PreconditionerAdapter.hpp>
#include <opm/simulators/linalg/gpuistl/detail/has_function.hpp>

#if HAVE_MPI
#include <opm/simulators/linalg/gpuistl/GpuOwnerOverlapCopy.hpp>
#endif

#ifdef OPEN_MPI
#if OPEN_MPI
#include "mpi-ext.h"
#endif
#endif

#include <memory>

namespace Opm::gpuistl
{

// GPU-specific well operator that adapts a CPU well operator to work with GPU vectors
template <class X, class XGPU>
class GpuWellOperator : public Opm::LinearOperatorExtra<XGPU, XGPU> {
public:
    using Base = Opm::LinearOperatorExtra<XGPU, XGPU>;
    using typename Base::field_type;
    using typename Base::PressureMatrix;
    static constexpr auto block_size = X::block_type::dimension;

    // Use shared_ptr instead of reference to ensure the CPU operator stays alive
    explicit GpuWellOperator(std::shared_ptr<const Opm::LinearOperatorExtra<X, X>> cpuOp)
        : m_cpuOp(std::move(cpuOp)) {}

    void apply(const XGPU& x, XGPU& y) const override {
        // Convert GPU vectors to CPU for the well operator
        const int numBlocks = x.dim() / block_size;
        X x_cpu(numBlocks);
        X y_cpu(numBlocks); 

        // Copy data from GPU to CPU
        x.copyToHost(x_cpu);
        y.copyToHost(y_cpu);

        // Apply the CPU well operator
        m_cpuOp->apply(x_cpu, y_cpu);

        // Copy results back to GPU
        y.copyFromHost(y_cpu);
    }

    void applyscaleadd(field_type alpha, const XGPU& x, XGPU& y) const override {
        // Convert GPU vectors to CPU for the well operator
        const int numBlocks = x.dim() / block_size;
        X x_cpu(numBlocks);
        X y_cpu(numBlocks); 

        // Copy data from GPU to CPU
        x.copyToHost(x_cpu);
        y.copyToHost(y_cpu);

        // Apply the CPU well operator
        m_cpuOp->applyscaleadd(alpha, x_cpu, y_cpu);

        // Copy results back to GPU
        y.copyFromHost(y_cpu);
    }

    Dune::SolverCategory::Category category() const override {
        return m_cpuOp->category();
    }

    void addWellPressureEquations(PressureMatrix& jacobian,
                                 const XGPU& weights,
                                 const bool use_well_weights) const override {

        const int numBlocks = weights.dim() / block_size;
        X weights_cpu(numBlocks);

        // Copy data from GPU to CPU
        weights.copyToHost(weights_cpu);

        // Forward to CPU version
        m_cpuOp->addWellPressureEquations(jacobian, weights_cpu, use_well_weights);
    }

    void addWellPressureEquationsStruct(PressureMatrix& jacobian) const override {
        m_cpuOp->addWellPressureEquationsStruct(jacobian);
    }

    int getNumberOfExtraEquations() const override {
        return m_cpuOp->getNumberOfExtraEquations();
    }

private:
    // Store a shared_ptr to the original CPU operator
    std::shared_ptr<const Opm::LinearOperatorExtra<X, X>> m_cpuOp;
};


//! @brief Wraps a CUDA solver to work with CPU data.
//!
//! @tparam Operator the Dune::LinearOperator to use
//! @tparam UnderlyingSolver a Dune solver like class, eg Dune::BiCGSTABSolver
//! @tparam X the outer type to use (eg. Dune::BlockVector<Dune::FieldVector<...>>)
template <class Operator, template <class> class UnderlyingSolver, class X>
class SolverAdapter : public Dune::IterativeSolver<X, X>
{
public:
    using typename Dune::IterativeSolver<X, X>::domain_type;
    using typename Dune::IterativeSolver<X, X>::range_type;
    using typename Dune::IterativeSolver<X, X>::field_type;
    using typename Dune::IterativeSolver<X, X>::real_type;
    using typename Dune::IterativeSolver<X, X>::scalar_real_type;
    static constexpr auto block_size = domain_type::block_type::dimension;
    using XGPU = Opm::gpuistl::GpuVector<real_type>;

    //! @brief constructor
    //!
    //! @param op the linear operator (assumed CPU, the output (matrix) of which will be converted to a GPU variant)
    //! @param sp the scalar product (assumed CPU, this will be converted to a GPU variant)
    //! @param reduction the reduction factor passed to the iterative solver
    //! @param maxit maximum number of iterations for the linear solver
    //! @param verbose verbosity level
    //! @param comm the communication object. If this is Dune::Amg::SequentialInformation, we assume a serial setup
    //!
    //! @todo Use a std::forward in this function
    template<class Comm>
    SolverAdapter(Operator& op,
                  Dune::ScalarProduct<X>& sp,
                  std::shared_ptr<Dune::Preconditioner<X, X>> prec,
                  scalar_real_type reduction,
                  int maxit,
                  int verbose,
                  const Comm& comm)
        : Dune::IterativeSolver<X, X>(op, sp, *prec, reduction, maxit, verbose)
        , m_opOnCPUWithMatrix(op)
        , m_matrix(GpuSparseMatrix<real_type>::fromMatrix(op.getmat()))
        , m_underlyingSolver(constructSolver(prec, reduction, maxit, verbose, comm))
    {
        OPM_ERROR_IF(
            (detail::is_a_well_operator<Operator>::value && !std::is_same_v<Comm, Dune::Amg::SequentialInformation>),
            "Currently we only support well operators in serial for the CUDA/HIP solver. "
            "Use --matrix-add-well-contributions=true. "
            "Using WellModelMatrixAdapter with SolverAdapter in parallel is not well-defined so it will not work well, or at all.");
    }

    virtual void apply(X& x, X& b, double reduction, Dune::InverseOperatorResult& res) override
    {
        // TODO: Can we do this without reimplementing the other function?
        // TODO: [perf] Do we need to update the matrix every time? Probably yes
        m_matrix.updateNonzeroValues(m_opOnCPUWithMatrix.getmat());

        if (!m_inputBuffer) {
            m_inputBuffer.reset(new XGPU(b.dim()));
            m_outputBuffer.reset(new XGPU(x.dim()));
        }

        m_inputBuffer->copyFromHost(b);
        // TODO: [perf] do we need to copy x here?
        m_outputBuffer->copyFromHost(x);

        m_underlyingSolver.apply(*m_outputBuffer, *m_inputBuffer, reduction, res);

        // TODO: [perf] do we need to copy b here?
        m_inputBuffer->copyToHost(b);
        m_outputBuffer->copyToHost(x);
    }
    virtual void apply(X& x, X& b, Dune::InverseOperatorResult& res) override
    {
        // TODO: [perf] Do we need to update the matrix every time? Probably yes
        m_matrix.updateNonzeroValues(m_opOnCPUWithMatrix.getmat());

        if (!m_inputBuffer) {
            m_inputBuffer.reset(new XGPU(b.dim()));
            m_outputBuffer.reset(new XGPU(x.dim()));
        }

        m_inputBuffer->copyFromHost(b);
        // TODO: [perf] do we need to copy x here?
        m_outputBuffer->copyFromHost(x);

        m_underlyingSolver.apply(*m_outputBuffer, *m_inputBuffer, res);

        // TODO: [perf] do we need to copy b here?
        m_inputBuffer->copyToHost(b);
        m_outputBuffer->copyToHost(x);
    }

private:
    Operator& m_opOnCPUWithMatrix;
    GpuSparseMatrix<real_type> m_matrix;

    UnderlyingSolver<XGPU> m_underlyingSolver;

    // TODO: Use a std::forward
    // This is the MPI parallel case (general communication object)
#if HAVE_MPI
    template <class Comm>
    UnderlyingSolver<XGPU> constructSolver(std::shared_ptr<Dune::Preconditioner<X, X>> prec,
                                           scalar_real_type reduction,
                                           int maxit,
                                           int verbose,
                                           const Comm& communication)
    {
        // TODO: See the below TODO over the definition of precHolder in the other overload of constructSolver
        // TODO: We are currently double wrapping preconditioners in the preconditioner factory to be extra
        //       compatible with CPU. Probably a cleaner way eventually would be to do more modifications to the
        //       flexible solver to accomodate the pure GPU better.
        auto precAsHolder = std::dynamic_pointer_cast<PreconditionerHolder<X, X>>(prec);
        if (!precAsHolder) {
            OPM_THROW(std::invalid_argument,
                      "The preconditioner needs to be a CUDA preconditioner (eg. GPUDILU) wrapped in a "
                      "Opm::gpuistl::PreconditionerAdapter wrapped in a "
                      "Opm::gpuistl::GpuBlockPreconditioner. If you are unsure what this means, set "
                      "preconditioner to 'GPUDILU'");
        }

        auto preconditionerAdapter = precAsHolder->getUnderlyingPreconditioner();
        auto preconditionerAdapterAsHolder
            = std::dynamic_pointer_cast<PreconditionerHolder<XGPU, XGPU>>(preconditionerAdapter);
        if (!preconditionerAdapterAsHolder) {
            OPM_THROW(std::invalid_argument,
                      "The preconditioner needs to be a CUDA preconditioner (eg. GPUDILU) wrapped in a "
                      "Opm::gpuistl::PreconditionerAdapter wrapped in a "
                      "Opm::gpuistl::GpuBlockPreconditioner. If you are unsure what this means, set "
                      "preconditioner to 'GPUDILU'");
        }
        // We need to get the underlying preconditioner:
        auto preconditionerReallyOnGPU = preconditionerAdapterAsHolder->getUnderlyingPreconditioner();

        // Temporary solution use the GPU Direct communication solely based on these prepcrosessor statements
        bool mpiSupportsCudaAwareAtCompileTime = false;
        bool mpiSupportsCudaAwareAtRunTime = false;

#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
        mpiSupportsCudaAwareAtCompileTime = true;
#endif /* MPIX_CUDA_AWARE_SUPPORT */

#if defined(MPIX_CUDA_AWARE_SUPPORT)
        if (1 == MPIX_Query_cuda_support()) {
            mpiSupportsCudaAwareAtRunTime = true;
        }
#endif /* MPIX_CUDA_AWARE_SUPPORT */


        // TODO add typename Operator communication type as a named type with using
        std::shared_ptr<Opm::gpuistl::GPUSender<real_type, Comm>> gpuComm;
        if (mpiSupportsCudaAwareAtCompileTime && mpiSupportsCudaAwareAtRunTime) {
            gpuComm = std::make_shared<
                Opm::gpuistl::GPUAwareMPISender<real_type, block_size, Comm>>(
                communication);
        } else {
            gpuComm = std::make_shared<
                Opm::gpuistl::GPUObliviousMPISender<real_type, block_size, Comm>>(
                communication);
        }

        using CudaCommunication = GpuOwnerOverlapCopy<real_type, block_size, Comm>;
        using SchwarzOperator
            = Dune::OverlappingSchwarzOperator<GpuSparseMatrix<real_type>, XGPU, XGPU, CudaCommunication>;
        auto cudaCommunication = std::make_shared<CudaCommunication>(gpuComm);

        auto mpiPreconditioner = std::make_shared<GpuBlockPreconditioner<XGPU, XGPU, CudaCommunication>>(
            preconditionerReallyOnGPU, cudaCommunication);

        auto scalarProduct = std::make_shared<Dune::ParallelScalarProduct<XGPU, CudaCommunication>>(
            cudaCommunication, m_opOnCPUWithMatrix.category());


        // NOTE: Ownership of cudaCommunication is handled by mpiPreconditioner. However, just to make sure we
        //       remember this, we add this check. So remember that we hold one count in this scope, and one in the
        //       GpuBlockPreconditioner instance. We accommodate for the fact that it could be passed around in
        //       GpuBlockPreconditioner, hence we do not test for != 2
        OPM_ERROR_IF(cudaCommunication.use_count() < 2, "Internal error. Shared pointer not owned properly.");
        auto overlappingCudaOperator = std::make_shared<SchwarzOperator>(m_matrix, *cudaCommunication);

        return UnderlyingSolver<XGPU>(
            overlappingCudaOperator, scalarProduct, mpiPreconditioner, reduction, maxit, verbose);
    }
#endif

    // This is the serial case (specific overload for Dune::Amg::SequentialInformation)
    UnderlyingSolver<XGPU> constructSolver(std::shared_ptr<Dune::Preconditioner<X, X>> prec,
                                           scalar_real_type reduction,
                                           int maxit,
                                           int verbose,
                                           [[maybe_unused]] const Dune::Amg::SequentialInformation& communication)
    {
        // Dune::Amg::SequentialInformation is the serial case
        return constructSolver(prec, reduction, maxit, verbose);
    }

    // TODO: Use a std::forward
    UnderlyingSolver<XGPU> constructSolver(std::shared_ptr<Dune::Preconditioner<X, X>> prec,
                                           scalar_real_type reduction,
                                           int maxit,
                                           int verbose)
    {
        // TODO: Fix the reliance on casting here. This is a code smell to a certain degree, and assumes
        //       a certain setup beforehand. The only reason we do it this way is to minimize edits to the
        //       flexible solver. We could design it differently, but keep this for the time being until
        //       we figure out how we want to GPU-ify the rest of the system.
        auto precAsHolder = std::dynamic_pointer_cast<PreconditionerHolder<XGPU, XGPU>>(prec);
        if (!precAsHolder) {
            OPM_THROW(std::invalid_argument,
                      "The preconditioner needs to be a CUDA preconditioner wrapped in a "
                      "Opm::gpuistl::PreconditionerHolder (eg. GPUDILU).");
        }
        auto preconditionerOnGPU = precAsHolder->getUnderlyingPreconditioner();

        auto matrixOperator = std::make_shared<Dune::MatrixAdapter<GpuSparseMatrix<real_type>, XGPU, XGPU>>(m_matrix);

        // Try to get the well operator from the original operator
        using SeqOperatorType = Opm::WellModelMatrixAdapter<Dune::BCRSMatrix<Opm::MatrixBlock<real_type, block_size, block_size>>, X, X>;
        auto* wellOp = dynamic_cast<const SeqOperatorType*>(&m_opOnCPUWithMatrix);

        // Create our scalar product
        auto scalarProduct = std::make_shared<Dune::SeqScalarProduct<XGPU>>();

        // Check if we have a well operator
        if (wellOp) {
            // Create a shared_ptr to the well operator to ensure it stays alive
            // The shared_ptr will be owned by the GpuWellOperator
            m_wellOperator = std::shared_ptr<const Opm::LinearOperatorExtra<X, X>>(
                &wellOp->getWellOperator(),
                [](const Opm::LinearOperatorExtra<X, X>*) { /* non-owning deleter */ }
            );

            // Create adapter to convert between CPU and GPU well operators
            m_gpuWellOperator = std::make_shared<GpuWellOperator<X, XGPU>>(m_wellOperator);

            // Create matrix adapter for the GPU
            m_gpuMatrixOperator = std::make_shared<Opm::WellModelMatrixAdapter<GpuSparseMatrix<real_type>, XGPU, XGPU>>(
                m_matrix, *m_gpuWellOperator);

            return UnderlyingSolver<XGPU>(m_gpuMatrixOperator, scalarProduct, preconditionerOnGPU, 
                                          reduction, maxit, verbose);
        } else {
            // No well operator, use the basic matrix operator
            return UnderlyingSolver<XGPU>(matrixOperator, scalarProduct, preconditionerOnGPU, 
                                          reduction, maxit, verbose);
        }
    }

    std::unique_ptr<XGPU> m_inputBuffer;
    std::unique_ptr<XGPU> m_outputBuffer;

    // Store the GPU well operator to manage its lifetime
    // These must be kept alive as long as m_underlyingSolver is alive
    std::shared_ptr<const Opm::LinearOperatorExtra<X, X>> m_wellOperator;
    std::shared_ptr<GpuWellOperator<X, XGPU>> m_gpuWellOperator;
    std::shared_ptr<Opm::WellModelMatrixAdapter<GpuSparseMatrix<real_type>, XGPU, XGPU>> m_gpuMatrixOperator;
};

} // namespace Opm::gpuistl

#endif
