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

#ifndef OPM_STANDARDPRECONDITIONERS_GPU_MPI_HEADER
#define OPM_STANDARDPRECONDITIONERS_GPU_MPI_HEADER

#include <dune/istl/bcrsmatrix.hh>
#include <string>

#include <opm/simulators/linalg/PreconditionerFactory.hpp>
#include <opm/simulators/linalg/PropertyTree.hpp>
#include <opm/simulators/linalg/is_gpu_operator.hpp>

#if HAVE_CUDA
#include <opm/simulators/linalg/gpuistl/GpuDILU.hpp>
#include <opm/simulators/linalg/gpuistl/GpuSeqILU0.hpp>
#include <opm/simulators/linalg/gpuistl/GpuJac.hpp>
#include <opm/simulators/linalg/gpuistl/OpmGpuILU0.hpp>
#include <opm/simulators/linalg/gpuistl/GpuBlockPreconditioner.hpp>
#include <opm/simulators/linalg/gpuistl/detail/gpu_preconditioner_utils.hpp>
#endif

namespace Opm {

template <class Operator, class Comm>
struct StandardPreconditioners<Operator, Comm, typename std::enable_if_t<Opm::is_gpu_operator_v<Operator>>>
{
    using O = Operator;
    using C = Comm;
    using F = PreconditionerFactory<O, C>;
    using M = typename F::Matrix;
    using V = typename F::Vector;
    using P = PropertyTree;
    using PrecPtr = typename F::PrecPtr;

    using field_type = typename V::field_type;

    static constexpr int maxblocksize = 6;

    static void add()
    {
#if HAVE_CUDA
        F::addCreator("ilu0", [](const O& op, const P& prm, const std::function<V()>&, std::size_t, const C& comm) {
            const double w = prm.get<double>("relaxation", 1.0);
            using GpuILU0 = typename gpuistl::GpuSeqILU0<M, V, V>;
            auto gpuPrec = std::make_shared<GpuILU0>(op.getmat(), w);
            return std::make_shared<gpuistl::GpuBlockPreconditioner<V, V, C>>(gpuPrec, comm);
        });

        F::addCreator("jac", [](const O& op, const P& prm, const std::function<V()>&, std::size_t, const C& comm) {
            const double w = prm.get<double>("relaxation", 1.0);
            using GpuJac = typename gpuistl::GpuJac<M, V, V>;
            auto gpuPrec = std::make_shared<GpuJac>(op.getmat(), w);
            return std::make_shared<gpuistl::GpuBlockPreconditioner<V, V, C>>(gpuPrec, comm);
        });

        F::addCreator("dilu", [](const O& op, [[maybe_unused]] const P& prm, const std::function<V()>&, std::size_t, const C& comm) -> PrecPtr {
            return op.getmat().dispatchOnBlocksize([&](auto blockSizeVal) -> PrecPtr {
                constexpr int blockSize = decltype(blockSizeVal)::value;
                const auto cpuMatrix = gpuistl::detail::makeCPUMatrix<O, Dune::FieldMatrix<field_type, blockSize, blockSize>>(op);
                const bool split_matrix = prm.get<bool>("split_matrix", true);
                const bool tune_gpu_kernels = prm.get<bool>("tune_gpu_kernels", true);
                const int mixed_precision_scheme = prm.get<int>("mixed_precision_scheme", 0);
                const bool reorder = prm.get<bool>("reorder", true);
                using CPUMatrixType = std::remove_const_t<std::remove_reference_t<decltype(cpuMatrix)>>;
                using GPUDILU = typename gpuistl::GpuDILU<CPUMatrixType, V, V>;
                auto gpuPrec = std::make_shared<GPUDILU>(op.getmat(), cpuMatrix, split_matrix, tune_gpu_kernels, mixed_precision_scheme, reorder);
                return std::make_shared<gpuistl::GpuBlockPreconditioner<V, V, C>>(gpuPrec, comm);
            });
        });

        F::addCreator("opmilu0", [](const O& op, [[maybe_unused]] const P& prm, const std::function<V()>&, std::size_t, const C& comm) -> PrecPtr {
            return op.getmat().dispatchOnBlocksize([&](auto blockSizeVal) -> PrecPtr {
                constexpr int blockSize = decltype(blockSizeVal)::value;
                const auto cpuMatrix = gpuistl::detail::makeCPUMatrix<O, Dune::FieldMatrix<field_type, blockSize, blockSize>>(op);
                const bool split_matrix = prm.get<bool>("split_matrix", true);
                const bool tune_gpu_kernels = prm.get<bool>("tune_gpu_kernels", true);
                const int mixed_precision_scheme = prm.get<int>("mixed_precision_scheme", 0);
                using CPUMatrixType = std::remove_const_t<std::remove_reference_t<decltype(cpuMatrix)>>;
                using GPUILU0 = typename gpuistl::OpmGpuILU0<CPUMatrixType, V, V>;
                auto gpuPrec = std::make_shared<GPUILU0>(op.getmat(), cpuMatrix, split_matrix, tune_gpu_kernels, mixed_precision_scheme);
                return std::make_shared<gpuistl::GpuBlockPreconditioner<V, V, C>>(gpuPrec, comm);
            });
        });

#if HAVE_HYPRE && HYPRE_USING_CUDA || HYPRE_USING_HIP
         // Only register Hypre preconditioner for double precision
        if constexpr (std::is_same_v<HYPRE_Real, typename V::field_type>) {
            F::addCreator("hypre", [](const O& op, const P& prm, const std::function<V()>&, std::size_t, const C& comm) {
                // Only create Hypre preconditioner for scalar matrices
                if (op.getmat().blockSize() == 1) {
                        return std::make_shared<Hypre::HyprePreconditioner<M, V, V, C>>(op.getmat(), prm, comm);
                    } else {
                        OPM_THROW(std::logic_error, "Hypre preconditioner only works with scalar matrices (block size 1).");
                    }
                });
        }
#endif
        if constexpr (std::is_same_v<O, Dune::OverlappingSchwarzOperator<M, V, V, C>>) {
            F::addCreator(
                "cpr",
                [](const O& op,
                   const P& prm,
                   const std::function<V()> weightsCalculator,
                   std::size_t pressureIndex,
                   const C& comm) {
                    assert(weightsCalculator);
                    if (pressureIndex == std::numeric_limits<std::size_t>::max()) {
                        OPM_THROW(std::logic_error, "Pressure index out of bounds. It needs to specified for CPR");
                    }
                    using Scalar = typename V::field_type;
                    using GpuVector = gpuistl::GpuVector<Scalar>;
                    using LevelTransferPolicy = Opm::gpuistl::GpuPressureTransferPolicy<O, C, Scalar, false>;
                    return std::make_shared<Dune::OwningTwoLevelPreconditioner<O, GpuVector, LevelTransferPolicy, C>>(
                        op, prm, weightsCalculator, pressureIndex, comm);
                });

            F::addCreator(
                "cprt",
                [](const O& op,
                   const P& prm,
                   const std::function<V()> weightsCalculator,
                   std::size_t pressureIndex,
                   const C& comm) {
                    assert(weightsCalculator);
                    if (pressureIndex == std::numeric_limits<std::size_t>::max()) {
                        OPM_THROW(std::logic_error, "Pressure index out of bounds. It needs to specified for CPR");
                    }
                    using Scalar = typename V::field_type;
                    using GpuVector = gpuistl::GpuVector<Scalar>;
                    using LevelTransferPolicy = Opm::gpuistl::GpuPressureTransferPolicy<O, C, Scalar, true>;
                    return std::make_shared<Dune::OwningTwoLevelPreconditioner<O, GpuVector, LevelTransferPolicy, C>>(
                        op, prm, weightsCalculator, pressureIndex, comm);
                });
        }
#endif // HAVE_CUDA
    }
};
}// namespace Opm


#endif // OPM_STANDARDPRECONDITIONERS_GPU_MPI_HEADER
