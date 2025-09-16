/*
  Copyright 2025 Equinor ASA.

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

#ifndef OPM_GPU_PRESSURE_TRANSFER_POLICY_HEADER_INCLUDED
#define OPM_GPU_PRESSURE_TRANSFER_POLICY_HEADER_INCLUDED

#include <opm/simulators/linalg/PropertyTree.hpp>
#include <opm/simulators/linalg/WellOperators.hpp>
#include <opm/simulators/linalg/gpuistl/GpuOwnerOverlapCopy.hpp>
#include <opm/simulators/linalg/gpuistl/GpuSparseMatrixWrapper.hpp>
#include <opm/simulators/linalg/gpuistl/GpuVector.hpp>
#include <opm/simulators/linalg/gpuistl/detail/cpr_amg_operations.hpp>
#include <opm/simulators/linalg/twolevelmethodcpr.hh>

#include <cstddef>

namespace Opm
{
namespace Details
{
    template <class Scalar>
    using GpuPressureMatrixType = gpuistl::GpuSparseMatrixWrapper<Scalar>;
    template <class Scalar>
    using GpuPressureVectorType = gpuistl::GpuVector<Scalar>;
    template <class Scalar>
    using GpuSeqCoarseOperatorType = Dune::MatrixAdapter<GpuPressureMatrixType<Scalar>,
                                     GpuPressureVectorType<Scalar>,
                                     GpuPressureVectorType<Scalar>>;

    template <class Scalar, class Comm>

    using GpuParCoarseOperatorType
        = Dune::OverlappingSchwarzOperator<GpuPressureMatrixType<Scalar>,
                                      GpuPressureVectorType<Scalar>,
                                      GpuPressureVectorType<Scalar>,
                                      Comm>;
    template <class Scalar, class Comm>
    using GpuCoarseOperatorType = std::conditional_t<std::is_same<Comm, Dune::Amg::SequentialInformation>::value,
                                                  GpuSeqCoarseOperatorType<Scalar>,
                                                  GpuParCoarseOperatorType<Scalar, Comm>>;
} // namespace Details
} // namespace Opm

namespace Opm::gpuistl
{
template <class FineOperator, class Communication, class Scalar, bool transpose = false>
class GpuPressureTransferPolicy
    : public Dune::Amg::LevelTransferPolicyCpr<FineOperator, Details::GpuCoarseOperatorType<Scalar, Communication>>
{
public:
    using CoarseOperator = typename Details::GpuCoarseOperatorType<Scalar, Communication>;
    using ParentType = Dune::Amg::LevelTransferPolicyCpr<FineOperator, CoarseOperator>;
    using ParallelInformation = Communication;
    using FineVectorType = typename FineOperator::domain_type;

public:
    GpuPressureTransferPolicy(const Communication& comm,
                              const FineVectorType& weights,
                              [[maybe_unused]] const PropertyTree& prm,
                              int pressure_var_index)
        : communication_(&const_cast<Communication&>(comm))
        , weights_(weights)
        , pressure_var_index_(pressure_var_index)
    {
    }


    void createCoarseLevelSystem(const FineOperator& fineOperator) override
    {
        using CoarseMatrix = typename CoarseOperator::matrix_type;
        const auto& fineLevelMatrix = fineOperator.getmat();
        // Create coarse level matrix on GPU with block size 1 but same sparsity pattern
        coarseLevelMatrix_ = std::make_shared<CoarseMatrix>(fineLevelMatrix.getRowIndices(),
                                                            fineLevelMatrix.getColumnIndices(),
                                                            1 // block size 1
        );

        // Calculate entries for coarse matrix
        calculateCoarseEntries(fineOperator);

        // Create coarse level communication with block size 1
        if constexpr (std::is_same_v<Communication, Dune::Amg::SequentialInformation>) {
            coarseLevelCommunication_.reset(communication_, [](Communication*) {});
        } else {
            // For GPU MPI case, we need to create a new communication object with block size 1
            // Extract the CPU communication from the fine level GPU communication
            const auto& cpuCommunication = communication_->getCpuCommunication();
            // Create new GPU communication with block size 1 for the coarse system
            coarseLevelCommunication_ = makeGpuOwnerOverlapCopy<Scalar, 1>(cpuCommunication);
        }

        this->lhs_.resize(coarseLevelMatrix_->N());
        this->rhs_.resize(coarseLevelMatrix_->N());

        // Create a MatrixAdapter that wraps the matrix without copying it
        // For GPU MPI case, we need to pass the communication object
        if constexpr (std::is_same_v<Communication, Dune::Amg::SequentialInformation>) {
            this->operator_ = std::make_shared<CoarseOperator>(*coarseLevelMatrix_);
        } else {
            // For GPU MPI case, pass the GPU communication object directly
            this->operator_ = std::make_shared<CoarseOperator>(*coarseLevelMatrix_, *coarseLevelCommunication_);
        }
    }

    void calculateCoarseEntries(const FineOperator& fineOperator) override
    {
        const auto& fineLevelMatrix = fineOperator.getmat();

        // Set coarse matrix to zero
        coarseLevelMatrix_->getNonZeroValues() = 0.0;

        // Calculate coarse entries on GPU
        detail::calculateCoarseEntries<Scalar, transpose>(fineLevelMatrix, *coarseLevelMatrix_, weights_, pressure_var_index_);
    }

    void moveToCoarseLevel(const typename ParentType::FineRangeType& fine) override
    {
        // Set coarse vector to zero
        this->rhs_ = 0;

        // Restrict vector on GPU
        detail::restrictVector<Scalar, transpose>(fine, this->rhs_, weights_, pressure_var_index_);

        this->lhs_ = 0;
    }

    void moveToFineLevel(typename ParentType::FineDomainType& fine) override
    {
        // Prolongate vector on GPU
        detail::prolongateVector<Scalar, transpose>(this->lhs_, fine, weights_, pressure_var_index_);
    }

    GpuPressureTransferPolicy* clone() const override
    {
        return new GpuPressureTransferPolicy(*this);
    }

    const Communication& getCoarseLevelCommunication() const
    {
        return *coarseLevelCommunication_;
    }

    std::size_t getPressureIndex() const
    {
        return pressure_var_index_;
    }

private:
    Communication* communication_;
    const FineVectorType& weights_;
    const std::size_t pressure_var_index_;
    std::shared_ptr<Communication> coarseLevelCommunication_;
    std::shared_ptr<typename CoarseOperator::matrix_type> coarseLevelMatrix_;
};

} // namespace Opm::gpuistl

#endif // OPM_GPU_PRESSURE_TRANSFER_POLICY_HEADER_INCLUDED
