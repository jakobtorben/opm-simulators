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

#ifndef OPM_GPU_WELL_MATRICES_HPP
#define OPM_GPU_WELL_MATRICES_HPP

#include <opm/simulators/linalg/gpuistl/GpuMatrix.hpp>
#include <opm/simulators/linalg/gpuistl/GpuVector.hpp>
#include <memory>
#include <vector>
#include <tuple>

namespace Opm::gpuistl
{

class GpuWellMatricesBase {
public:
    virtual ~GpuWellMatricesBase() = default;
};

/**
 * @brief Storage class for GPU Well matrices
 *
 * This class holds GPU-compatible matrices for multiple wells, to be used in
 * well operator implementations for GPU-based linear solvers.
 *
 * @tparam Scalar The scalar type to use (double or float)
 */
template <class Scalar>
class GpuWellMatrices : public GpuWellMatricesBase {
public:
    /**
     * @brief Tuple type for a well's matrices (B, C, invD)
     *
     * B - Well-to-reservoir matrix
     * C - Reservoir-to-well matrix
     * invD - Inverted well diagonal matrix
     */
    using MatrixTuple = std::tuple<std::unique_ptr<gpuistl::GpuMatrix<Scalar>>,
                                 std::unique_ptr<gpuistl::GpuMatrix<Scalar>>,
                                 std::unique_ptr<gpuistl::GpuMatrix<Scalar>>>;

    /**
     * @brief Add a well's matrices to the storage
     * @param matrices Tuple of matrices for a well
     * @param cellIndices Vector of global cell indices for this well's perforations
     * @param should_skip Whether to skip this well
     */
    void addWell(MatrixTuple&& matrices, GpuVector<int>&& cellIndices, bool should_skip) {
        wellMatrices_.push_back(std::move(matrices));
        wellCellIndices_.push_back(std::move(cellIndices));
        should_skip_.push_back(should_skip);
    }

    /**
     * @brief Clear all stored matrices and indices
     */
    void clear() {
        wellMatrices_.clear();
        wellCellIndices_.clear();
    }

    /**
     * @brief Get access to the stored matrices
     * @return Reference to the vector of matrix tuples
     */
    const std::vector<MatrixTuple>& getMatrices() const {
        return wellMatrices_;
    }

    /**
     * @brief Get the cell indices for a specific well
     * @param wellIndex Index of the well
     * @return Reference to the vector of cell indices
     */
    const GpuVector<int>& getWellCellIndices(size_t wellIndex) const {
        if (wellIndex >= wellCellIndices_.size()) {
            throw std::out_of_range("Well index out of range");
        }
        return wellCellIndices_[wellIndex];
    }

    /**
     * @brief Get the skip flag for a specific well
     * @param wellIndex Index of the well
     * @return True if the well should be skipped
     */
    bool shouldSkip(size_t wellIndex) const {
        if (wellIndex >= should_skip_.size()) {
            throw std::out_of_range("Well index out of range");
        }
        return should_skip_[wellIndex];
    }

    /**
     * @brief Check if there are any matrices stored
     * @return True if no wells are stored
     */
    bool empty() const {
        return wellMatrices_.empty();
    }

    /**
     * @brief Get the number of wells stored
     * @return Count of wells in storage
     */
    size_t size() const {
        return wellMatrices_.size();
    }

private:
    /** Storage for well matrices */
    std::vector<MatrixTuple> wellMatrices_;

    /** Storage for cell indices for each well */
    std::vector<GpuVector<int>> wellCellIndices_;

    /** Storage for whether to skip this well */
    std::vector<bool> should_skip_;
};

} // namespace Opm::gpuistl

#endif // OPM_GPU_WELL_MATRICES_HPP
