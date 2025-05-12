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
#include <string>

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
     * @brief Structure to hold well data
     */
    struct WellData {
        MatrixTuple matrices;
        GpuVector<int> cellIndices;
        bool should_skip;
        std::string name;

        // Constructor that takes moved objects
        WellData(std::string wellName, MatrixTuple&& mats, GpuVector<int>&& cells, bool skip)
            : matrices(std::move(mats))
            , cellIndices(std::move(cells))
            , should_skip(skip)
            , name(std::move(wellName))
        {}

        // Delete copy constructors to prevent accidental copying
        WellData(const WellData&) = delete;
        WellData& operator=(const WellData&) = delete;

        // Allow move operations
        WellData(WellData&&) noexcept = default;
        WellData& operator=(WellData&&) noexcept = default;
    };

    /**
     * @brief Add a well's matrices to the storage
     * @param wellName Name of the well for tracking
     * @param matrices Tuple of matrices for a well
     * @param cellIndices Vector of global cell indices for this well's perforations
     * @param should_skip Whether to skip this well
     */
    void addWell(const std::string& wellName,
                MatrixTuple&& matrices,
                GpuVector<int>&& cellIndices,
                bool should_skip)
    {
        // Check if the well already exists
        for (size_t i = 0; i < well_data_.size(); ++i) {
            if (well_data_[i]->name == wellName) {
                // Remove the old well data
                well_data_.erase(well_data_.begin() + i);
                break;
            }
        }
        well_data_.push_back(
            std::make_unique<WellData>(
                wellName,
                std::move(matrices),
                std::move(cellIndices),
                should_skip
            )
        );
    }


    /**
     * @brief Remove a well from the storage
     * @param wellName Name of the well to remove
     */
    void removeWell(const std::string& wellName) {
        auto it = std::remove_if(well_data_.begin(), well_data_.end(),
            [&](const auto& well) {
                return well->name == wellName;
            });
        well_data_.erase(it, well_data_.end());
    }

    /**
     * @brief Clear all stored matrices and indices
     */
    void clear() {
        well_data_.clear();
    }

    /**
     * @brief Check if a well exists in the storage
     * @param wellName Name of the well to check
     * @return True if the well exists, false otherwise
     */
    bool hasWell(const std::string& wellName) const {
        return findWellIndex(wellName) != -1;
    }

    /**
     * @brief Update properties of an existing well without changing its matrices
     * @param wellName Name of the well to update
     * @param should_skip New skip flag for this well
     * @return True if the well was updated successfully
     * @throws std::out_of_range if wellName is not found
     */
    bool updateWellProperties(const std::string& wellName, bool should_skip)
    {
        int idx = findWellIndex(wellName);
        if (idx == -1) {
            throw std::out_of_range("Well name " + wellName + " not found in updateWellProperties");
        }
        well_data_[idx]->should_skip = should_skip;
        return true;
    }

    /**
     * @brief Get access to a well's matrices
     * @param wellName Name of the well
     * @return Reference to the matrix tuple
     * @throws std::out_of_range if wellName is not found
     */
    const MatrixTuple& getWellMatrices(const std::string& wellName) const {
        int idx = findWellIndex(wellName);
        if (idx == -1) {
            throw std::out_of_range("Well name " + wellName + " not found in getWellMatrices");
        }
        return well_data_[idx]->matrices;
    }

    /**
     * @brief Get the cell indices for a specific well
     * @param wellName Name of the well
     * @return Reference to the vector of cell indices
     * @throws std::out_of_range if wellName is not found
     */
    const GpuVector<int>& getWellCellIndices(const std::string& wellName) const {
        int idx = findWellIndex(wellName);
        if (idx == -1) {
            throw std::out_of_range("Well name " + wellName + " not found in getWellCellIndices");
        }
        return well_data_[idx]->cellIndices;
    }

    /**
     * @brief Get the skip flag for a specific well
     * @param wellName Name of the well
     * @return True if the well should be skipped
     * @throws std::out_of_range if wellName is not found
     */
    bool shouldSkip(const std::string& wellName) const {
        int idx = findWellIndex(wellName);
        if (idx == -1) {
            throw std::out_of_range("Well name " + wellName + " not found in shouldSkip");
        }
        return well_data_[idx]->should_skip;
    }

    /**
     * @brief Get all well names
     * @return Vector of well names
     */
    std::vector<std::string> getWellNames() const {
        std::vector<std::string> names;
        names.reserve(well_data_.size());
        for (const auto& well : well_data_) {
            names.push_back(well->name);
        }
        return names;
    }

    /**
     * @brief Check if there are any matrices stored
     * @return True if no wells are stored
     */
    bool empty() const {
        return well_data_.empty();
    }

    /**
     * @brief Get the number of wells stored
     * @return Count of wells in storage
     */
    std::size_t size() const {
        return well_data_.size();
    }

private:
    /** Storage for well data - using a vector of unique_ptr to avoid any copy operations */
    std::vector<std::unique_ptr<WellData>> well_data_;

    /**
     * @brief Find the index of a well in the well_data_ vector
     * @param wellName Name of the well to find
     * @return Index of the well in the vector, or -1 if not found
     */
    std::size_t findWellIndex(const std::string& wellName) const {
        auto wd_size = well_data_.size();
        for (auto i = 0*wd_size; i < wd_size; ++i) {
            if (well_data_[i]->name == wellName) {
                return i;
            }
        }
        return -1;
    }
};

} // namespace Opm::gpuistl

#endif // OPM_GPU_WELL_MATRICES_HPP
