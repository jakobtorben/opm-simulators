/*
  Copyright 2023 OPM Contributors

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

#ifndef OPM_WELLMATRIXMERGER_HEADER_INCLUDED
#define OPM_WELLMATRIXMERGER_HEADER_INCLUDED

#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>

#include <vector>
#include <map>
#include <unordered_map>
#include <stdexcept>
#include <string>

namespace Opm {

/**
 * @brief Class for merging BCR matrices from multiple wells into consolidated matrices
 * 
 * This class merges the B, C, and D matrices from multiple wells into three large matrices,
 * preserving the well and perforation numbering. It can be used with both standard wells
 * and multi-segment wells.
 * 
 * @tparam Scalar The scalar type (typically double)
 * @tparam numWellEq Number of equations per well
 * @tparam numEq Number of equations per grid cell
 */
template<class Scalar, int numWellEq, int numEq>
class WellMatrixMerger
{
public:
    // Matrix types for well equations
    using DiagMatrixBlockWellType = Dune::FieldMatrix<Scalar, numWellEq, numWellEq>;
    using DiagMatWell = Dune::BCRSMatrix<DiagMatrixBlockWellType>;
    
    using OffDiagMatrixBlockWellType = Dune::FieldMatrix<Scalar, numWellEq, numEq>;
    using OffDiagMatWell = Dune::BCRSMatrix<OffDiagMatrixBlockWellType>;
    
    // Vector types
    using VectorBlockWellType = Dune::FieldVector<Scalar, numWellEq>;
    using BVectorWell = Dune::BlockVector<VectorBlockWellType>;
    
    using VectorBlockType = Dune::FieldVector<Scalar, numEq>;
    using BVector = Dune::BlockVector<VectorBlockType>;
    
    /**
     * @brief Constructor
     */
    WellMatrixMerger();
    
    /**
     * @brief Add a well's matrices to the merged matrices
     * 
     * @param B The B matrix for the well
     * @param C The C matrix for the well
     * @param D The D matrix for the well
     * @param wellCells The indices of cells perforated by the well
     * @param wellIndex The index of the well
     * @param wellName The name of the well (optional)
     */
    void addWell(const OffDiagMatWell& B, 
                 const OffDiagMatWell& C, 
                 const DiagMatWell& D,
                 const std::vector<int>& wellCells,
                 int wellIndex,
                 const std::string& wellName = "");
    
    /**
     * @brief Finalize the matrix merging process
     * 
     * This method must be called after all wells have been added and before
     * accessing the merged matrices.
     */
    void finalize();
    
    /**
     * @brief Get the merged B matrix
     * 
     * @return const OffDiagMatWell& The merged B matrix
     */
    const OffDiagMatWell& getMergedB() const { return mergedB_; }
    
    /**
     * @brief Get the merged C matrix
     * 
     * @return const OffDiagMatWell& The merged C matrix
     */
    const OffDiagMatWell& getMergedC() const { return mergedC_; }
    
    /**
     * @brief Get the merged D matrix
     * 
     * @return const DiagMatWell& The merged D matrix
     */
    const DiagMatWell& getMergedD() const { return mergedD_; }
    
    /**
     * @brief Get the number of wells
     * 
     * @return size_t The number of wells
     */
    size_t getNumWells() const { return wellIndices_.size(); }
    
    /**
     * @brief Get the total number of perforations
     * 
     * @return size_t The total number of perforations
     */
    size_t getNumPerforations() const { return totalPerforations_; }
    
    /**
     * @brief Get the well index for a given cell
     * 
     * @param cellIndex The cell index
     * @return int The well index, or -1 if the cell is not perforated by any well
     */
    int getWellIndexForCell(int cellIndex) const;
    
    /**
     * @brief Get the well name for a given well index
     * 
     * @param wellIndex The well index
     * @return const std::string& The well name
     */
    const std::string& getWellName(int wellIndex) const;
    
    /**
     * @brief Apply the merged matrices to a vector
     * 
     * Computes y -= C^T * (D^-1 * (B * x))
     * 
     * @param x The input vector
     * @param y The output vector (modified in-place)
     */
    void apply(const BVector& x, BVector& y) const;
    
    /**
     * @brief Clear all data and reset the merger
     */
    void clear();
    
private:
    // Initialize the merged matrices
    void initializeMatrices();
    
    // Merged matrices
    OffDiagMatWell mergedB_;
    OffDiagMatWell mergedC_;
    DiagMatWell mergedD_;
    
    // Mapping information
    std::vector<int> wellIndices_;
    std::vector<std::string> wellNames_;
    std::unordered_map<int, int> perforationMap_; // Maps cell index to well index
    
    // Size information
    size_t totalWells_ = 0;
    size_t totalPerforations_ = 0;
    
    // Well data storage
    struct WellData {
        OffDiagMatWell B;
        OffDiagMatWell C;
        DiagMatWell D;
        std::vector<int> cells;
        int index;
        std::string name;
    };
    std::vector<WellData> wellData_;
    
    // Flags
    bool initialized_ = false;
    bool finalized_ = false;
};

} // namespace Opm

#include "WellMatrixMerger_impl.hpp"

#endif // OPM_WELLMATRIXMERGER_HEADER_INCLUDED 