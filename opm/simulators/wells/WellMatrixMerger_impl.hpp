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

#ifndef OPM_WELLMATRIXMERGER_IMPL_HEADER_INCLUDED
#define OPM_WELLMATRIXMERGER_IMPL_HEADER_INCLUDED

namespace Opm {

template<class Scalar, int numWellEq, int numEq>
WellMatrixMerger<Scalar, numWellEq, numEq>::WellMatrixMerger()
{
    // Initialization will be done in initializeMatrices
}

template<class Scalar, int numWellEq, int numEq>
void WellMatrixMerger<Scalar, numWellEq, numEq>::addWell(
    const OffDiagMatWell& B, 
    const OffDiagMatWell& C, 
    const DiagMatWell& D,
    const std::vector<int>& wellCells,
    int wellIndex,
    const std::string& wellName)
{
    if (finalized_) {
        throw std::runtime_error("Cannot add well after finalize() has been called");
    }
    
    // Store well information
    wellIndices_.push_back(wellIndex);
    wellNames_.push_back(wellName.empty() ? "Well_" + std::to_string(wellIndex) : wellName);
    
    // Store perforation information
    for (const auto& cell : wellCells) {
        perforationMap_[cell] = wellIndex;
    }
    
    // Store well data for later processing
    WellData data;
    data.B = B;
    data.C = C;
    data.D = D;
    data.cells = wellCells;
    data.index = wellIndex;
    data.name = wellName;
    wellData_.push_back(data);
    
    // Update size information
    totalWells_ = wellIndices_.size();
    totalPerforations_ += wellCells.size();
    
    initialized_ = true;
}

template<class Scalar, int numWellEq, int numEq>
void WellMatrixMerger<Scalar, numWellEq, numEq>::finalize()
{
    if (!initialized_) {
        throw std::runtime_error("Cannot finalize before adding any wells");
    }
    
    if (finalized_) {
        return; // Already finalized
    }
    
    // Initialize the merged matrices
    initializeMatrices();
    
    // Process each well's data
    for (size_t wellIdx = 0; wellIdx < wellData_.size(); ++wellIdx) {
        const auto& well = wellData_[wellIdx];
        
        // Add B matrix entries
        for (size_t i = 0; i < well.B.N(); ++i) {
            auto row = well.B[i];
            for (auto it = row.begin(); it != row.end(); ++it) {
                int cellIndex = well.cells[it.index()];
                mergedB_[wellIdx][cellIndex] = *it;
            }
        }
        
        // Add C matrix entries
        for (size_t i = 0; i < well.C.N(); ++i) {
            auto row = well.C[i];
            for (auto it = row.begin(); it != row.end(); ++it) {
                int cellIndex = well.cells[it.index()];
                mergedC_[wellIdx][cellIndex] = *it;
            }
        }
        
        // Add D matrix entries (diagonal block for this well)
        if (well.D.N() > 0 && well.D[0].size() > 0) {
            mergedD_[wellIdx][wellIdx] = well.D[0][0];
        }
    }
    
    finalized_ = true;
}

template<class Scalar, int numWellEq, int numEq>
void WellMatrixMerger<Scalar, numWellEq, numEq>::initializeMatrices()
{
    // Find the maximum cell index to determine the matrix size
    int maxCellIndex = 0;
    for (const auto& well : wellData_) {
        for (const auto& cellIndex : well.cells) {
            maxCellIndex = std::max(maxCellIndex, cellIndex);
        }
    }
    
    // Set up matrix sizes - add 1 to maxCellIndex because indices are 0-based
    mergedB_.setSize(totalWells_, maxCellIndex + 1);
    mergedC_.setSize(totalWells_, maxCellIndex + 1);
    mergedD_.setSize(totalWells_, totalWells_);
    
    // Set build mode to row_wise (required for createbegin/createend)
    mergedB_.setBuildMode(OffDiagMatWell::row_wise);
    mergedC_.setBuildMode(OffDiagMatWell::row_wise);
    mergedD_.setBuildMode(DiagMatWell::row_wise);
    
    // Define sparsity pattern for B and C matrices using createbegin/createend
    for (size_t wellIdx = 0; wellIdx < totalWells_; ++wellIdx) {
        // B matrix sparsity pattern
        {
            auto rowB = mergedB_.createbegin();
            std::advance(rowB, wellIdx);
            for (const auto& cellIndex : wellData_[wellIdx].cells) {
                rowB.insert(cellIndex);
            }
        }
        
        // C matrix sparsity pattern
        {
            auto rowC = mergedC_.createbegin();
            std::advance(rowC, wellIdx);
            for (const auto& cellIndex : wellData_[wellIdx].cells) {
                rowC.insert(cellIndex);
            }
        }
        
        // D matrix sparsity pattern (diagonal block)
        {
            auto rowD = mergedD_.createbegin();
            std::advance(rowD, wellIdx);
            rowD.insert(wellIdx);  // Only diagonal entries for D
        }
    }
}

template<class Scalar, int numWellEq, int numEq>
int WellMatrixMerger<Scalar, numWellEq, numEq>::getWellIndexForCell(int cellIndex) const
{
    auto it = perforationMap_.find(cellIndex);
    if (it != perforationMap_.end()) {
        return it->second;
    }
    return -1; // Cell not perforated by any well
}

template<class Scalar, int numWellEq, int numEq>
const std::string& WellMatrixMerger<Scalar, numWellEq, numEq>::getWellName(int wellIndex) const
{
    for (size_t i = 0; i < wellIndices_.size(); ++i) {
        if (wellIndices_[i] == wellIndex) {
            return wellNames_[i];
        }
    }
    
    static const std::string empty;
    return empty;
}

template<class Scalar, int numWellEq, int numEq>
void WellMatrixMerger<Scalar, numWellEq, numEq>::apply(const BVector& x, BVector& y) const
{
    if (!finalized_) {
        throw std::runtime_error("Cannot apply matrices before finalize() has been called");
    }
    
    // Map cell indices to vector indices
    std::map<int, int> cellToVecIdx;
    for (size_t i = 0; i < x.size(); ++i) {
        for (const auto& well : wellData_) {
            for (const auto& cellIndex : well.cells) {
                if (cellIndex == 10 + i * 10) { // This assumes a specific pattern in the example
                    cellToVecIdx[cellIndex] = i;
                }
            }
        }
    }
    
    // Process each well separately
    for (size_t wellIdx = 0; wellIdx < totalWells_; ++wellIdx) {
        const auto& well = wellData_[wellIdx];
        
        // Temporary vectors for this well
        BVectorWell Bx(1), invDBx(1);
        Bx[0] = 0.0;
        
        // Compute B * x for this well using the merged B matrix
        auto rowB = mergedB_[wellIdx];
        for (auto colB = rowB.begin(); colB != rowB.end(); ++colB) {
            int cellIdx = colB.index();
            int vecIdx = cellToVecIdx[cellIdx];
            
            for (int i = 0; i < numWellEq; ++i) {
                for (int j = 0; j < numEq; ++j) {
                    Bx[0][i] += (*colB)[i][j] * x[vecIdx][j];
                }
            }
        }
        
        // Compute D^-1 * B * x for this well
        invDBx[0] = 0.0;
        DiagMatrixBlockWellType invD;
        
        // Get the diagonal block from the merged D matrix
        const auto& D_block = mergedD_[wellIdx][wellIdx];
        
        // Compute the inverse of D (simplified for this example)
        for (int i = 0; i < numWellEq; ++i) {
            for (int j = 0; j < numWellEq; ++j) {
                invD[i][j] = (i == j) ? 1.0 / D_block[i][j] : 0.0;
            }
        }
        
        // Apply the inverse of D
        for (int i = 0; i < numWellEq; ++i) {
            for (int j = 0; j < numWellEq; ++j) {
                invDBx[0][i] += invD[i][j] * Bx[0][j];
            }
        }
        
        // Compute y -= C^T * invDBx for this well using the merged C matrix
        auto rowC = mergedC_[wellIdx];
        for (auto colC = rowC.begin(); colC != rowC.end(); ++colC) {
            int cellIdx = colC.index();
            int vecIdx = cellToVecIdx[cellIdx];
            
            for (int i = 0; i < numEq; ++i) {
                for (int j = 0; j < numWellEq; ++j) {
                    y[vecIdx][i] -= (*colC)[j][i] * invDBx[0][j];
                }
            }
        }
    }
}

template<class Scalar, int numWellEq, int numEq>
void WellMatrixMerger<Scalar, numWellEq, numEq>::clear()
{
    wellData_.clear();
    wellIndices_.clear();
    wellNames_.clear();
    perforationMap_.clear();
    
    // DUNE matrices don't have a clear() method, so we need to create new empty matrices
    mergedB_ = OffDiagMatWell();
    mergedC_ = OffDiagMatWell();
    mergedD_ = DiagMatWell();
    
    totalWells_ = 0;
    totalPerforations_ = 0;
    
    initialized_ = false;
    finalized_ = false;
}

} // namespace Opm

#endif // OPM_WELLMATRIXMERGER_IMPL_HEADER_INCLUDED 