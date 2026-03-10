/*
  Copyright 2021 Total SE

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

#ifndef OPM_SUBDOMAIN_HEADER_INCLUDED
#define OPM_SUBDOMAIN_HEADER_INCLUDED

#include <opm/grid/common/SubGridPart.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <utility>
#include <vector>

namespace Opm
{
    //! \brief Solver approach for NLDD.
    enum class DomainSolveApproach {
        Jacobi,
        GaussSeidel
    };

    //! \brief How Gauss-Seidel treats overlap-cell updates after a successful local solve.
    enum class GaussSeidelOverlapWriteback {
        Restricted,
        Unrestricted
    };

    //! \brief Measure to use for domain ordering.
    enum class DomainOrderingMeasure {
        AveragePressure,
        MaxPressure,
        Residual
    };

    inline DomainOrderingMeasure domainOrderingMeasureFromString(const std::string_view measure)
    {
        if (measure == "residual") {
            return DomainOrderingMeasure::Residual;
        } else if (measure == "maxpressure") {
            return DomainOrderingMeasure::MaxPressure;
        } else if (measure == "averagepressure") {
            return DomainOrderingMeasure::AveragePressure;
        } else {
            throw std::runtime_error(fmt::format(fmt::runtime("Invalid domain ordering '{}' specified"), measure));
        }
    }

    inline GaussSeidelOverlapWriteback gaussSeidelOverlapWritebackFromString(const std::string_view mode)
    {
        if (mode == "restricted") {
            return GaussSeidelOverlapWriteback::Restricted;
        } else if (mode == "unrestricted") {
            return GaussSeidelOverlapWriteback::Unrestricted;
        } else {
            throw std::runtime_error(fmt::format(fmt::runtime("Invalid Gauss-Seidel overlap writeback '{}' specified"), mode));
        }
    }

    /// Representing a part of a grid, in a way suitable for performing
    /// local solves.
    struct SubDomainIndices
    {
        // The index of a subdomain is arbitrary, but can be used by the
        // solvers to keep track of well locations etc.
        int index;
        // All cells in this subdomain (interior + overlap), sorted.
        // Used for assembly, matrix extraction, and linear solve.
        std::vector<int> cells;
        // Interior cells only (the original partition cells), sorted.
        // Used for convergence checking, solution updates, and well assignment.
        std::vector<int> interior_cells;
        // Flag for each cell of the current MPI rank, true if the cell is part
        // of the subdomain's interior. If empty, assumed to be all true.
        std::vector<bool> interior;
        // Flag indicating if this domain should be skipped during solves
        bool skip;
        // Construct with overlap cells. cells = interior_cells + overlap_cells (sorted).
        SubDomainIndices(const int i, std::vector<int>&& int_cells,
                         std::vector<int>&& ovlp_cells,
                         std::vector<bool>&& in, bool s)
            : index(i)
            , interior_cells(std::move(int_cells))
            , interior(std::move(in))
            , skip(s)
        {
            cells = interior_cells;
            if (!ovlp_cells.empty()) {
                cells.insert(cells.end(), ovlp_cells.begin(), ovlp_cells.end());
                std::sort(cells.begin(), cells.end());
            }
        }
    };

    /// Representing a part of a grid, in a way suitable for performing
    /// local solves.
    template <class Grid>
    struct SubDomain : public SubDomainIndices
    {
        Dune::SubGridPart<Grid> view;
        // Constructor that moves from its argument.
        SubDomain(const int i, std::vector<int>&& c, std::vector<int>&& ovlp,
                  std::vector<bool>&& in, Dune::SubGridPart<Grid>&& v, bool s)
            : SubDomainIndices(i, std::move(c), std::move(ovlp), std::move(in), s)
            , view(std::move(v))
        {}
    };

} // namespace Opm


#endif // OPM_SUBDOMAIN_HEADER_INCLUDED
