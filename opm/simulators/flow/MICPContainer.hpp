// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 2 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
  Consult the COPYING file in the top-level source directory of this
  module for the precise wording of the license and the list of
  copyright holders.
*/
/*!
 * \file
 * \copydoc Opm::OutputBlackOilModule
 */
#ifndef OPM_MICP_CONTAINER_HPP
#define OPM_MICP_CONTAINER_HPP

#include <vector>

namespace Opm {

namespace data { class Solution; }

template<class Scalar>
class MICPContainer
{
    using ScalarBuffer = std::vector<Scalar>;

public:
    void allocate(const unsigned bufferSize);

    void outputRestart(data::Solution& sol);

    bool allocated() const
    { return allocated_; }

    Scalar getMicrobialConcentration(unsigned elemIdx) const
    {
        if (cMicrobes_.size() > elemIdx)
            return cMicrobes_[elemIdx];

        return 0;
    }

    Scalar getOxygenConcentration(unsigned elemIdx) const
    {
        if (cOxygen_.size() > elemIdx)
            return cOxygen_[elemIdx];

        return 0;
    }

    Scalar getUreaConcentration(unsigned elemIdx) const
    {
        if (cUrea_.size() > elemIdx)
            return cUrea_[elemIdx];

        return 0;
    }

    Scalar getBiofilmConcentration(unsigned elemIdx) const
    {
        if (cBiofilm_.size() > elemIdx)
            return cBiofilm_[elemIdx];

        return 0;
    }

    Scalar getCalciteConcentration(unsigned elemIdx) const
    {
        if (cCalcite_.size() > elemIdx)
            return cCalcite_[elemIdx];

        return 0;
    }

    bool allocated_ = false;
    ScalarBuffer cMicrobes_;
    ScalarBuffer cOxygen_;
    ScalarBuffer cUrea_;
    ScalarBuffer cBiofilm_;
    ScalarBuffer cCalcite_;
};

} // namespace Opm

#endif // OPM_MICP_CONTAINER_HPP
