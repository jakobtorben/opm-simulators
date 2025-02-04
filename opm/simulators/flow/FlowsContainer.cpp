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
#include <config.h>
#include <opm/simulators/flow/FlowsContainer.hpp>

#include <opm/input/eclipse/EclipseState/SummaryConfig/SummaryConfig.hpp>
#include <opm/input/eclipse/Schedule/Schedule.hpp>
#include <opm/input/eclipse/Units/UnitSystem.hpp>

#include <opm/material/fluidsystems/BlackOilDefaultIndexTraits.hpp>
#include <opm/material/fluidsystems/BlackOilFluidSystem.hpp>
#include <opm/material/fluidsystems/GenericOilGasWaterFluidSystem.hpp>

#include <opm/output/data/Solution.hpp>

#include <algorithm>
#include <tuple>

namespace {

    template<class Scalar>
    using DataEntry = std::tuple<std::string, Opm::UnitSystem::measure, std::vector<Scalar>&>;

    template<int idx, class Array, class Scalar>
    void addEntry(std::vector<DataEntry<Scalar>>& container,
                  const std::string& name,
                  Opm::UnitSystem::measure measure,
                  Array& flowArray)
    {
        if constexpr (idx >= 0) {  // Only add if index is valid
            container.emplace_back(name, measure, flowArray[idx]);
        }
    }

    template<int idx, class Array, class Scalar>
    void assignToVec(Array& array,
                     const unsigned globalDofIdx,
                     const Scalar value)
    {
        if constexpr (idx != -1) {
            if (!array[idx].empty()) {
                array[idx][globalDofIdx] = value;
            }
        }
    }

    template<int idx, class Array, class Scalar>
    void assignToNnc(Array& array,
                     unsigned nncId,
                     const Scalar value)
    {
        if constexpr (idx != -1) {
            if (!array[idx].indices.empty()) {
                array[idx].indices[nncId] = nncId;
                array[idx].values[nncId] = value;
            }
        }
    }

}

namespace Opm {

template<class FluidSystem>
FlowsContainer<FluidSystem>::
FlowsContainer(const Schedule& schedule,
               const SummaryConfig& summaryConfig)
{
    // Check for any BFLOW[I|J|K] summary keys
    blockFlows_ = summaryConfig.keywords("BFLOW*").size() > 0;

    // Check if FLORES/FLOWS is set in any RPTRST in the schedule
    enableFlores_ = false;  // Used for the output of i+, j+, k+
    enableFloresn_ = false; // Used for the special case of nnc
    enableFlows_ = false;
    enableFlowsn_ = false;

    anyFlores_ = std::any_of(schedule.begin(), schedule.end(),
                             [](const auto& block)
                             {
                                const auto& rstkw = block.rst_config().keywords;
                                return rstkw.find("FLORES") != rstkw.end();
                             });
    anyFlows_ = std::any_of(schedule.begin(), schedule.end(),
                            [](const auto& block)
                            {
                                const auto& rstkw = block.rst_config().keywords;
                                return rstkw.find("FLOWS") != rstkw.end();
                            });
}

template<class FluidSystem>
void FlowsContainer<FluidSystem>::
allocate(const std::size_t bufferSize,
         const unsigned numOutputNnc,
         const bool allocRestart,
         std::map<std::string, int>& rstKeywords)
{
    using Dir = FaceDir::DirEnum;

    // Flows may need to be allocated even when there is no restart due to BFLOW* summary keywords
    if (blockFlows_ ) {
        const std::array<int, 3> phaseIdxs { gasPhaseIdx, oilPhaseIdx, waterPhaseIdx };
        const std::array<int, 3> compIdxs { gasCompIdx, oilCompIdx, waterCompIdx };

        for (unsigned ii = 0; ii < phaseIdxs.size(); ++ii) {
            if (FluidSystem::phaseIsActive(phaseIdxs[ii])) {
                flows_[FaceDir::ToIntersectionIndex(Dir::XPlus)][compIdxs[ii]].resize(bufferSize, 0.0);
                flows_[FaceDir::ToIntersectionIndex(Dir::YPlus)][compIdxs[ii]].resize(bufferSize, 0.0);
                flows_[FaceDir::ToIntersectionIndex(Dir::ZPlus)][compIdxs[ii]].resize(bufferSize, 0.0);
            }
        }
    }

    if (!allocRestart) {
        return ;
    }

    enableFlows_ = false;
    enableFlowsn_ = false;
    const bool rstFlows = (rstKeywords["FLOWS"] > 0);
    if (rstFlows) {
        rstKeywords["FLOWS"] = 0;
        enableFlows_ = true;

        const std::array<int, 3> phaseIdxs = { gasPhaseIdx, oilPhaseIdx, waterPhaseIdx };
        const std::array<int, 3> compIdxs = { gasCompIdx, oilCompIdx, waterCompIdx };
        const auto rstName = std::array { "FLOGASN+", "FLOOILN+", "FLOWATN+" };

        for (unsigned ii = 0; ii < phaseIdxs.size(); ++ii) {
            if (FluidSystem::phaseIsActive(phaseIdxs[ii])) {
                if (!blockFlows_) { // Already allocated if summary vectors requested
                    flows_[FaceDir::ToIntersectionIndex(Dir::XPlus)][compIdxs[ii]].resize(bufferSize, 0.0);
                    flows_[FaceDir::ToIntersectionIndex(Dir::YPlus)][compIdxs[ii]].resize(bufferSize, 0.0);
                    flows_[FaceDir::ToIntersectionIndex(Dir::ZPlus)][compIdxs[ii]].resize(bufferSize, 0.0);
                }

                if (rstKeywords["FLOWS-"] > 0) {
                    flows_[FaceDir::ToIntersectionIndex(Dir::XMinus)][compIdxs[ii]].resize(bufferSize, 0.0);
                    flows_[FaceDir::ToIntersectionIndex(Dir::YMinus)][compIdxs[ii]].resize(bufferSize, 0.0);
                    flows_[FaceDir::ToIntersectionIndex(Dir::ZMinus)][compIdxs[ii]].resize(bufferSize, 0.0);
                }

                if (numOutputNnc > 0) {
                    enableFlowsn_ = true;

                    flowsn_[compIdxs[ii]].name = rstName[ii];
                    flowsn_[compIdxs[ii]].indices.resize(numOutputNnc, -1);
                    flowsn_[compIdxs[ii]].values.resize(numOutputNnc, 0.0);
                }
            }
        }
        if (rstKeywords["FLOWS-"] > 0) {
            rstKeywords["FLOWS-"] = 0;
        }
    }

    enableFlores_ = false;
    enableFloresn_ = false;
    if (rstKeywords["FLORES"] > 0) {
        rstKeywords["FLORES"] = 0;
        enableFlores_ = true;

        const std::array<int, 3> phaseIdxs = { gasPhaseIdx, oilPhaseIdx, waterPhaseIdx };
        const std::array<int, 3> compIdxs = { gasCompIdx, oilCompIdx, waterCompIdx };
        const auto rstName = std::array{ "FLRGASN+", "FLROILN+", "FLRWATN+" };

        for (unsigned ii = 0; ii < phaseIdxs.size(); ++ii) {
            if (FluidSystem::phaseIsActive(phaseIdxs[ii])) {
                flores_[FaceDir::ToIntersectionIndex(Dir::XPlus)][compIdxs[ii]].resize(bufferSize, 0.0);
                flores_[FaceDir::ToIntersectionIndex(Dir::YPlus)][compIdxs[ii]].resize(bufferSize, 0.0);
                flores_[FaceDir::ToIntersectionIndex(Dir::ZPlus)][compIdxs[ii]].resize(bufferSize, 0.0);

                if (rstKeywords["FLORES-"] > 0) {
                    flores_[FaceDir::ToIntersectionIndex(Dir::XMinus)][compIdxs[ii]].resize(bufferSize, 0.0);
                    flores_[FaceDir::ToIntersectionIndex(Dir::YMinus)][compIdxs[ii]].resize(bufferSize, 0.0);
                    flores_[FaceDir::ToIntersectionIndex(Dir::ZMinus)][compIdxs[ii]].resize(bufferSize, 0.0);
                }

                if (numOutputNnc > 0) {
                    enableFloresn_ = true;

                    floresn_[compIdxs[ii]].name = rstName[ii];
                    floresn_[compIdxs[ii]].indices.resize(numOutputNnc, -1);
                    floresn_[compIdxs[ii]].values.resize(numOutputNnc, 0.0);
                }
            }
        }
        if (rstKeywords["FLORES-"] > 0) {
            rstKeywords["FLORES-"] = 0;
        }
    }
}

template<class FluidSystem>
void FlowsContainer<FluidSystem>::
assignFlows(const unsigned globalDofIdx,
            const int faceId,
            const unsigned nncId,
            const Scalar gas,
            const Scalar oil,
            const Scalar water)
{
    if (faceId >= 0) {
        assignToVec<gasCompIdx>(this->flows_[faceId], globalDofIdx, gas);
        assignToVec<oilCompIdx>(this->flows_[faceId], globalDofIdx, oil);
        assignToVec<waterCompIdx>(this->flows_[faceId], globalDofIdx, water);
    }
    else if (faceId == -2) {
        assignToNnc<gasCompIdx>(this->flowsn_, nncId, gas);
        assignToNnc<oilCompIdx>(this->flowsn_, nncId, oil);
        assignToNnc<waterCompIdx>(this->flowsn_, nncId, water);
    }
}

template<class FluidSystem>
void FlowsContainer<FluidSystem>::
outputRestart(data::Solution& sol)
{
    auto doInsert = [&sol](DataEntry<Scalar>& entry,
                           const data::TargetType   target)
    {
        if (!std::get<2>(entry).empty()) {
            sol.insert(std::get<std::string>(entry),
                       std::get<UnitSystem::measure>(entry),
                       std::move(std::get<2>(entry)),
                       target);
        }
    };

    using Dir = FaceDir::DirEnum;
    std::vector<DataEntry<Scalar>> floresSolutionVector;
    addEntry<gasCompIdx>(floresSolutionVector, "FLRGASI+", UnitSystem::measure::rate,   flores_[FaceDir::ToIntersectionIndex(Dir::XPlus)]);
    addEntry<gasCompIdx>(floresSolutionVector, "FLRGASJ+", UnitSystem::measure::rate,   flores_[FaceDir::ToIntersectionIndex(Dir::YPlus)]);
    addEntry<gasCompIdx>(floresSolutionVector, "FLRGASK+", UnitSystem::measure::rate,   flores_[FaceDir::ToIntersectionIndex(Dir::ZPlus)]);
    addEntry<oilCompIdx>(floresSolutionVector, "FLROILI+", UnitSystem::measure::rate,   flores_[FaceDir::ToIntersectionIndex(Dir::XPlus)]);
    addEntry<oilCompIdx>(floresSolutionVector, "FLROILJ+", UnitSystem::measure::rate,   flores_[FaceDir::ToIntersectionIndex(Dir::YPlus)]);
    addEntry<oilCompIdx>(floresSolutionVector, "FLROILK+", UnitSystem::measure::rate,   flores_[FaceDir::ToIntersectionIndex(Dir::ZPlus)]);
    addEntry<waterCompIdx>(floresSolutionVector, "FLRWATI+", UnitSystem::measure::rate, flores_[FaceDir::ToIntersectionIndex(Dir::XPlus)]);
    addEntry<waterCompIdx>(floresSolutionVector, "FLRWATJ+", UnitSystem::measure::rate, flores_[FaceDir::ToIntersectionIndex(Dir::YPlus)]);
    addEntry<waterCompIdx>(floresSolutionVector, "FLRWATK+", UnitSystem::measure::rate, flores_[FaceDir::ToIntersectionIndex(Dir::ZPlus)]);

    std::vector<DataEntry<Scalar>> flowsSolutionVector;
    addEntry<gasCompIdx>(flowsSolutionVector, "FLOGASI+", UnitSystem::measure::gas_surface_rate,      flows_[FaceDir::ToIntersectionIndex(Dir::XPlus)]);
    addEntry<gasCompIdx>(flowsSolutionVector, "FLOGASJ+", UnitSystem::measure::gas_surface_rate,      flows_[FaceDir::ToIntersectionIndex(Dir::YPlus)]);
    addEntry<gasCompIdx>(flowsSolutionVector, "FLOGASK+", UnitSystem::measure::gas_surface_rate,      flows_[FaceDir::ToIntersectionIndex(Dir::ZPlus)]);
    addEntry<oilCompIdx>(flowsSolutionVector, "FLOOILI+", UnitSystem::measure::liquid_surface_rate,   flows_[FaceDir::ToIntersectionIndex(Dir::XPlus)]);
    addEntry<oilCompIdx>(flowsSolutionVector, "FLOOILJ+", UnitSystem::measure::liquid_surface_rate,   flows_[FaceDir::ToIntersectionIndex(Dir::YPlus)]);
    addEntry<oilCompIdx>(flowsSolutionVector, "FLOOILK+", UnitSystem::measure::liquid_surface_rate,   flows_[FaceDir::ToIntersectionIndex(Dir::ZPlus)]);
    addEntry<waterCompIdx>(flowsSolutionVector, "FLOWATI+", UnitSystem::measure::liquid_surface_rate, flows_[FaceDir::ToIntersectionIndex(Dir::XPlus)]);
    addEntry<waterCompIdx>(flowsSolutionVector, "FLOWATJ+", UnitSystem::measure::liquid_surface_rate, flows_[FaceDir::ToIntersectionIndex(Dir::YPlus)]);
    addEntry<waterCompIdx>(flowsSolutionVector, "FLOWATK+", UnitSystem::measure::liquid_surface_rate, flows_[FaceDir::ToIntersectionIndex(Dir::ZPlus)]);
    addEntry<gasCompIdx>(flowsSolutionVector, "FLOGASI-", UnitSystem::measure::gas_surface_rate,      flows_[FaceDir::ToIntersectionIndex(Dir::XMinus)]);
    addEntry<gasCompIdx>(flowsSolutionVector, "FLOGASJ-", UnitSystem::measure::gas_surface_rate,      flows_[FaceDir::ToIntersectionIndex(Dir::YMinus)]);
    addEntry<gasCompIdx>(flowsSolutionVector, "FLOGASK-", UnitSystem::measure::gas_surface_rate,      flows_[FaceDir::ToIntersectionIndex(Dir::ZMinus)]);
    addEntry<oilCompIdx>(flowsSolutionVector, "FLOOILI-", UnitSystem::measure::liquid_surface_rate,   flows_[FaceDir::ToIntersectionIndex(Dir::XMinus)]);
    addEntry<oilCompIdx>(flowsSolutionVector, "FLOOILJ-", UnitSystem::measure::liquid_surface_rate,   flows_[FaceDir::ToIntersectionIndex(Dir::YMinus)]);
    addEntry<oilCompIdx>(flowsSolutionVector, "FLOOILK-", UnitSystem::measure::liquid_surface_rate,   flows_[FaceDir::ToIntersectionIndex(Dir::ZMinus)]);
    addEntry<waterCompIdx>(flowsSolutionVector, "FLOWATI-", UnitSystem::measure::liquid_surface_rate, flows_[FaceDir::ToIntersectionIndex(Dir::XMinus)]);
    addEntry<waterCompIdx>(flowsSolutionVector, "FLOWATJ-", UnitSystem::measure::liquid_surface_rate, flows_[FaceDir::ToIntersectionIndex(Dir::YMinus)]);
    addEntry<waterCompIdx>(flowsSolutionVector, "FLOWATK-", UnitSystem::measure::liquid_surface_rate, flows_[FaceDir::ToIntersectionIndex(Dir::ZMinus)]);
    addEntry<gasCompIdx>(flowsSolutionVector, "FLRGASI-", UnitSystem::measure::rate,                  flores_[FaceDir::ToIntersectionIndex(Dir::XMinus)]);
    addEntry<gasCompIdx>(flowsSolutionVector, "FLRGASJ-", UnitSystem::measure::rate,                  flores_[FaceDir::ToIntersectionIndex(Dir::YMinus)]);
    addEntry<gasCompIdx>(flowsSolutionVector, "FLRGASK-", UnitSystem::measure::rate,                  flores_[FaceDir::ToIntersectionIndex(Dir::ZMinus)]);
    addEntry<oilCompIdx>(flowsSolutionVector, "FLROILI-", UnitSystem::measure::rate,                  flores_[FaceDir::ToIntersectionIndex(Dir::XMinus)]);
    addEntry<oilCompIdx>(flowsSolutionVector, "FLROILJ-", UnitSystem::measure::rate,                  flores_[FaceDir::ToIntersectionIndex(Dir::YMinus)]);
    addEntry<oilCompIdx>(flowsSolutionVector, "FLROILK-", UnitSystem::measure::rate,                  flores_[FaceDir::ToIntersectionIndex(Dir::ZMinus)]);
    addEntry<waterCompIdx>(flowsSolutionVector, "FLRWATI-", UnitSystem::measure::rate,                flores_[FaceDir::ToIntersectionIndex(Dir::XMinus)]);
    addEntry<waterCompIdx>(flowsSolutionVector, "FLRWATJ-", UnitSystem::measure::rate,                flores_[FaceDir::ToIntersectionIndex(Dir::YMinus)]);
    addEntry<waterCompIdx>(flowsSolutionVector, "FLRWATK-", UnitSystem::measure::rate,                flores_[FaceDir::ToIntersectionIndex(Dir::ZMinus)]);

    std::for_each(floresSolutionVector.begin(), floresSolutionVector.end(),
                  [doInsert](auto& array)
                  { doInsert(array, data::TargetType::RESTART_SOLUTION); });

    if (this->enableFlows_) {
        std::for_each(flowsSolutionVector.begin(), flowsSolutionVector.end(),
                      [doInsert](auto& array)
                      { doInsert(array, data::TargetType::RESTART_SOLUTION); });
    }
}

template<class T> using FS = BlackOilFluidSystem<T,BlackOilDefaultIndexTraits>;

#define INSTANTIATE_TYPE(T) \
    template class FlowsContainer<FS<T>>;

INSTANTIATE_TYPE(double)

#if FLOW_INSTANTIATE_FLOAT
INSTANTIATE_TYPE(float)
#endif

#define INSTANTIATE_COMP_THREEPHASE(NUM) \
    template<class T> using FS##NUM = GenericOilGasWaterFluidSystem<T, NUM, true>; \
    template class FlowsContainer<FS##NUM<double>>;

#define INSTANTIATE_COMP_TWOPHASE(NUM) \
    template<class T> using GFS##NUM = GenericOilGasWaterFluidSystem<T, NUM, false>; \
    template class FlowsContainer<GFS##NUM<double>>;

#define INSTANTIATE_COMP(NUM) \
    INSTANTIATE_COMP_THREEPHASE(NUM) \
    INSTANTIATE_COMP_TWOPHASE(NUM)

INSTANTIATE_COMP_THREEPHASE(0)  // \Note: to register the parameter ForceDisableFluidInPlaceOutput
INSTANTIATE_COMP(2)
INSTANTIATE_COMP(3)
INSTANTIATE_COMP(4)
INSTANTIATE_COMP(5)
INSTANTIATE_COMP(6)
INSTANTIATE_COMP(7)

} // namespace Opm
