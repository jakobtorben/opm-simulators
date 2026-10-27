/*
  Copyright 2026 SINTEF AS.

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

#include <config.h>
#define BOOST_TEST_MODULE ConvergenceMonitorTest
#include <boost/test/unit_test.hpp>

#include <opm/simulators/flow/BlackoilModelConvergenceMonitor.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace {

using Monitor = Opm::BlackoilModelConvergenceMonitor<double>;
using Params = Monitor::MonitorParams;
using Distance = Params::Distance;

// Sequence of scaled residuals with the monitor parameters and the expected
// monitor state for each iteration.
struct GoldenCase
{
    std::string name;
    Distance distance;
    int memory;
    int target;
    int buffer;
    int margin;
    double thetaFast;
    double thetaSlow;
    std::vector<double> scaledResidual;
    std::vector<double> dist;
    std::vector<double> theta;
    std::vector<double> thetaTarget;
    std::vector<int> oscillating;
    std::vector<int> status;
    std::vector<int> numViolations;
};

// Reference values, see the header of convergencemonitor_golden.inc.
const std::vector<GoldenCase> goldenCases = {
#include "convergencemonitor_golden.inc"
};

Params makeParams(const GoldenCase& c)
{
    Params p;
    p.enabled_ = true;
    p.distance_ = c.distance;
    p.memory_ = c.memory;
    p.target_iterations_ = c.target;
    p.iteration_buffer_ = c.buffer;
    p.iteration_margin_ = c.margin;
    p.theta_fast_ = c.thetaFast;
    p.theta_slow_ = c.thetaSlow;
    p.violations_cut_ = 3;
    return p;
}

bool close(const double a, const double b)
{
    return std::abs(a - b) <= 1.0e-12 * std::max({1.0, std::abs(a), std::abs(b)});
}

} // Anonymous namespace

BOOST_AUTO_TEST_CASE(MatchesReference)
{
    BOOST_REQUIRE(!goldenCases.empty());
    for (const auto& c : goldenCases) {
        BOOST_TEST_CONTEXT("Case " << c.name) {
            auto monitor = Monitor { makeParams(c) };
            for (std::size_t j = 0; j < c.scaledResidual.size(); ++j) {
                BOOST_TEST_CONTEXT("Iteration " << j) {
                    const auto& s = monitor.update(c.scaledResidual[j], static_cast<int>(j));
                    BOOST_CHECK(close(s.distance, c.dist[j]));
                    BOOST_CHECK(close(s.theta, c.theta[j]));
                    BOOST_CHECK(close(s.thetaTarget, c.thetaTarget[j]));
                    BOOST_CHECK_EQUAL(s.oscillating, c.oscillating[j] != 0);
                    BOOST_CHECK_EQUAL(s.status, c.status[j]);
                    BOOST_CHECK_EQUAL(s.numViolations, c.numViolations[j]);
                }
            }
        }
    }
}

// Stagnating iterations are cut after violations_cut_ + 1 violations.
BOOST_AUTO_TEST_CASE(CutStagnating)
{
    Params p;
    p.target_iterations_ = 6;
    p.violations_cut_ = 1;
    auto monitor = Monitor { p };

    const std::vector<double> residuals = { 1.0e3, 9.0e2, 8.0e2, 7.0e2 };
    int cutIteration = -1;
    for (std::size_t j = 0; j < residuals.size(); ++j) {
        monitor.update(residuals[j], static_cast<int>(j));
        if (monitor.shouldCut()) {
            cutIteration = static_cast<int>(j);
            break;
        }
    }
    BOOST_CHECK_EQUAL(cutIteration, 2);
    BOOST_CHECK_EQUAL(monitor.state().status, -1);
    BOOST_CHECK_EQUAL(monitor.state().numViolations, 2);
}

BOOST_AUTO_TEST_CASE(FirstIterationAndReset)
{
    Params p;
    p.violations_cut_ = 0;
    auto monitor = Monitor { p };

    // First iteration: no contraction estimate, ok, no violations.
    auto s = monitor.update(1.0e3, 0);
    BOOST_CHECK_EQUAL(s.theta, 0.0);
    BOOST_CHECK_EQUAL(s.thetaTarget, 0.0);
    BOOST_CHECK_EQUAL(s.status, 0);
    BOOST_CHECK_EQUAL(s.numViolations, 0);
    BOOST_CHECK(!monitor.shouldCut());

    // Diverging iteration is bad.
    s = monitor.update(2.0e3, 1);
    BOOST_CHECK_CLOSE(s.theta, 2.0, 1.0e-10);
    BOOST_CHECK_EQUAL(s.status, -1);
    BOOST_CHECK(monitor.shouldCut());

    // Iteration zero starts a new time step.
    s = monitor.update(5.0e2, 0);
    BOOST_CHECK_CLOSE(s.distance, 5.0e2, 1.0e-10);
    BOOST_CHECK_EQUAL(s.theta, 0.0);
    BOOST_CHECK_EQUAL(s.numViolations, 0);
    BOOST_CHECK(!monitor.shouldCut());

    // Converged iterations are good, the log distance is one.
    p.distance_ = Distance::Log;
    auto logMonitor = Monitor { p };
    logMonitor.update(1.0e2, 0);
    s = logMonitor.update(0.5, 1);
    BOOST_CHECK_CLOSE(s.distance, 1.0, 1.0e-10);
    BOOST_CHECK_CLOSE(s.theta, 1.0 / 3.0, 1.0e-10);
    BOOST_CHECK_EQUAL(s.status, 1);
}
