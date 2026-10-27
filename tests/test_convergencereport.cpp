/*
  Copyright 2018 SINTEF Digital, Mathematics and Cybernetics.
  Copyright 2018 Equinor.

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
#define BOOST_TEST_MODULE ConvergenceReportTest
#include <boost/test/unit_test.hpp>

#include <opm/simulators/timestepping/ConvergenceReport.hpp>

using CR = Opm::ConvergenceReport;

BOOST_AUTO_TEST_CASE(DefaultConstructor)
{
    Opm::ConvergenceReport s;
    BOOST_CHECK(s.converged());
    BOOST_CHECK(!s.reservoirFailed());
    BOOST_CHECK(!s.wellFailed());
    BOOST_CHECK(s.severityOfWorstFailure() == CR::Severity::None);
}

BOOST_AUTO_TEST_CASE(Failures)
{
    Opm::ConvergenceReport s1;
    s1.setReservoirFailed({CR::ReservoirFailure::Type::Cnv, CR::Severity::Normal, 2});
    {
        BOOST_CHECK(!s1.converged());
        BOOST_CHECK(s1.reservoirFailed());
        BOOST_CHECK(!s1.wellFailed());
        BOOST_REQUIRE(s1.reservoirFailures().size() == 1);
        const auto f = s1.reservoirFailures()[0];
        BOOST_CHECK(f.type() == CR::ReservoirFailure::Type::Cnv);
        BOOST_CHECK(f.severity() == CR::Severity::Normal);
        BOOST_CHECK(f.phase() == 2);
        BOOST_CHECK(s1.wellFailures().empty());
        BOOST_CHECK(s1.severityOfWorstFailure() == CR::Severity::Normal);
    }

    Opm::ConvergenceReport s2;
    s2.setWellFailed({CR::WellFailure::Type::ControlTHP, CR::Severity::Normal, -1, "PRODUCER-123"});
    s2.setWellFailed({CR::WellFailure::Type::MassBalance, CR::Severity::TooLarge, 2, "INJECTOR-XYZ"});
    {
        BOOST_CHECK(!s2.converged());
        BOOST_CHECK(!s2.reservoirFailed());
        BOOST_CHECK(s2.wellFailed());
        BOOST_CHECK(s2.reservoirFailures().empty());
        BOOST_REQUIRE(s2.wellFailures().size() == 2);
        const auto f0 = s2.wellFailures()[0];
        BOOST_CHECK(f0.type() == CR::WellFailure::Type::ControlTHP);
        BOOST_CHECK(f0.severity() == CR::Severity::Normal);
        BOOST_CHECK(f0.phase() == -1);
        BOOST_CHECK(f0.wellName() == "PRODUCER-123");
        const auto f1 = s2.wellFailures()[1];
        BOOST_CHECK(f1.type() == CR::WellFailure::Type::MassBalance);
        BOOST_CHECK(f1.severity() == CR::Severity::TooLarge);
        BOOST_CHECK(f1.phase() == 2);
        BOOST_CHECK(f1.wellName() == "INJECTOR-XYZ");
        BOOST_CHECK(s2.severityOfWorstFailure() == CR::Severity::TooLarge);
    }

    s1 += s2;
    {
        BOOST_CHECK(!s1.converged());
        BOOST_CHECK(s1.reservoirFailed());
        BOOST_CHECK(s1.wellFailed());
        BOOST_REQUIRE(s1.reservoirFailures().size() == 1);
        const auto f = s1.reservoirFailures()[0];
        BOOST_CHECK(f.type() == CR::ReservoirFailure::Type::Cnv);
        BOOST_CHECK(f.severity() == CR::Severity::Normal);
        BOOST_CHECK(f.phase() == 2);
        BOOST_REQUIRE(s1.wellFailures().size() == 2);
        const auto f0 = s1.wellFailures()[0];
        BOOST_CHECK(f0.type() == CR::WellFailure::Type::ControlTHP);
        BOOST_CHECK(f0.severity() == CR::Severity::Normal);
        BOOST_CHECK(f0.phase() == -1);
        BOOST_CHECK(f0.wellName() == "PRODUCER-123");
        const auto f1 = s1.wellFailures()[1];
        BOOST_CHECK(f1.type() == CR::WellFailure::Type::MassBalance);
        BOOST_CHECK(f1.severity() == CR::Severity::TooLarge);
        BOOST_CHECK(f1.phase() == 2);
        BOOST_CHECK(f1.wellName() == "INJECTOR-XYZ");
        BOOST_CHECK(s1.severityOfWorstFailure() == CR::Severity::TooLarge);
    }

    s1.clear();
    {
        BOOST_CHECK(s1.converged());
        BOOST_CHECK(!s1.reservoirFailed());
        BOOST_CHECK(!s1.wellFailed());
        BOOST_CHECK(s1.severityOfWorstFailure() == CR::Severity::None);
    }

    s1 += s2;
    {
        BOOST_CHECK(!s1.converged());
        BOOST_CHECK(!s1.reservoirFailed());
        BOOST_CHECK(s1.wellFailed());
        BOOST_CHECK(s1.reservoirFailures().empty());
        BOOST_REQUIRE(s1.wellFailures().size() == 2);
        const auto f0 = s1.wellFailures()[0];
        BOOST_CHECK(f0.type() == CR::WellFailure::Type::ControlTHP);
        BOOST_CHECK(f0.severity() == CR::Severity::Normal);
        BOOST_CHECK(f0.phase() == -1);
        BOOST_CHECK(f0.wellName() == "PRODUCER-123");
        const auto f1 = s1.wellFailures()[1];
        BOOST_CHECK(f1.type() == CR::WellFailure::Type::MassBalance);
        BOOST_CHECK(f1.severity() == CR::Severity::TooLarge);
        BOOST_CHECK(f1.phase() == 2);
        BOOST_CHECK(f1.wellName() == "INJECTOR-XYZ");
        BOOST_CHECK(s1.severityOfWorstFailure() == CR::Severity::TooLarge);
    }
}

BOOST_AUTO_TEST_CASE(ScaledResiduals)
{
    CR s1;
    BOOST_CHECK_EQUAL(s1.maxScaledResidual(), 0.0);

    s1.setScaledReservoirResidual(10.0, CR::ReservoirFailure::Type::Cnv, 1);
    s1.setScaledReservoirResidual(2.0, CR::ReservoirFailure::Type::MassBalance, 0);
    BOOST_CHECK_EQUAL(s1.scaledReservoirResidual().value, 10.0);
    BOOST_CHECK(s1.scaledReservoirResidual().type == CR::ReservoirFailure::Type::Cnv);
    BOOST_CHECK_EQUAL(s1.scaledReservoirResidual().phase, 1);

    CR s2;
    s2.setScaledWellResidual(5.0, CR::WellFailure::Type::MassBalance, "PROD1");
    s2.setScaledWellResidual(20.0, CR::WellFailure::Type::ControlBHP, "PROD2");
    s2.setScaledWellResidual(1.0, CR::WellFailure::Type::ControlRate, "INJ1");
    BOOST_CHECK_EQUAL(s2.scaledWellResidual().value, 20.0);
    BOOST_CHECK_EQUAL(s2.scaledWellResidual().wellName, "PROD2");
    BOOST_CHECK_EQUAL(s2.maxScaledResidual(), 20.0);

    // Combined reports keep the largest values.
    s1 += s2;
    BOOST_CHECK_EQUAL(s1.scaledReservoirResidual().value, 10.0);
    BOOST_CHECK_EQUAL(s1.scaledWellResidual().value, 20.0);
    BOOST_CHECK(s1.scaledWellResidual().type == CR::WellFailure::Type::ControlBHP);
    BOOST_CHECK_EQUAL(s1.maxScaledResidual(), 20.0);

    // The monitor state is only carried by the report that has one.
    CR s3;
    s3.setConvergenceMonitorState({3.0, 0.5, 0.25, false, -1, 2});
    s1 += s3;
    BOOST_CHECK_EQUAL(s1.convergenceMonitorState().distance, 3.0);
    BOOST_CHECK_EQUAL(s1.convergenceMonitorState().numViolations, 2);
    s1 += s2;
    BOOST_CHECK_EQUAL(s1.convergenceMonitorState().status, -1);
}
