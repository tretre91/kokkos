// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <Kokkos_Macros.hpp>
#include "impl/Kokkos_AnalyzePolicy.hpp"
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

#include <TestDefaultDeviceType_Category.hpp>

namespace Test {

TEST(defaultdevicetype, development_test) {
  Kokkos::TeamPolicy<Kokkos::Rank<3>, Kokkos::HIP> policy(
      Kokkos::HIP(), {4, 2, 3}, Kokkos::AUTO);
  Kokkos::parallel_for(
      "test", policy, KOKKOS_LAMBDA(decltype(policy)::member_type team) {
        auto [x, y, z] = team.league_ranks();
        Kokkos::printf("%d = (%d,%d,%d)\n", team.league_rank(), x, y, z);
      });
}

}  // namespace Test
