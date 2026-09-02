// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_MD_TEAM_HPP
#define KOKKOS_MD_TEAM_HPP

#include <Kokkos_Macros.hpp>
#include <Kokkos_Array.hpp>

namespace Kokkos {
namespace Impl {

template <unsigned rank, class TeamMember>
class MDTeamMember : public TeamMember {
 public:
  using execution_space      = TeamMember::execution_space;
  using scratch_memory_space = execution_space::scratch_memory_space;
  using team_handle          = MDTeamMember<rank, TeamMember>;

 private:
  using indices_type = Kokkos::Array<int, rank>;
  indices_type m_league_ranks;
  indices_type m_league_sizes;

 public:
  using TeamMember::league_rank;
  using TeamMember::league_size;

  KOKKOS_INLINE_FUNCTION const indices_type& league_ranks() const {
    return m_league_ranks;
  }
  KOKKOS_INLINE_FUNCTION int league_rank(int dim) const {
    return m_league_ranks[dim];
  }
  KOKKOS_INLINE_FUNCTION const indices_type& league_sizes() const {
    return m_league_sizes;
  }
  KOKKOS_INLINE_FUNCTION int league_size(int dim) const {
    return m_league_sizes[dim];
  }

  template <class... Args>
  KOKKOS_INLINE_FUNCTION MDTeamMember(const indices_type& arg_league_ranks,
                                      const indices_type& arg_league_sizes,
                                      Args&&... args)
      : TeamMember(std::forward<Args>(args)...),
        m_league_ranks(arg_league_ranks),
        m_league_sizes(arg_league_sizes) {}
};

template <typename iType, unsigned rank, typename TeamMember>
struct TeamThreadRangeBoundariesStruct<iType, MDTeamMember<rank, TeamMember>>
    : public TeamThreadRangeBoundariesStruct<iType, TeamMember> {
  using base_t = TeamThreadRangeBoundariesStruct<iType, TeamMember>;
  using base_t::base_t;
};

template <typename iType, unsigned rank, typename TeamMember>
struct TeamVectorRangeBoundariesStruct<iType, MDTeamMember<rank, TeamMember>>
    : public TeamVectorRangeBoundariesStruct<iType, TeamMember> {
  using base_t = TeamVectorRangeBoundariesStruct<iType, TeamMember>;
  using base_t::base_t;
};

template <typename iType, unsigned rank, typename TeamMember>
struct ThreadVectorRangeBoundariesStruct<iType, MDTeamMember<rank, TeamMember>>
    : public ThreadVectorRangeBoundariesStruct<iType, TeamMember> {
  using base_t = ThreadVectorRangeBoundariesStruct<iType, TeamMember>;
  using base_t::base_t;
};

}  // namespace Impl
}  // namespace Kokkos

#endif
