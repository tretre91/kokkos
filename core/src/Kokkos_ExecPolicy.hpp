// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <concepts>
#include "Kokkos_Array.hpp"
#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif
#ifndef KOKKOS_EXECPOLICY_HPP
#define KOKKOS_EXECPOLICY_HPP

#include <Kokkos_Core_fwd.hpp>
#include <impl/Kokkos_Traits.hpp>
#include <impl/Kokkos_Error.hpp>
#include <impl/Kokkos_AnalyzePolicy.hpp>
#include <Kokkos_BitManipulation.hpp>
#include <Kokkos_Concepts.hpp>
#include <Kokkos_TypeInfo.hpp>
#ifndef KOKKOS_ENABLE_IMPL_TYPEINFO
#include <typeinfo>
#endif
#include <limits>
#include <sstream>
#include <type_traits>

//----------------------------------------------------------------------------

namespace Kokkos {

struct ParallelForTag {};
struct ParallelScanTag {};
struct ParallelReduceTag {};

struct ChunkSize {
  int value;
  explicit ChunkSize(int value_) : value(value_) {}
};

template <typename... Properties>
class RangePolicy;

namespace Impl {

/** \brief Handle for thread-level parallelism within a team.
 *
 *  Use with RangePolicy to parallelize within a thread using vector resources
 *  (ThreadVectorRange semantics).
 *  The concept is Kokkos::Experimental::ThreadHandle.
 */
template <class TeamMemberType>
struct ThreadHandleType {
  TeamMemberType const& team_member;
  using member_type     = TeamMemberType;
  using execution_space = typename TeamMemberType::execution_space;
  using thread_handle   = ThreadHandleType;

  KOKKOS_INLINE_FUNCTION
  constexpr ThreadHandleType(TeamMemberType const& m) : team_member(m) {}

  KOKKOS_INLINE_FUNCTION
  int team_rank() const { return team_member.team_rank(); }

  KOKKOS_INLINE_FUNCTION
  int team_size() const { return team_member.team_size(); }
};

// Private tag that can be used to make a copy of another execution policy
// and set the underlying execution space instance.
// It does NOT perform any sanity check.
// For now, it is used in Kokkos::Experimental::Graph.
struct PolicyUpdate {};

template <typename T, typename... Properties>
class ImplRangePolicy;

// Specialization of RangePolicy for defining work over a range of an integral
// type, split up among all resources of an execution space.
template <ExecutionSpace ExecSpace, class... Properties>
class ImplRangePolicy<ExecSpace, Properties...>
    : public Impl::PolicyTraits<Properties...> {
 public:
  using traits = Impl::PolicyTraits<Properties...>;
  static_assert(std::same_as<typename traits::execution_type, ExecSpace>);

 private:
  typename traits::execution_space m_space;
  typename traits::index_type m_begin;
  typename traits::index_type m_end;
  typename traits::index_type m_granularity;
  typename traits::index_type m_granularity_mask;

  template <class T, class... OtherProperties>
  friend class ImplRangePolicy;

  template <class... OtherProperties>
  friend class Kokkos::RangePolicy;

 public:
  //! Tag this class as an execution policy
  using execution_policy = Kokkos::RangePolicy<Properties...>;
  using member_type      = typename traits::index_type;
  using index_type       = typename traits::index_type;

  const typename traits::execution_space& space() const { return m_space; }
  KOKKOS_INLINE_FUNCTION member_type begin() const { return m_begin; }
  KOKKOS_INLINE_FUNCTION member_type end() const { return m_end; }

  KOKKOS_INLINE_FUNCTION member_type chunk_size() const {
    return m_granularity;
  }

  // TODO: find a better workaround for Clangs weird instantiation order
  // This thing is here because of an instantiation error, where the RangePolicy
  // is inserted into FunctorValue Traits, which tries decltype on the operator.
  // It tries to do this even though the first argument of parallel for clearly
  // doesn't match.
  void operator()(const int&) const {}

  template <class... OtherProperties>
  ImplRangePolicy(const ImplRangePolicy<OtherProperties...>& p)
      : traits(p),  // base class may contain data such as desired occupancy
        m_space(p.m_space),
        m_begin(p.m_begin),
        m_end(p.m_end),
        m_granularity(p.m_granularity),
        m_granularity_mask(p.m_granularity_mask) {}

  inline ImplRangePolicy()
      : m_space(),
        m_begin(0),
        m_end(0),
        m_granularity(0),
        m_granularity_mask(0) {}

  /** \brief  Total range */
  template <typename IndexType1, typename IndexType2,
            std::enable_if_t<(std::is_convertible_v<IndexType1, member_type> &&
                              std::is_convertible_v<IndexType2, member_type>),
                             bool> = false>
  ImplRangePolicy(const IndexType1 work_begin, const IndexType2 work_end)
      : m_space(typename traits::execution_space()),
        m_begin(work_begin),
        m_end(work_end),
        m_granularity(0),
        m_granularity_mask(0) {
    check_conversion_safety(work_begin);
    check_conversion_safety(work_end);
    check_bounds_validity();
    impl_set_auto_chunk_size();
  }

  /** \brief  Total range */
  template <typename IndexType1, typename IndexType2,
            std::enable_if_t<(std::is_convertible_v<IndexType1, member_type> &&
                              std::is_convertible_v<IndexType2, member_type>),
                             bool> = false>
  KOKKOS_FUNCTION ImplRangePolicy(
      const typename traits::execution_space& work_space,
      const IndexType1 work_begin, const IndexType2 work_end)
      : m_space(work_space),
        m_begin(work_begin),
        m_end(work_end),
        m_granularity(0),
        m_granularity_mask(0) {
    KOKKOS_IF_ON_HOST(check_conversion_safety(work_begin);
                      check_conversion_safety(work_end);
                      check_bounds_validity(); impl_set_auto_chunk_size();)
  }

  template <typename IndexType1, typename IndexType2,
            std::enable_if_t<(std::is_convertible_v<IndexType1, member_type> &&
                              std::is_convertible_v<IndexType2, member_type>),
                             bool> = false>
  ImplRangePolicy(const typename traits::execution_space& work_space,
                  const IndexType1 work_begin, const IndexType2 work_end,
                  const ChunkSize chunk_size)
      : m_space(work_space),
        m_begin(work_begin),
        m_end(work_end),
        m_granularity(0),
        m_granularity_mask(0) {
    check_conversion_safety(work_begin);
    check_conversion_safety(work_end);
    check_bounds_validity();
    impl_set_chunk_size(chunk_size.value);
  }

  /** \brief  Total range */
  template <typename IndexType1, typename IndexType2, typename... Args,
            std::enable_if_t<(std::is_convertible_v<IndexType1, member_type> &&
                              std::is_convertible_v<IndexType2, member_type>),
                             bool> = false>
  ImplRangePolicy(const IndexType1 work_begin, const IndexType2 work_end,
                  const ChunkSize chunk_size)
      : ImplRangePolicy(typename traits::execution_space(), work_begin,
                        work_end, chunk_size) {}

  ImplRangePolicy(const Impl::PolicyUpdate, const ImplRangePolicy& other,
                  typename traits::execution_space space)
      : ImplRangePolicy(other) {
    this->m_space = std::move(space);
  }

 private:
  /** \brief set chunk_size to a discrete value*/
  inline void impl_set_chunk_size(int chunk_size) {
    m_granularity      = chunk_size;
    m_granularity_mask = m_granularity - 1;
  }

  /** \brief finalize chunk_size if it was set to AUTO*/
  inline void impl_set_auto_chunk_size() {
#ifdef KOKKOS_ENABLE_SYCL
    if (std::is_same_v<typename traits::execution_space, Kokkos::SYCL>) {
      // chunk_size <=1 lets the compiler choose the workgroup size when
      // launching kernels
      m_granularity      = 1;
      m_granularity_mask = 0;
      return;
    }
#endif
#ifdef KOKKOS_ENABLE_OPENACC
    if (std::is_same_v<typename traits::execution_space,
                       Kokkos::Experimental::OpenACC>) {
      // chunk_size <=1 lets the compiler choose the chunk size when
      // launching kernels
      m_granularity      = 1;
      m_granularity_mask = 0;
      return;
    }
#endif
    auto concurrency = static_cast<int64_t>(m_space.concurrency());
    if (concurrency == 0) concurrency = 1;

    if (m_granularity > 0 &&
        !Kokkos::has_single_bit(static_cast<unsigned>(m_granularity))) {
      Kokkos::abort("RangePolicy blocking granularity must be power of two");
    }

    int64_t new_chunk_size = 1;
    while (new_chunk_size * 100 * concurrency <
           static_cast<int64_t>(m_end - m_begin))
      new_chunk_size *= 2;
    if (new_chunk_size < 128) {
      new_chunk_size = 1;
      while ((new_chunk_size * 40 * concurrency <
              static_cast<int64_t>(m_end - m_begin)) &&
             (new_chunk_size < 128))
        new_chunk_size *= 2;
    }
    m_granularity      = new_chunk_size;
    m_granularity_mask = m_granularity - 1;
  }

  void check_bounds_validity() {
    if (m_end < m_begin) {
      std::string msg = "Kokkos::RangePolicy bounds error: The lower bound (" +
                        std::to_string(m_begin) +
                        ") is greater than the upper bound (" +
                        std::to_string(m_end) + ").\n";
      Kokkos::abort(msg.c_str());
    }
  }

  // Arithmetic member_type and IndexType: signedness / numeric_limits checks.
  // Always run the round-trip check below as well; for non-arithmetic IndexType
  // only that line applies inside the is_convertible gate.
  template <typename IndexType>
  static void check_conversion_safety([[maybe_unused]] const IndexType bound) {
    if constexpr (std::is_convertible_v<member_type, IndexType>) {
      bool error = false;

      if constexpr (std::is_arithmetic_v<member_type> &&
                    std::is_arithmetic_v<IndexType>) {
        if constexpr (std::is_signed_v<IndexType> !=
                      std::is_signed_v<member_type>) {
          if constexpr (std::is_signed_v<IndexType>) error |= (bound < 0);
          if constexpr (std::is_signed_v<member_type>) {
            if constexpr (sizeof(member_type) <= sizeof(IndexType))
              error |= (bound > static_cast<IndexType>(
                                    std::numeric_limits<member_type>::max()));
            else {
              error |= (bound > std::numeric_limits<IndexType>::max());
            }
          }
        }
      }

      error |=
          (static_cast<IndexType>(static_cast<member_type>(bound)) != bound);

      if (error) {
        std::string bound_to_text = []<typename T>(const T& b) {
          if constexpr (std::is_arithmetic_v<T>)
            return std::to_string(b);
          else
            return std::to_string(static_cast<member_type>(b));
        }(bound);
        std::string msg =
            "Kokkos::RangePolicy bound type error: an unsafe implicit "
            "conversion is performed on a bound (" +
            bound_to_text + "), which may not preserve its original value.\n";

        Kokkos::abort(msg.c_str());
      }
    }
  }

 public:
  /** \brief  Subrange for a partition's rank and size.
   *
   *  Typically used to partition a range over a group of threads.
   */
  struct WorkRange {
    using work_tag =
        typename ImplRangePolicy<ExecSpace, Properties...>::traits::work_tag;
    using member_type =
        typename ImplRangePolicy<ExecSpace, Properties...>::member_type;

    KOKKOS_INLINE_FUNCTION member_type begin() const { return m_begin; }
    KOKKOS_INLINE_FUNCTION member_type end() const { return m_end; }

    /** \brief  Subrange for a partition's rank and size.
     *
     *  Typically used to partition a range over a group of threads.
     */
    KOKKOS_INLINE_FUNCTION
    WorkRange(const ImplRangePolicy& range, const int part_rank,
              const int part_size)
        : m_begin(0), m_end(0) {
      if (part_size) {
        // Split evenly among partitions, then round up to the granularity.
        const member_type work_part =
            ((((range.end() - range.begin()) + (part_size - 1)) / part_size) +
             range.m_granularity_mask) &
            ~member_type(range.m_granularity_mask);

        m_begin = range.begin() + work_part * part_rank;
        m_end   = m_begin + work_part;

        if (range.end() < m_begin) m_begin = range.end();
        if (range.end() < m_end) m_end = range.end();
      }
    }

   private:
    member_type m_begin;
    member_type m_end;
  };
};

}  // namespace Impl

/** \brief  Execution policy to execute work over a single thread
 */
template <typename... Properties>
class SinglePolicy : public Kokkos::Impl::PolicyTraits<Properties...> {
 public:
  using traits = Kokkos::Impl::PolicyTraits<Properties...>;
  static_assert(
      Kokkos::Impl::PolicyTraits<Properties...>::launch_bounds_is_defaulted,
      "SinglePolicy can't be declared with a LaunchBounds");
  using execution_policy = Kokkos::SinglePolicy<Properties...>;
  using execution_space  = typename traits::execution_space;

 private:
  using range_policy = RangePolicy<Kokkos::LaunchBounds<1, 1>, Properties...>;
  range_policy m_range_policy;

 public:
  inline const execution_space& space() const { return m_range_policy.space(); }

  const range_policy& impl_get_range_policy() const { return m_range_policy; }

  template <class... OtherProperties>
  SinglePolicy(const SinglePolicy<OtherProperties...>& p)
      : traits(p), m_range_policy(p.space(), 0, 1) {}

  SinglePolicy() : m_range_policy(execution_space(), 0, 1) {}

  SinglePolicy(const execution_space& exec_space)
      : m_range_policy(exec_space, 0, 1) {}
};

// Deduction guide
SinglePolicy() -> SinglePolicy<>;
SinglePolicy(const DefaultExecutionSpace&)
    -> SinglePolicy<DefaultExecutionSpace>;
template <ExecutionSpace ExecSpace>
SinglePolicy(const ExecSpace&) -> SinglePolicy<ExecSpace>;
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {

template <class ExecSpace, class... Properties>
class TeamPolicyInternal : public Impl::PolicyTraits<Properties...> {
 private:
  using traits = Impl::PolicyTraits<Properties...>;

 public:
  using index_type = typename traits::index_type;

  //----------------------------------------
  /** \brief  Query maximum team size for a given functor.
   *
   *  This size takes into account execution space concurrency limitations and
   *  scratch memory space limitations for reductions, team reduce/scan, and
   *  team shared memory.
   *
   *  This function only works for single-operator functors.
   *  With multi-operator functors it cannot be determined
   *  which operator will be called.
   */
  template <class FunctorType>
  static int team_size_max(const FunctorType&);

  /** \brief  Query recommended team size for a given functor.
   *
   *  This size takes into account execution space concurrency limitations and
   *  scratch memory space limitations for reductions, team reduce/scan, and
   *  team shared memory.
   *
   *  This function only works for single-operator functors.
   *  With multi-operator functors it cannot be determined
   *  which operator will be called.
   */
  template <class FunctorType>
  static int team_size_recommended(const FunctorType&);

  template <class FunctorType>
  static int team_size_recommended(const FunctorType&, const int&);

  template <class FunctorType>
  int team_size_recommended(const FunctorType& functor,
                            const int vector_length);

  //----------------------------------------
  /** \brief  Construct policy with the given instance of the execution space */
  TeamPolicyInternal(const typename traits::execution_space&,
                     int league_size_request, int team_size_request,
                     int vector_length_request = 1);

  TeamPolicyInternal(const typename traits::execution_space&,
                     int league_size_request, const Kokkos::AUTO_t&,
                     int vector_length_request = 1);

  /** \brief  Construct policy with the default instance of the execution space
   */
  TeamPolicyInternal(int league_size_request, int team_size_request,
                     int vector_length_request = 1);

  TeamPolicyInternal(int league_size_request, const Kokkos::AUTO_t&,
                     int vector_length_request = 1);

  /*  TeamPolicyInternal( int league_size_request , int team_size_request );

    TeamPolicyInternal( int league_size_request , const Kokkos::AUTO_t & );*/

  /** \brief  The actual league size (number of teams) of the policy.
   *
   *  This may be smaller than the requested league size due to limitations
   *  of the execution space.
   */
  KOKKOS_INLINE_FUNCTION int league_size() const;

  /** \brief  The actual team size (number of threads per team) of the policy.
   *
   *  This may be smaller than the requested team size due to limitations
   *  of the execution space.
   */
  KOKKOS_INLINE_FUNCTION int team_size() const;

  /** \brief Whether the policy has an automatically determined team size
   */
  inline bool impl_auto_team_size() const;
  /** \brief Whether the policy has an automatically determined vector length
   */
  inline bool impl_auto_vector_length() const;

  static int vector_length_max();

  KOKKOS_INLINE_FUNCTION int impl_vector_length() const;

  inline typename traits::index_type chunk_size() const;

  inline TeamPolicyInternal& set_chunk_size(int chunk_size);

  /** \brief  Parallel execution of a functor calls the functor once with
   *          each member of the execution policy.
   */
  struct member_type {
    /** \brief  Handle to the currently executing team shared scratch memory */
    KOKKOS_INLINE_FUNCTION
    typename traits::execution_space::scratch_memory_space team_shmem() const;

    /** \brief  Rank of this team within the league of teams */
    KOKKOS_INLINE_FUNCTION int league_rank() const;

    /** \brief  Number of teams in the league */
    KOKKOS_INLINE_FUNCTION int league_size() const;

    /** \brief  Rank of this thread within this team */
    KOKKOS_INLINE_FUNCTION int team_rank() const;

    /** \brief  Number of threads in this team */
    KOKKOS_INLINE_FUNCTION int team_size() const;

    /** \brief  Barrier among the threads of this team */
    KOKKOS_INLINE_FUNCTION void team_barrier() const;

    /** \brief  Intra-team reduction. Returns join of all values of the team
     * members. */
    template <class JoinOp>
    KOKKOS_INLINE_FUNCTION typename JoinOp::value_type team_reduce(
        const typename JoinOp::value_type, const JoinOp&) const;

    /** \brief  Intra-team exclusive prefix sum with team_rank() ordering.
     *
     *  The highest rank thread can compute the reduction total as
     *    reduction_total = dev.team_scan( value ) + value ;
     */
    template <typename Type>
    KOKKOS_INLINE_FUNCTION Type team_scan(const Type& value) const;

    /** \brief  Intra-team exclusive prefix sum with team_rank() ordering
     *          with intra-team non-deterministic ordering accumulation.
     *
     *  The global inter-team accumulation value will, at the end of the
     *  league's parallel execution, be the scan's total.
     *  Parallel execution ordering of the league's teams is non-deterministic.
     *  As such the base value for each team's scan operation is similarly
     *  non-deterministic.
     */
    template <typename Type>
    KOKKOS_INLINE_FUNCTION Type team_scan(const Type& value,
                                          Type* const global_accum) const;
  };
};

struct PerTeamValue {
  size_t value;
  PerTeamValue(size_t arg);
};

struct PerThreadValue {
  size_t value;
  PerThreadValue(size_t arg);
};

template <class iType, class... Args>
struct ExtractVectorLength {
  static inline iType value(
      std::enable_if_t<std::is_integral_v<iType>, iType> val, Args...) {
    return val;
  }
  static inline std::enable_if_t<!std::is_integral_v<iType>, int> value(
      std::enable_if_t<!std::is_integral_v<iType>, iType>, Args...) {
    return 1;
  }
};

template <class iType, class... Args>
inline std::enable_if_t<std::is_integral_v<iType>, iType> extract_vector_length(
    iType val, Args...) {
  return val;
}

template <class iType, class... Args>
inline std::enable_if_t<!std::is_integral_v<iType>, int> extract_vector_length(
    iType, Args...) {
  return 1;
}

// Causes abnormal program termination if level is not `0` or `1`
void team_policy_check_valid_storage_level_argument(int level);

}  // namespace Impl

Impl::PerTeamValue PerTeam(const size_t& arg);
Impl::PerThreadValue PerThread(const size_t& arg);

struct ScratchRequest {
  int level;

  size_t per_team;
  size_t per_thread;

  inline ScratchRequest(const int& level_,
                        const Impl::PerTeamValue& team_value) {
    level      = level_;
    per_team   = team_value.value;
    per_thread = 0;
  }

  inline ScratchRequest(const int& level_,
                        const Impl::PerThreadValue& thread_value) {
    level      = level_;
    per_team   = 0;
    per_thread = thread_value.value;
  }

  inline ScratchRequest(const int& level_, const Impl::PerTeamValue& team_value,
                        const Impl::PerThreadValue& thread_value) {
    level      = level_;
    per_team   = team_value.value;
    per_thread = thread_value.value;
  }

  inline ScratchRequest(const int& level_,
                        const Impl::PerThreadValue& thread_value,
                        const Impl::PerTeamValue& team_value) {
    level      = level_;
    per_team   = team_value.value;
    per_thread = thread_value.value;
  }
};

/** \brief  Execution policy for parallel work over a league of teams of
 * threads.
 *
 *  The work functor is called for each thread of each team such that
 *  the team's member threads are guaranteed to be concurrent.
 *
 *  The team's threads have access to team shared scratch memory and
 *  team collective operations.
 *
 *  If the WorkTag is non-void then the first calling argument of the
 *  work functor's parentheses operator is 'const WorkTag &'.
 *  This allows a functor to have multiple work member functions.
 *
 *  Order of template arguments does not matter, since the implementation
 *  uses variadic templates. Each and any of the template arguments can
 *  be omitted.
 *
 *  Possible Template arguments and their default values:
 *    ExecutionSpace (DefaultExecutionSpace): where to execute code. Must be
 * enabled. WorkTag (none): Tag which is used as the first argument for the
 * functor operator. Schedule<Type> (Schedule<Static>): Scheduling Policy
 * (Dynamic, or Static). IndexType<Type> (IndexType<ExecutionSpace::size_type>:
 * Integer Index type used to iterate over the Index space.
 *    LaunchBounds<unsigned,unsigned> Launch Bounds for CUDA compilation,
 *    default of LaunchBounds<0,0> indicates no launch bounds specified.
 */
template <class... Properties>
class TeamPolicy
    : public Impl::TeamPolicyInternal<
          typename Impl::PolicyTraits<Properties...>::execution_space,
          Properties...> {
  using internal_policy = Impl::TeamPolicyInternal<
      typename Impl::PolicyTraits<Properties...>::execution_space,
      Properties...>;

  template <class... OtherProperties>
  friend class TeamPolicy;

  static int validate_league_size_argument(int league_size) {
    if (league_size < 0) {
      std::stringstream err;
      err << "Kokkos::TeamPolicy error: league_size (" << league_size
          << ") must be greater than or equal to 0";
      Kokkos::abort(err.str().c_str());
    }
    return league_size;
  }
  static int validate_team_size_argument(int team_size) {
    if (team_size < 1) {
      std::stringstream err;
      err << "Kokkos::TeamPolicy error: team_size (" << team_size
          << ") must be greater than or equal to 1";
      Kokkos::abort(err.str().c_str());
    }
    return team_size;
  }
  static int validate_vector_length_argument(int vector_length) {
    if (vector_length < 1) {
      std::stringstream err;
      err << "Kokkos::TeamPolicy error: vector_length (" << vector_length
          << ") must be greater than or equal to 1";
      Kokkos::abort(err.str().c_str());
    }
#ifndef KOKKOS_ENABLE_DEPRECATED_CODE_5
    int const vector_length_max = internal_policy::vector_length_max();
    if (vector_length > vector_length_max) {
      std::stringstream err;
      err << "Kokkos::TeamPolicy error: vector_length (" << vector_length
          << ") exceeds the maximum allowed (" << vector_length_max << ")";
      Kokkos::abort(err.str().c_str());
    }
    if (!Kokkos::has_single_bit(static_cast<unsigned>(vector_length))) {
      std::stringstream err;
      err << "Kokkos::TeamPolicy error: vector_length (" << vector_length
          << ") must be a power of 2";
      Kokkos::abort(err.str().c_str());
    }
#endif
    return vector_length;
  }

 public:
  using traits = Impl::PolicyTraits<Properties...>;

  using execution_policy = TeamPolicy<Properties...>;

  TeamPolicy() : internal_policy(0, AUTO) {}

  /** \brief  Construct policy with the given instance of the execution space */
  TeamPolicy(const typename traits::execution_space& space_, int league_size,
             int team_size, int vector_length = 1)
      : internal_policy(space_, validate_league_size_argument(league_size),
                        validate_team_size_argument(team_size),
                        validate_vector_length_argument(vector_length)) {}

  TeamPolicy(const typename traits::execution_space& space_, int league_size,
             Kokkos::AUTO_t, int vector_length = 1)
      : internal_policy(space_, validate_league_size_argument(league_size),
                        Kokkos::AUTO,
                        validate_vector_length_argument(vector_length)) {}

  TeamPolicy(const typename traits::execution_space& space_, int league_size,
             Kokkos::AUTO_t, Kokkos::AUTO_t)
      : internal_policy(space_, league_size, Kokkos::AUTO, Kokkos::AUTO) {}

  TeamPolicy(const typename traits::execution_space& space_, int league_size,
             const int team_size, Kokkos::AUTO_t)
      : internal_policy(space_, validate_league_size_argument(league_size),
                        validate_team_size_argument(team_size), Kokkos::AUTO) {}

  /** \brief  Construct policy with the default instance of the execution space
   */
  TeamPolicy(int league_size, int team_size, int vector_length = 1)
      : internal_policy(validate_league_size_argument(league_size),
                        validate_team_size_argument(team_size),
                        validate_vector_length_argument(vector_length)) {}

  TeamPolicy(int league_size, Kokkos::AUTO_t, int vector_length = 1)
      : internal_policy(validate_league_size_argument(league_size),
                        Kokkos::AUTO,
                        validate_vector_length_argument(vector_length)) {}

  TeamPolicy(int league_size, Kokkos::AUTO_t, Kokkos::AUTO_t)
      : internal_policy(validate_league_size_argument(league_size),
                        Kokkos::AUTO, Kokkos::AUTO) {}

  TeamPolicy(int league_size, int team_size, Kokkos::AUTO_t)
      : internal_policy(validate_league_size_argument(league_size),
                        validate_team_size_argument(team_size), Kokkos::AUTO) {}

  template <class... OtherProperties>
  TeamPolicy(const TeamPolicy<OtherProperties...>& p) : internal_policy(p) {
    // Cannot call converting constructor in the member initializer list because
    // it is not a direct base.
    internal_policy::traits::operator=(p);
  }

  TeamPolicy(const Impl::PolicyUpdate tag, const TeamPolicy& other,
             typename traits::execution_space space)
      : internal_policy(tag, other, std::move(space)) {}

 private:
  TeamPolicy(const internal_policy& p) : internal_policy(p) {}

 public:
  inline TeamPolicy& set_chunk_size(int chunk) {
    static_assert(
        std::is_same_v<decltype(internal_policy::set_chunk_size(chunk)),
                       internal_policy&>,
        "internal set_chunk_size should return a reference");
    return static_cast<TeamPolicy&>(internal_policy::set_chunk_size(chunk));
  }

  inline TeamPolicy& set_scratch_size(const int& level,
                                      const Impl::PerTeamValue& per_team) {
    static_assert(std::is_same_v<decltype(internal_policy::set_scratch_size(
                                     level, per_team)),
                                 internal_policy&>,
                  "internal set_chunk_size should return a reference");

    Impl::team_policy_check_valid_storage_level_argument(level);
    return static_cast<TeamPolicy&>(
        internal_policy::set_scratch_size(level, per_team));
  }
  inline TeamPolicy& set_scratch_size(const int& level,
                                      const Impl::PerThreadValue& per_thread) {
    Impl::team_policy_check_valid_storage_level_argument(level);
    return static_cast<TeamPolicy&>(
        internal_policy::set_scratch_size(level, per_thread));
  }
  inline TeamPolicy& set_scratch_size(const int& level,
                                      const Impl::PerTeamValue& per_team,
                                      const Impl::PerThreadValue& per_thread) {
    Impl::team_policy_check_valid_storage_level_argument(level);
    return static_cast<TeamPolicy&>(
        internal_policy::set_scratch_size(level, per_team, per_thread));
  }
  inline TeamPolicy& set_scratch_size(const int& level,
                                      const Impl::PerThreadValue& per_thread,
                                      const Impl::PerTeamValue& per_team) {
    Impl::team_policy_check_valid_storage_level_argument(level);
    return static_cast<TeamPolicy&>(
        internal_policy::set_scratch_size(level, per_team, per_thread));
  }
};

// Execution space not provided deduces to TeamPolicy<>

TeamPolicy() -> TeamPolicy<>;

TeamPolicy(int, int) -> TeamPolicy<>;
TeamPolicy(int, int, int) -> TeamPolicy<>;
TeamPolicy(int, Kokkos::AUTO_t const&) -> TeamPolicy<>;
TeamPolicy(int, Kokkos::AUTO_t const&, int) -> TeamPolicy<>;
TeamPolicy(int, Kokkos::AUTO_t const&, Kokkos::AUTO_t const&) -> TeamPolicy<>;
TeamPolicy(int, int, Kokkos::AUTO_t const&) -> TeamPolicy<>;

// DefaultExecutionSpace deduces to TeamPolicy<>

TeamPolicy(DefaultExecutionSpace const&, int, int) -> TeamPolicy<>;
TeamPolicy(DefaultExecutionSpace const&, int, int, int) -> TeamPolicy<>;
TeamPolicy(DefaultExecutionSpace const&, int, Kokkos::AUTO_t const&)
    -> TeamPolicy<>;
TeamPolicy(DefaultExecutionSpace const&, int, Kokkos::AUTO_t const&, int)
    -> TeamPolicy<>;
TeamPolicy(DefaultExecutionSpace const&, int, Kokkos::AUTO_t const&,
           Kokkos::AUTO_t const&) -> TeamPolicy<>;
TeamPolicy(DefaultExecutionSpace const&, int, int, Kokkos::AUTO_t const&)
    -> TeamPolicy<>;

// ES != DefaultExecutionSpace deduces to TeamPolicy<ES>

template <typename ES,
          typename = std::enable_if_t<Kokkos::is_execution_space_v<ES>>>
TeamPolicy(ES const&, int, int) -> TeamPolicy<ES>;

template <typename ES,
          typename = std::enable_if_t<Kokkos::is_execution_space_v<ES>>>
TeamPolicy(ES const&, int, int, int) -> TeamPolicy<ES>;

template <typename ES,
          typename = std::enable_if_t<Kokkos::is_execution_space_v<ES>>>
TeamPolicy(ES const&, int, Kokkos::AUTO_t const&) -> TeamPolicy<ES>;

template <typename ES,
          typename = std::enable_if_t<Kokkos::is_execution_space_v<ES>>>
TeamPolicy(ES const&, int, Kokkos::AUTO_t const&, int) -> TeamPolicy<ES>;

template <typename ES,
          typename = std::enable_if_t<Kokkos::is_execution_space_v<ES>>>
TeamPolicy(ES const&, int, Kokkos::AUTO_t const&, Kokkos::AUTO_t const&)
    -> TeamPolicy<ES>;

template <typename ES,
          typename = std::enable_if_t<Kokkos::is_execution_space_v<ES>>>
TeamPolicy(ES const&, int, int, Kokkos::AUTO_t const&) -> TeamPolicy<ES>;

namespace Impl {

// template <unsigned rank, class BaseMember>
// class MDTeamMember {
//  private:
//   using array_type = Kokkos::Array<int, rank>;
//
//   array_type m_league_ranks;
//   array_type m_league_sizes;
//   BaseMember m_base_member;
//
//  public:
//   using scratch_memory_space = typename BaseMember::scratch_memory_space;
//   using execution_space      = typename BaseMember::execution_space;
//   using team_handle          = MDTeamMember;
//
//   template <class... Args>
//   constexpr MDTeamMember(const array_type& league_ranks,
//                          const array_type& league_sizes, Args&&... args)
//       : m_league_ranks(league_ranks),
//         m_league_sizes(league_sizes),
//         m_base_member(args...) {}
//
//   template <class... Args>
//   constexpr MDTeamMember(const array_type& league_sizes, Args&&... args)
//       : m_league_sizes(league_sizes), m_base_member(args...) {
//     int flat_rank = m_base_member.league_rank();
//     for (unsigned i = 0; i < rank; i++) {
//       m_league_ranks[i] = flat_rank % league_sizes[i];
//       flat_rank /= league_sizes[i];
//     }
//   }
//
//   //----------------------------------------
//
//   KOKKOS_INLINE_FUNCTION
//   int team_rank() const noexcept { return m_base_member.team_rank(); }
//
//   KOKKOS_INLINE_FUNCTION
//   int team_size() const noexcept { return m_base_member.team_size(); }
//
//   KOKKOS_INLINE_FUNCTION
//   int league_rank() const noexcept { return m_base_member.league_rank(); }
//
//   KOKKOS_INLINE_FUNCTION
//   int league_rank(int idx) const noexcept { return m_league_ranks[idx]; }
//
//   KOKKOS_INLINE_FUNCTION
//   const array_type& league_ranks() const noexcept { return m_league_ranks; }
//
//   KOKKOS_INLINE_FUNCTION
//   int league_size() const noexcept { return m_base_member.league_size(); }
//
//   KOKKOS_INLINE_FUNCTION
//   int league_size(int idx) const noexcept { return m_league_sizes[idx]; }
//
//   KOKKOS_INLINE_FUNCTION
//   const array_type& league_sizes() const noexcept { return m_league_sizes; }
//
//   //----------------------------------------
//
//   KOKKOS_INLINE_FUNCTION
//   const scratch_memory_space& team_shmem() const {
//     return m_base_member.team_shmem();
//   }
//
//   KOKKOS_INLINE_FUNCTION
//   const scratch_memory_space& team_scratch(int level) const {
//     return m_base_member.team_scratch(level);
//   }
//
//   KOKKOS_INLINE_FUNCTION
//   const scratch_memory_space& thread_scratch(int level) const {
//     return m_base_member.thread_scratch(level);
//   }
//
//   //--------------------------------------------------------------------------
//   // Team collectives
//   //--------------------------------------------------------------------------
//
//   KOKKOS_INLINE_FUNCTION void team_barrier() const noexcept {
//     m_base_member.team_barrier();
//   }
//
//   //--------------------------------------------------------------------------
//
//   template <typename T>
//   KOKKOS_INLINE_FUNCTION void team_broadcast(
//       T& value, const int source_team_rank) const noexcept {
//     m_base_member.team_broadcast(value, source_team_rank);
//   }
//
//   //--------------------------------------------------------------------------
//
//   template <class Closure, typename T>
//   KOKKOS_INLINE_FUNCTION void team_broadcast(
//       Closure const& f, T& value, const int source_team_rank) const noexcept
//       {
//     m_base_member.team_broadcast(f, value, source_team_rank);
//   }
//
//   //--------------------------------------------------------------------------
//   // team_reduce( Sum(result) );
//   // team_reduce( Min(result) );
//   // team_reduce( Max(result) );
//
//   template <Reducer ReducerType>
//   KOKKOS_INLINE_FUNCTION void team_reduce(
//       ReducerType const& reducer) const noexcept {
//     m_base_member.team_reduce(reducer);
//   }
//
//   template <Reducer ReducerType>
//   KOKKOS_INLINE_FUNCTION void team_reduce(
//       ReducerType const& reducer,
//       typename ReducerType::value_type contribution) const noexcept {
//     m_base_member.team_reduce(reducer, contribution);
//   }
//
//   //--------------------------------------------------------------------------
//
//   template <typename T>
//   KOKKOS_INLINE_FUNCTION T team_scan(T const& value,
//                                      T* const global = nullptr) const
//                                      noexcept {
//     return m_base_member.team_scan(value, global);
//   }
// };

template <unsigned rank, class BaseMember>
class MDTeamMember : public BaseMember {
 private:
  using array_type = Kokkos::Array<int, rank>;

  array_type m_league_ranks;
  array_type m_league_sizes;
  // BaseMember m_base_member;

 public:
  // using scratch_memory_space = typename BaseMember::scratch_memory_space;
  // using execution_space      = typename BaseMember::execution_space;
  using team_handle = MDTeamMember;

  template <class... Args>
  constexpr KOKKOS_FUNCTION MDTeamMember(const array_type& league_ranks,
                                         const array_type& league_sizes,
                                         Args&&... args)
      : BaseMember(args...),
        m_league_ranks(league_ranks),
        m_league_sizes(league_sizes) {}

  template <class... Args>
  constexpr KOKKOS_FUNCTION MDTeamMember(const array_type& league_sizes,
                                         Args&&... args)
      : BaseMember(args...), m_league_sizes(league_sizes) {
    int flat_rank = BaseMember::league_rank();
    for (int i = rank - 1; i >= 0; i--) {
      m_league_ranks[i] = flat_rank % league_sizes[i];
      flat_rank /= league_sizes[i];
    }
  }

  operator BaseMember() const { return static_cast<BaseMember>(*this); }

  void update_league_ranks(int offset) {
    int carry = offset;
    for (int i = rank - 1; i >= 0 && carry != 0; i--) {
      m_league_ranks[i] += carry;
      carry = m_league_ranks[i] / m_league_sizes[i];
      m_league_ranks[i] %= m_league_sizes[i];
    }
    // the last rank overflowing would mean that we iterated beyond the Policy's
    // domain
    KOKKOS_ASSERT(carry == 0);
  }

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION
  int league_rank() const noexcept { return BaseMember::league_rank(); }

  KOKKOS_INLINE_FUNCTION
  int league_rank(int idx) const noexcept { return m_league_ranks[idx]; }

  KOKKOS_INLINE_FUNCTION
  const array_type& league_ranks() const noexcept { return m_league_ranks; }

  KOKKOS_INLINE_FUNCTION
  int league_size() const noexcept { return BaseMember::league_size(); }

  KOKKOS_INLINE_FUNCTION
  int league_size(int idx) const noexcept { return m_league_sizes[idx]; }

  KOKKOS_INLINE_FUNCTION
  const array_type& league_sizes() const noexcept { return m_league_sizes; }
};

// template <class ExecSpace, unsigned rank_, class... Properties>
// class TeamPolicyInternal<
//     Kokkos::Rank<rank_, Kokkos::Iterate::Default, Kokkos::Iterate::Default>,
//     ExecSpace, Properties...> : public Impl::PolicyTraits<Properties...> {
//   // template <class ExecSpace, class P, class... Properties>
//   // requires (!std::is_void_v<typename Impl::PolicyTraits<P,
//   // Properties...>::iteration_pattern>) class TeamPolicyInternal<P,
//   ExecSpace,
//   // Properties...> : public Impl::PolicyTraits<P, Properties...> {
//  public:
//   static constexpr unsigned rank = rank_;
//
//  private:
//   using traits          = Impl::PolicyTraits<Properties...>;
//   using base_policy     = TeamPolicyInternal<ExecSpace, Properties...>;
//   using array_type      = Kokkos::Array<int, rank>;
//   using execution_space = ExecSpace;
//
//   int compute_league_size(const array_type& league_sizes) {
//     int res = 1;
//     for (unsigned i = 0; i < rank; i++) {
//       res *= league_sizes[i];
//     }
//     return res;
//   }
//
//   array_type m_league_sizes;
//   base_policy m_flat_policy;
//
//  public:
//   const execution_space& space() const { return m_flat_policy.space(); }
//
//   template <class FunctorType>
//   int team_size_max(const FunctorType& f, const ParallelForTag& t) const {
//     return m_flat_policy.team_size_max(f, t);
//   }
//   template <class FunctorType>
//   int team_size_max(const FunctorType& f, const ParallelReduceTag& t) const {
//     return m_flat_policy.team_size_max(f, t);
//   }
//   template <class FunctorType, class ReducerType>
//   int team_size_max(const FunctorType& f, const ReducerType& reducer,
//                     const ParallelReduceTag& t) const {
//     return m_flat_policy.team_size_max(f, reducer, t);
//   }
//   template <class FunctorType>
//   int team_size_recommended(const FunctorType& f,
//                             const ParallelForTag& t) const {
//     return m_flat_policy.team_size_recommended(f, t);
//   }
//   template <class FunctorType>
//   int team_size_recommended(const FunctorType& f,
//                             const ParallelReduceTag& t) const {
//     return m_flat_policy.team_size_recommended(f, t);
//   }
//   template <class FunctorType, class ReducerType>
//   int team_size_recommended(const FunctorType& f, const ReducerType& reducer,
//                             const ParallelReduceTag& t) const {
//     return m_flat_policy.team_size_recommended(f, reducer, t);
//   }
//
//   //----------------------------------------
//
//   inline int team_size() const { return m_flat_policy.team_size(); }
//   inline bool impl_auto_team_size() const {
//     return m_flat_policy.impl_auto_team_size();
//   }
//   inline bool impl_auto_vector_length() const {
//     return m_flat_policy.impl_auto_vector_length();
//   }
//   inline void impl_set_team_size(size_t team_size) {
//     m_flat_policy.impl_set_team_size(team_size);
//   }
//   inline void impl_set_vector_length(size_t vector_length) {
//     m_flat_policy.impl_set_vector_length(vector_length);
//   }
//   inline int league_size() const { return m_flat_policy.league_size(); }
//   inline int league_size(int dim) const {
//     return m_league_sizes[dim]; /* FIXME: bounds checking? */
//   }
//   inline const array_type& league_sizes() const { return m_league_sizes; }
//
//   inline size_t scratch_size(const int& level, int team_size = -1) const {
//     return m_flat_policy.scratch_size(level, team_size);
//   }
//   inline size_t team_scratch_size(int level) const {
//     return m_flat_policy.team_scratch_size(level);
//   }
//   inline size_t thread_scratch_size(int level) const {
//     return m_flat_policy.thread_scratch_size(level);
//   }
//
//   inline int impl_vector_length() const {
//     return m_flat_policy.impl_vector_length();
//   }
//   inline static int vector_length_max() {
//     return base_policy::vector_length_max();
//   }
//
//   inline static int scratch_size_max(int level) {
//     return base_policy::scratch_size_max(level);
//   }
//
//   /** \brief  Specify league size, request team size */
//   TeamPolicyInternal(const execution_space& space,
//                      const array_type& league_sizes_request,
//                      int team_size_request, int vector_length_request = 1)
//       : m_league_sizes(league_sizes_request),
//         m_flat_policy(space, compute_league_size(league_sizes_request),
//                       team_size_request, vector_length_request) {}
//
//   TeamPolicyInternal(const execution_space& space,
//                      const array_type& league_sizes_request,
//                      const Kokkos::AUTO_t& /**team_size_request*/,
//                      int vector_length_request = 1)
//       : TeamPolicyInternal(space, league_sizes_request, -1,
//                            vector_length_request) {}
//
//   TeamPolicyInternal(const execution_space& space,
//                      const array_type& league_sizes_request,
//                      const Kokkos::AUTO_t& /* team_size_request */
//                      ,
//                      const Kokkos::AUTO_t& /* vector_length_request */
//                      )
//       : TeamPolicyInternal(space, league_sizes_request, -1, -1) {}
//
//   TeamPolicyInternal(const execution_space& space,
//                      const array_type& league_sizes_request,
//                      int team_size_request,
//                      const Kokkos::AUTO_t& /* vector_length_request */
//                      )
//       : TeamPolicyInternal(space, league_sizes_request, team_size_request,
//       -1) {
//   }
//
//   TeamPolicyInternal(const array_type& league_sizes_request,
//                      const Kokkos::AUTO_t& team_size_request,
//                      int vector_length_request = 1)
//       : TeamPolicyInternal(typename traits::execution_space(),
//                            league_sizes_request, team_size_request,
//                            vector_length_request) {}
//
//   TeamPolicyInternal(const array_type& league_sizes_request,
//                      const Kokkos::AUTO_t& team_size_request,
//                      const Kokkos::AUTO_t& vector_length_request)
//       : TeamPolicyInternal(typename traits::execution_space(),
//                            league_sizes_request, team_size_request,
//                            vector_length_request) {}
//   TeamPolicyInternal(const array_type& league_sizes_request,
//                      int team_size_request,
//                      const Kokkos::AUTO_t& vector_length_request)
//       : TeamPolicyInternal(typename traits::execution_space(),
//                            league_sizes_request, team_size_request,
//                            vector_length_request) {}
//
//   TeamPolicyInternal(const array_type& league_sizes_request,
//                      int team_size_request, int vector_length_request = 1)
//       : TeamPolicyInternal(typename traits::execution_space(),
//                            league_sizes_request, team_size_request,
//                            vector_length_request) {}
//
//   TeamPolicyInternal(const PolicyUpdate, const TeamPolicyInternal& other,
//                      typename traits::execution_space space)
//       : TeamPolicyInternal(other) {
//     this->m_space = std::move(space);
//   }
//
//   inline int chunk_size() const { return m_flat_policy.chunk_size(); }
//
//   /** \brief set chunk_size to a discrete value*/
//   inline TeamPolicyInternal& set_chunk_size(
//       typename traits::index_type chunk_size_) {
//     m_flat_policy.set_chunk_size(chunk_size_);
//     return *this;
//   }
//
//   /** \brief set per team scratch size for a specific level of the scratch
//    * hierarchy */
//   inline TeamPolicyInternal& set_scratch_size(const int& level,
//                                               const PerTeamValue& per_team) {
//     m_flat_policy.set_scratch_size(level, per_team);
//     return *this;
//   }
//
//   /** \brief set per thread scratch size for a specific level of the scratch
//    * hierarchy */
//   inline TeamPolicyInternal& set_scratch_size(
//       const int& level, const PerThreadValue& per_thread) {
//     m_flat_policy.set_scratch_size(level, per_thread);
//     return *this;
//   }
//
//   /** \brief set per thread and per team scratch size for a specific level of
//    * the scratch hierarchy */
//   inline TeamPolicyInternal& set_scratch_size(
//       const int& level, const PerTeamValue& per_team,
//       const PerThreadValue& per_thread) {
//     m_flat_policy.set_scratch_size(level, per_team, per_thread);
//     return *this;
//   }
//
//   using member_type =
//       Impl::MDTeamMember<rank, typename base_policy::member_type>;
// };

template <class ExecSpace, unsigned rank_, class... Properties>
class TeamPolicyInternal<
    Kokkos::Rank<rank_, Kokkos::Iterate::Default, Kokkos::Iterate::Default>,
    ExecSpace, Properties...>
    : public TeamPolicyInternal<ExecSpace, Properties...> {
 public:
  static constexpr unsigned rank = rank_;

 private:
  using traits          = Impl::PolicyTraits<Properties...>;
  using base_policy     = TeamPolicyInternal<ExecSpace, Properties...>;
  using array_type      = Kokkos::Array<int, rank>;
  using execution_space = ExecSpace;

  static int compute_league_size(const array_type& league_sizes) {
    int res = 1;
    for (unsigned i = 0; i < rank; i++) {
      res *= league_sizes[i];
    }
    return res;
  }

  array_type m_league_sizes;

 public:
  inline int league_size() const { return base_policy::league_size(); }
  inline int league_size(int dim) const {
    return m_league_sizes[dim]; /* FIXME: bounds checking? */
  }
  inline const array_type& league_sizes() const { return m_league_sizes; }

  /** \brief  Specify league size, request team size */
  TeamPolicyInternal(const execution_space& space,
                     const array_type& league_sizes_request,
                     int team_size_request, int vector_length_request = 1)
      : base_policy(space, compute_league_size(league_sizes_request),
                    team_size_request, vector_length_request),
        m_league_sizes(league_sizes_request) {}

  TeamPolicyInternal(const execution_space& space,
                     const array_type& league_sizes_request,
                     const Kokkos::AUTO_t& /**team_size_request*/,
                     int vector_length_request = 1)
      : TeamPolicyInternal(space, league_sizes_request, -1,
                           vector_length_request) {}

  TeamPolicyInternal(const execution_space& space,
                     const array_type& league_sizes_request,
                     const Kokkos::AUTO_t& /* team_size_request */
                     ,
                     const Kokkos::AUTO_t& /* vector_length_request */
                     )
      : TeamPolicyInternal(space, league_sizes_request, -1, -1) {}

  TeamPolicyInternal(const execution_space& space,
                     const array_type& league_sizes_request,
                     int team_size_request,
                     const Kokkos::AUTO_t& /* vector_length_request */
                     )
      : TeamPolicyInternal(space, league_sizes_request, team_size_request, -1) {
  }

  TeamPolicyInternal(const array_type& league_sizes_request,
                     const Kokkos::AUTO_t& team_size_request,
                     int vector_length_request = 1)
      : TeamPolicyInternal(typename traits::execution_space(),
                           league_sizes_request, team_size_request,
                           vector_length_request) {}

  TeamPolicyInternal(const array_type& league_sizes_request,
                     const Kokkos::AUTO_t& team_size_request,
                     const Kokkos::AUTO_t& vector_length_request)
      : TeamPolicyInternal(typename traits::execution_space(),
                           league_sizes_request, team_size_request,
                           vector_length_request) {}
  TeamPolicyInternal(const array_type& league_sizes_request,
                     int team_size_request,
                     const Kokkos::AUTO_t& vector_length_request)
      : TeamPolicyInternal(typename traits::execution_space(),
                           league_sizes_request, team_size_request,
                           vector_length_request) {}

  TeamPolicyInternal(const array_type& league_sizes_request,
                     int team_size_request, int vector_length_request = 1)
      : TeamPolicyInternal(typename traits::execution_space(),
                           league_sizes_request, team_size_request,
                           vector_length_request) {}

  TeamPolicyInternal(const PolicyUpdate, const TeamPolicyInternal& other,
                     typename traits::execution_space space)
      : TeamPolicyInternal(other) {
    this->m_space = std::move(space);
  }

  /** \brief set chunk_size to a discrete value*/
  inline TeamPolicyInternal& set_chunk_size(
      typename traits::index_type chunk_size_) {
    base_policy::set_chunk_size(chunk_size_);
    return *this;
  }

  /** \brief set per team scratch size for a specific level of the scratch
   * hierarchy */
  inline TeamPolicyInternal& set_scratch_size(const int& level,
                                              const PerTeamValue& per_team) {
    base_policy::set_scratch_size(level, per_team);
    return *this;
  }

  /** \brief set per thread scratch size for a specific level of the scratch
   * hierarchy */
  inline TeamPolicyInternal& set_scratch_size(
      const int& level, const PerThreadValue& per_thread) {
    base_policy::set_scratch_size(level, per_thread);
    return *this;
  }

  /** \brief set per thread and per team scratch size for a specific level of
   * the scratch hierarchy */
  inline TeamPolicyInternal& set_scratch_size(
      const int& level, const PerTeamValue& per_team,
      const PerThreadValue& per_thread) {
    base_policy::set_scratch_size(level, per_team, per_thread);
    return *this;
  }

  using member_type =
      Impl::MDTeamMember<rank, typename base_policy::member_type>;
};
}  // namespace Impl

template <class... Properties>
class MDTeamPolicy;

template <class P, class... Properties>
class MDTeamPolicy<P, Properties...>
    : public Impl::TeamPolicyInternal<
          P, typename Impl::PolicyTraits<P, Properties...>::execution_space,
          Properties...> {
 public:
  using traits           = Kokkos::Impl::PolicyTraits<P, Properties...>;
  using execution_policy = MDTeamPolicy<P, Properties...>;
  using execution_space  = traits::execution_space;

  static constexpr int rank = traits::iteration_pattern::rank;

 private:
  using team_policy = TeamPolicy<Properties...>;

  using internal_policy = Impl::TeamPolicyInternal<
      P, typename Impl::PolicyTraits<P, Properties...>::execution_space,
      Properties...>;

  using array_type = Kokkos::Array<int, rank>;

  template <class T>
  array_type validate_league_sizes_argument(
      const Kokkos::Array<T, rank>& league_sizes) {
    array_type res;
    for (unsigned i = 0; i < rank; i++) {
      int size = static_cast<int>(league_sizes[i]);
      if (league_sizes[i] < 0 || size < 0) {
        std::stringstream err;
        err << "Kokkos::MDTeamPolicy error: league_size for dimension " << i
            << " (" << league_sizes[i]
            << ") must be greater than or equal to 0";
        Kokkos::abort(err.str().c_str());
      }
      res[i] = size;
    }
    return res;
  }

  template <class T>
  array_type validate_league_sizes_argument(const T (&league_sizes)[rank]) {
    array_type res;
    for (unsigned i = 0; i < rank; i++) {
      int size = static_cast<int>(league_sizes[i]);
      if (league_sizes[i] < 0 || size < 0) {
        std::stringstream err;
        err << "Kokkos::MDTeamPolicy error: league_size for dimension " << i
            << " (" << league_sizes[i]
            << ") must be greater than or equal to 0";
        Kokkos::abort(err.str().c_str());
      }
      res[i] = size;
    }
    return res;
  }

  static int validate_team_size_argument(int team_size) {
    if (team_size < 1) {
      std::stringstream err;
      err << "Kokkos::MDTeamPolicy error: team_size (" << team_size
          << ") must be greater than or equal to 1";
      Kokkos::abort(err.str().c_str());
    }
    return team_size;
  }

  static int validate_vector_length_argument(int vector_length) {
    if (vector_length < 1) {
      std::stringstream err;
      err << "Kokkos::MDTeamPolicy error: vector_length (" << vector_length
          << ") must be greater than or equal to 1";
      Kokkos::abort(err.str().c_str());
    }
    return vector_length;
  }

 public:
  static_assert(!std::is_void_v<typename traits::iteration_pattern>,
                "Kokkos Error: MD iteration pattern not defined");

  MDTeamPolicy() : internal_policy({}, AUTO) {}

  template <std::integral T>
  MDTeamPolicy(const execution_space& space,
               const Kokkos::Array<T, rank>& league_sizes, int team_size,
               int vector_length = 1)
      : internal_policy(space, validate_league_sizes_argument(league_sizes),
                        validate_team_size_argument(team_size),
                        validate_vector_length_argument(vector_length)) {}

  template <std::integral T>
  MDTeamPolicy(const execution_space& space,
               const Kokkos::Array<T, rank>& league_sizes, Kokkos::AUTO_t,
               int vector_length = 1)
      : internal_policy(space, validate_league_sizes_argument(league_sizes),
                        Kokkos::AUTO,
                        validate_vector_length_argument(vector_length)) {}

  template <std::integral T>
  MDTeamPolicy(const execution_space& space,
               const Kokkos::Array<T, rank>& league_sizes, Kokkos::AUTO_t,
               Kokkos::AUTO_t)
      : internal_policy(space, validate_league_sizes_argument(league_sizes),
                        Kokkos::AUTO, Kokkos::AUTO) {}

  template <std::integral T>
  MDTeamPolicy(const execution_space& space,
               const Kokkos::Array<T, rank>& league_sizes, int team_size,
               Kokkos::AUTO_t)
      : internal_policy(space, validate_league_sizes_argument(league_sizes),
                        validate_team_size_argument(team_size), Kokkos::AUTO) {}

  /** \brief  Construct policy with the default instance of the execution space
   */
  template <std::integral T>
  MDTeamPolicy(const Kokkos::Array<T, rank>& league_sizes, int team_size,
               int vector_length = 1)
      : internal_policy(validate_league_sizes_argument(league_sizes),
                        validate_team_size_argument(team_size),
                        validate_vector_length_argument(vector_length)) {}

  template <std::integral T>
  MDTeamPolicy(const Kokkos::Array<T, rank>& league_sizes, Kokkos::AUTO_t,
               int vector_length = 1)
      : internal_policy(validate_league_sizes_argument(league_sizes),
                        Kokkos::AUTO,
                        validate_vector_length_argument(vector_length)) {}

  template <std::integral T>
  MDTeamPolicy(const Kokkos::Array<T, rank>& league_sizes, Kokkos::AUTO_t,
               Kokkos::AUTO_t)
      : internal_policy(validate_league_sizes_argument(league_sizes),
                        Kokkos::AUTO, Kokkos::AUTO) {}

  template <std::integral T>
  MDTeamPolicy(const Kokkos::Array<T, rank>& league_sizes, int team_size,
               Kokkos::AUTO_t)
      : internal_policy(validate_league_sizes_argument(league_sizes),
                        validate_team_size_argument(team_size), Kokkos::AUTO) {}

  template <std::integral T>
  MDTeamPolicy(const execution_space& space, const T (&league_sizes)[rank],
               int team_size, int vector_length = 1)
      : internal_policy(space, validate_league_sizes_argument(league_sizes),
                        validate_team_size_argument(team_size),
                        validate_vector_length_argument(vector_length)) {}

  template <std::integral T>
  MDTeamPolicy(const execution_space& space, const T (&league_sizes)[rank],
               Kokkos::AUTO_t, int vector_length = 1)
      : internal_policy(space, validate_league_sizes_argument(league_sizes),
                        Kokkos::AUTO,
                        validate_vector_length_argument(vector_length)) {}

  template <std::integral T>
  MDTeamPolicy(const execution_space& space, const T (&league_sizes)[rank],
               Kokkos::AUTO_t, Kokkos::AUTO_t)
      : internal_policy(space, validate_league_sizes_argument(league_sizes),
                        Kokkos::AUTO, Kokkos::AUTO) {}

  template <std::integral T>
  MDTeamPolicy(const execution_space& space, const T (&league_sizes)[rank],
               int team_size, Kokkos::AUTO_t)
      : internal_policy(space, validate_league_sizes_argument(league_sizes),
                        validate_team_size_argument(team_size), Kokkos::AUTO) {}

  /** \brief  Construct policy with the default instance of the execution space
   */
  template <std::integral T>
  MDTeamPolicy(const T (&league_sizes)[rank], int team_size,
               int vector_length = 1)
      : internal_policy(validate_league_sizes_argument(league_sizes),
                        validate_team_size_argument(team_size),
                        validate_vector_length_argument(vector_length)) {}

  template <std::integral T>
  MDTeamPolicy(const T (&league_sizes)[rank], Kokkos::AUTO_t,
               int vector_length = 1)
      : internal_policy(validate_league_sizes_argument(league_sizes),
                        Kokkos::AUTO,
                        validate_vector_length_argument(vector_length)) {}

  template <std::integral T>
  MDTeamPolicy(const T (&league_sizes)[rank], Kokkos::AUTO_t, Kokkos::AUTO_t)
      : internal_policy(validate_league_sizes_argument(league_sizes),
                        Kokkos::AUTO, Kokkos::AUTO) {}

  template <std::integral T>
  MDTeamPolicy(const T (&league_sizes)[rank], int team_size, Kokkos::AUTO_t)
      : internal_policy(validate_league_sizes_argument(league_sizes),
                        validate_team_size_argument(team_size), Kokkos::AUTO) {}

  template <class... OtherProperties>
  MDTeamPolicy(const MDTeamPolicy<OtherProperties...>& p) : internal_policy(p) {
    // Cannot call converting constructor in the member initializer list because
    // it is not a direct base.
    internal_policy::traits::operator=(p);
  }

  MDTeamPolicy(const Impl::PolicyUpdate tag, const MDTeamPolicy& other,
               typename traits::execution_space space)
      : internal_policy(tag, other, std::move(space)) {}

 private:
  MDTeamPolicy(const internal_policy& p) : internal_policy(p) {}

 public:
  inline MDTeamPolicy& set_chunk_size(int chunk) {
    static_assert(
        std::is_same_v<decltype(internal_policy::set_chunk_size(chunk)),
                       internal_policy&>,
        "internal set_chunk_size should return a reference");
    return static_cast<MDTeamPolicy&>(internal_policy::set_chunk_size(chunk));
  }

  inline MDTeamPolicy& set_scratch_size(const int& level,
                                        const Impl::PerTeamValue& per_team) {
    static_assert(std::is_same_v<decltype(internal_policy::set_scratch_size(
                                     level, per_team)),
                                 internal_policy&>,
                  "internal set_chunk_size should return a reference");

    Impl::team_policy_check_valid_storage_level_argument(level);
    return static_cast<MDTeamPolicy&>(
        internal_policy::set_scratch_size(level, per_team));
  }
  inline MDTeamPolicy& set_scratch_size(
      const int& level, const Impl::PerThreadValue& per_thread) {
    Impl::team_policy_check_valid_storage_level_argument(level);
    return static_cast<MDTeamPolicy&>(
        internal_policy::set_scratch_size(level, per_thread));
  }
  inline MDTeamPolicy& set_scratch_size(
      const int& level, const Impl::PerTeamValue& per_team,
      const Impl::PerThreadValue& per_thread) {
    Impl::team_policy_check_valid_storage_level_argument(level);
    return static_cast<MDTeamPolicy&>(
        internal_policy::set_scratch_size(level, per_team, per_thread));
  }
  inline MDTeamPolicy& set_scratch_size(const int& level,
                                        const Impl::PerThreadValue& per_thread,
                                        const Impl::PerTeamValue& per_team) {
    Impl::team_policy_check_valid_storage_level_argument(level);
    return static_cast<MDTeamPolicy&>(
        internal_policy::set_scratch_size(level, per_team, per_thread));
  }
};

namespace Impl {

template <typename iType, class TeamMemberType>
struct TeamThreadRangeBoundariesStruct {
 private:
  KOKKOS_INLINE_FUNCTION static iType ibegin(const iType& arg_begin,
                                             const iType& arg_end,
                                             const iType& arg_rank,
                                             const iType& arg_size) {
    return arg_begin +
           ((arg_end - arg_begin + arg_size - 1) / arg_size) * arg_rank;
  }

  KOKKOS_INLINE_FUNCTION static iType iend(const iType& arg_begin,
                                           const iType& arg_end,
                                           const iType& arg_rank,
                                           const iType& arg_size) {
    const iType end_ =
        arg_begin +
        ((arg_end - arg_begin + arg_size - 1) / arg_size) * (arg_rank + 1);
    return end_ < arg_end ? end_ : arg_end;
  }

 public:
  using index_type = iType;
  const iType start;
  const iType end;
  enum { increment = 1 };
  const TeamMemberType& member;

  KOKKOS_INLINE_FUNCTION
  TeamThreadRangeBoundariesStruct(const TeamMemberType& arg_thread,
                                  const iType& arg_count)
      : start(ibegin(0, arg_count, arg_thread.team_rank(),
                     arg_thread.team_size())),
        end(iend(0, arg_count, arg_thread.team_rank(), arg_thread.team_size())),
        member(arg_thread) {}

  KOKKOS_INLINE_FUNCTION
  TeamThreadRangeBoundariesStruct(const TeamMemberType& arg_thread,
                                  const iType& arg_begin, const iType& arg_end)
      : start(ibegin(arg_begin, arg_end, arg_thread.team_rank(),
                     arg_thread.team_size())),
        end(iend(arg_begin, arg_end, arg_thread.team_rank(),
                 arg_thread.team_size())),
        member(arg_thread) {}
};

// For some backends, vector length is required to be 1 (currently Serial, HPX
// and Threads), so there is no need for a TeamVectorRangeBoundariesStruct
// distinct from TeamThreadRangeBoundariesStruct for the base implementation.
// Backends with non-trivial vector length are responsible for implementing
// TeamVectorRangeBoundariesStruct for their specific TeamMemberType.

template <typename iType, class TeamMemberType>
struct TeamVectorRangeBoundariesStruct
    : public TeamThreadRangeBoundariesStruct<iType, TeamMemberType> {
  using base_t = TeamThreadRangeBoundariesStruct<iType, TeamMemberType>;
  using base_t::base_t;
};

template <typename iType, class TeamMemberType>
struct ThreadVectorRangeBoundariesStruct {
  using index_type = iType;
  const index_type start;
  const index_type end;
  enum { increment = 1 };

  KOKKOS_INLINE_FUNCTION
  constexpr ThreadVectorRangeBoundariesStruct(
      const TeamMemberType, const index_type& arg_count) noexcept
      : start(static_cast<index_type>(0)), end(arg_count) {}

  KOKKOS_INLINE_FUNCTION
  constexpr ThreadVectorRangeBoundariesStruct(
      const TeamMemberType, const index_type& arg_begin,
      const index_type& arg_end) noexcept
      : start(static_cast<index_type>(arg_begin)), end(arg_end) {}
};

template <class TeamMemberType>
struct ThreadSingleStruct {
  const TeamMemberType& team_member;
  KOKKOS_INLINE_FUNCTION
  ThreadSingleStruct(const TeamMemberType& team_member_)
      : team_member(team_member_) {}
};

template <class TeamMemberType>
struct VectorSingleStruct {
  const TeamMemberType& team_member;
  KOKKOS_INLINE_FUNCTION
  VectorSingleStruct(const TeamMemberType& team_member_)
      : team_member(team_member_) {}
};

}  // namespace Impl

/** \brief  Execution policy for parallel work over a threads within a team.
 *
 *  The range is split over all threads in a team. The Mapping scheme depends on
 * the architecture. This policy is used together with a parallel pattern as a
 * nested layer within a kernel launched with the TeamPolicy. This variant
 * expects a single count. So the range is (0,count].
 */
template <typename iType, class TeamMemberType, class _never_use_this_overload>
KOKKOS_INLINE_FUNCTION_DELETED
    Impl::TeamThreadRangeBoundariesStruct<iType, TeamMemberType>
    TeamThreadRange(const TeamMemberType&, const iType& count) = delete;

/** \brief  Execution policy for parallel work over a threads within a team.
 *
 *  The range is split over all threads in a team. The Mapping scheme depends on
 * the architecture. This policy is used together with a parallel pattern as a
 * nested layer within a kernel launched with the TeamPolicy. This variant
 * expects a begin and end. So the range is (begin,end].
 */
template <typename iType1, typename iType2, class TeamMemberType,
          class _never_use_this_overload>
KOKKOS_INLINE_FUNCTION_DELETED Impl::TeamThreadRangeBoundariesStruct<
    std::common_type_t<iType1, iType2>, TeamMemberType>
TeamThreadRange(const TeamMemberType&, const iType1& begin,
                const iType2& end) = delete;

/** \brief  Execution policy for parallel work over a threads within a team.
 *
 *  The range is split over all threads in a team. The Mapping scheme depends on
 * the architecture. This policy is used together with a parallel pattern as a
 * nested layer within a kernel launched with the TeamPolicy. This variant
 * expects a single count. So the range is (0,count].
 */
template <typename iType, class TeamMemberType, class _never_use_this_overload>
KOKKOS_INLINE_FUNCTION_DELETED
    Impl::TeamVectorRangeBoundariesStruct<iType, TeamMemberType>
    TeamVectorRange(const TeamMemberType&, const iType& count) = delete;

/** \brief  Execution policy for parallel work over a threads within a team.
 *
 *  The range is split over all threads in a team. The Mapping scheme depends on
 * the architecture. This policy is used together with a parallel pattern as a
 * nested layer within a kernel launched with the TeamPolicy. This variant
 * expects a begin and end. So the range is (begin,end].
 */
template <typename iType1, typename iType2, class TeamMemberType,
          class _never_use_this_overload>
KOKKOS_INLINE_FUNCTION_DELETED Impl::TeamVectorRangeBoundariesStruct<
    std::common_type_t<iType1, iType2>, TeamMemberType>
TeamVectorRange(const TeamMemberType&, const iType1& begin,
                const iType2& end) = delete;

/** \brief  Execution policy for a vector parallel loop.
 *
 *  The range is split over all vector lanes in a thread. The Mapping scheme
 * depends on the architecture. This policy is used together with a parallel
 * pattern as a nested layer within a kernel launched with the TeamPolicy. This
 * variant expects a single count. So the range is (0,count].
 */
template <typename iType, class TeamMemberType, class _never_use_this_overload>
KOKKOS_INLINE_FUNCTION_DELETED
    Impl::ThreadVectorRangeBoundariesStruct<iType, TeamMemberType>
    ThreadVectorRange(const TeamMemberType&, const iType& count) = delete;

template <typename iType1, typename iType2, class TeamMemberType,
          class _never_use_this_overload>
KOKKOS_INLINE_FUNCTION_DELETED Impl::ThreadVectorRangeBoundariesStruct<
    std::common_type_t<iType1, iType2>, TeamMemberType>
ThreadVectorRange(const TeamMemberType&, const iType1& arg_begin,
                  const iType2& arg_end) = delete;

namespace Impl {

enum class TeamMDRangeLastNestLevel : bool { NotLastNestLevel, LastNestLevel };
enum class TeamMDRangeParThread : bool { NotParThread, ParThread };
enum class TeamMDRangeParVector : bool { NotParVector, ParVector };
enum class TeamMDRangeThreadAndVector : bool { NotBoth, Both };

template <typename Rank, TeamMDRangeThreadAndVector ThreadAndVector>
struct HostBasedNestLevel;

template <typename Rank, TeamMDRangeThreadAndVector ThreadAndVector>
struct AcceleratorBasedNestLevel;

// ThreadAndVectorNestLevel determines on which nested level parallelization
// happens.
//   - Rank is Kokkos::Rank<TotalNestLevel, Iter>
//     - TotalNestLevel is the total number of loop nests
//     - Iter is whether to go forward or backward through ranks (i.e. the
//       iteration order for MDRangePolicy)
//   - ThreadAndVector determines whether both vector and thread parallelism is
//     in use
template <typename Rank, typename ExecSpace,
          TeamMDRangeThreadAndVector ThreadAndVector>
struct ThreadAndVectorNestLevel;

struct NoReductionTag {};

template <typename Rank, typename TeamMDPolicy, typename Lambda,
          typename ReductionValueType>
KOKKOS_INLINE_FUNCTION void md_parallel_impl(TeamMDPolicy const& policy,
                                             Lambda const& lambda,
                                             ReductionValueType&& val);
}  // namespace Impl

template <typename Rank, typename TeamHandle>
struct TeamThreadMDRange;

template <unsigned N, Iterate OuterDir, Iterate InnerDir, typename TeamHandle>
struct TeamThreadMDRange<Rank<N, OuterDir, InnerDir>, TeamHandle> {
  static_assert(N >= 2u, "Kokkos Error: TeamThreadMDRange requires rank >= 2");

  using NestLevelType  = int;
  using BoundaryType   = int;
  using TeamHandleType = TeamHandle;
  using ExecutionSpace = typename TeamHandleType::execution_space;
  using ArrayLayout    = typename ExecutionSpace::array_layout;

  static constexpr NestLevelType total_nest_level =
      Rank<N, OuterDir, InnerDir>::rank;
  static constexpr Iterate iter    = OuterDir;
  static constexpr auto par_thread = Impl::TeamMDRangeParThread::ParThread;
  static constexpr auto par_vector = Impl::TeamMDRangeParVector::NotParVector;

  static constexpr Iterate direction =
      OuterDir == Iterate::Default ? Impl::layout_iterate_type_selector<
                                         ArrayLayout>::outer_iteration_pattern
                                   : iter;

  template <class... Args>
  KOKKOS_FUNCTION TeamThreadMDRange(TeamHandleType const& team_, Args&&... args)
      : team(team_), boundaries{static_cast<BoundaryType>(args)...} {
    static_assert(sizeof...(Args) == total_nest_level);
  }

  TeamHandleType const& team;
  BoundaryType boundaries[total_nest_level];
};

template <typename TeamHandle, typename... Args>
KOKKOS_DEDUCTION_GUIDE TeamThreadMDRange(TeamHandle const&, Args&&...)
    -> TeamThreadMDRange<Rank<sizeof...(Args), Iterate::Default>, TeamHandle>;

template <typename Rank, typename TeamHandle>
struct ThreadVectorMDRange;

template <unsigned N, Iterate OuterDir, Iterate InnerDir, typename TeamHandle>
struct ThreadVectorMDRange<Rank<N, OuterDir, InnerDir>, TeamHandle> {
  static_assert(N >= 2u,
                "Kokkos Error: ThreadVectorMDRange requires rank >= 2");

  using NestLevelType  = int;
  using BoundaryType   = int;
  using TeamHandleType = TeamHandle;
  using ExecutionSpace = typename TeamHandleType::execution_space;
  using ArrayLayout    = typename ExecutionSpace::array_layout;

  static constexpr NestLevelType total_nest_level =
      Rank<N, OuterDir, InnerDir>::rank;
  static constexpr Iterate iter    = OuterDir;
  static constexpr auto par_thread = Impl::TeamMDRangeParThread::NotParThread;
  static constexpr auto par_vector = Impl::TeamMDRangeParVector::ParVector;

  static constexpr Iterate direction =
      OuterDir == Iterate::Default ? Impl::layout_iterate_type_selector<
                                         ArrayLayout>::outer_iteration_pattern
                                   : iter;

  template <class... Args>
  KOKKOS_INLINE_FUNCTION ThreadVectorMDRange(TeamHandleType const& team_,
                                             Args&&... args)
      : team(team_), boundaries{static_cast<BoundaryType>(args)...} {
    static_assert(sizeof...(Args) == total_nest_level);
  }

  TeamHandleType const& team;
  BoundaryType boundaries[total_nest_level];
};

template <typename TeamHandle, typename... Args>
KOKKOS_DEDUCTION_GUIDE ThreadVectorMDRange(TeamHandle const&, Args&&...)
    -> ThreadVectorMDRange<Rank<sizeof...(Args), Iterate::Default>, TeamHandle>;

template <typename Rank, typename TeamHandle>
struct TeamVectorMDRange;

template <unsigned N, Iterate OuterDir, Iterate InnerDir, typename TeamHandle>
struct TeamVectorMDRange<Rank<N, OuterDir, InnerDir>, TeamHandle> {
  static_assert(N >= 2u, "Kokkos Error: TeamVectorMDRange requires rank >= 2");

  using NestLevelType  = int;
  using BoundaryType   = int;
  using TeamHandleType = TeamHandle;
  using ExecutionSpace = typename TeamHandleType::execution_space;
  using ArrayLayout    = typename ExecutionSpace::array_layout;

  static constexpr NestLevelType total_nest_level =
      Rank<N, OuterDir, InnerDir>::rank;
  static constexpr Iterate iter    = OuterDir;
  static constexpr auto par_thread = Impl::TeamMDRangeParThread::ParThread;
  static constexpr auto par_vector = Impl::TeamMDRangeParVector::ParVector;

  static constexpr Iterate direction =
      iter == Iterate::Default ? Impl::layout_iterate_type_selector<
                                     ArrayLayout>::outer_iteration_pattern
                               : iter;

  template <class... Args>
  KOKKOS_INLINE_FUNCTION TeamVectorMDRange(TeamHandleType const& team_,
                                           Args&&... args)
      : team(team_), boundaries{static_cast<BoundaryType>(args)...} {
    static_assert(sizeof...(Args) == total_nest_level);
  }

  TeamHandleType const& team;
  BoundaryType boundaries[total_nest_level];
};

template <typename TeamHandle, typename... Args>
KOKKOS_DEDUCTION_GUIDE TeamVectorMDRange(TeamHandle const&, Args&&...)
    -> TeamVectorMDRange<Rank<sizeof...(Args), Iterate::Default>, TeamHandle>;

template <typename Rank, typename TeamHandle, typename Lambda,
          typename ReducerValueType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    TeamThreadMDRange<Rank, TeamHandle> const& policy, Lambda const& lambda,
    ReducerValueType& val) {
  static_assert(/*!Kokkos::is_view_v<ReducerValueType> &&*/
                !std::is_array_v<ReducerValueType> &&
                    !std::is_pointer_v<ReducerValueType> &&
                    !Kokkos::is_reducer_v<ReducerValueType>,
                "Only scalar return types are allowed!");

  val = ReducerValueType{};
  Impl::md_parallel_impl<Rank>(policy, lambda, val);
  policy.team.team_reduce(
      Kokkos::Sum<ReducerValueType, typename TeamHandle::execution_space>{val});
}

template <typename Rank, typename TeamHandle, typename Lambda>
KOKKOS_INLINE_FUNCTION void parallel_for(
    TeamThreadMDRange<Rank, TeamHandle> const& policy, Lambda const& lambda) {
  Impl::md_parallel_impl<Rank>(policy, lambda, Impl::NoReductionTag());
}

template <typename Rank, typename TeamHandle, typename Lambda,
          typename ReducerValueType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    ThreadVectorMDRange<Rank, TeamHandle> const& policy, Lambda const& lambda,
    ReducerValueType& val) {
  static_assert(/*!Kokkos::is_view_v<ReducerValueType> &&*/
                !std::is_array_v<ReducerValueType> &&
                    !std::is_pointer_v<ReducerValueType> &&
                    !Kokkos::is_reducer_v<ReducerValueType>,
                "Only a scalar return types are allowed!");

  val = ReducerValueType{};
  Impl::md_parallel_impl<Rank>(policy, lambda, val);
  if constexpr (false
#ifdef KOKKOS_ENABLE_CUDA
                || std::is_same_v<typename TeamHandle::execution_space,
                                  Kokkos::Cuda>
#elif defined(KOKKOS_ENABLE_HIP)
                || std::is_same_v<typename TeamHandle::execution_space,
                                  Kokkos::HIP>
#elif defined(KOKKOS_ENABLE_SYCL)
                || std::is_same_v<typename TeamHandle::execution_space,
                                  Kokkos::SYCL>
#endif
  )
    policy.team.vector_reduce(
        Kokkos::Sum<ReducerValueType, typename TeamHandle::execution_space>{
            val});
}

template <typename Rank, typename TeamHandle, typename Lambda>
KOKKOS_INLINE_FUNCTION void parallel_for(
    ThreadVectorMDRange<Rank, TeamHandle> const& policy, Lambda const& lambda) {
  Impl::md_parallel_impl<Rank>(policy, lambda, Impl::NoReductionTag());
}

template <typename Rank, typename TeamHandle, typename Lambda,
          typename ReducerValueType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    TeamVectorMDRange<Rank, TeamHandle> const& policy, Lambda const& lambda,
    ReducerValueType& val) {
  static_assert(/*!Kokkos::is_view_v<ReducerValueType> &&*/
                !std::is_array_v<ReducerValueType> &&
                    !std::is_pointer_v<ReducerValueType> &&
                    !Kokkos::is_reducer_v<ReducerValueType>,
                "Only a scalar return types are allowed!");

  val = ReducerValueType{};
  Impl::md_parallel_impl<Rank>(policy, lambda, val);
  if constexpr (false
#ifdef KOKKOS_ENABLE_CUDA
                || std::is_same_v<typename TeamHandle::execution_space,
                                  Kokkos::Cuda>
#elif defined(KOKKOS_ENABLE_HIP)
                || std::is_same_v<typename TeamHandle::execution_space,
                                  Kokkos::HIP>
#elif defined(KOKKOS_ENABLE_SYCL)
                || std::is_same_v<typename TeamHandle::execution_space,
                                  Kokkos::SYCL>
#endif
  )
    policy.team.vector_reduce(
        Kokkos::Sum<ReducerValueType, typename TeamHandle::execution_space>{
            val});
  policy.team.team_reduce(
      Kokkos::Sum<ReducerValueType, typename TeamHandle::execution_space>{val});
}

template <typename Rank, typename TeamHandle, typename Lambda>
KOKKOS_INLINE_FUNCTION void parallel_for(
    TeamVectorMDRange<Rank, TeamHandle> const& policy, Lambda const& lambda) {
  Impl::md_parallel_impl<Rank>(policy, lambda, Impl::NoReductionTag());
}

namespace Impl {

template <bool HasLabel>
struct OptionalLabelRef;

template <>
struct OptionalLabelRef<true> {
  static constexpr bool has_label = true;
  // Borrows the string, label must outlive this wrapper
  const std::string& get() const { return label_ref; }
  explicit OptionalLabelRef(const std::string& label) : label_ref(label) {}
  OptionalLabelRef(const std::string&&) = delete;

 private:
  const std::string& label_ref;
};

template <>
struct OptionalLabelRef<false> {
  static constexpr bool has_label = false;
};
using NoLabel   = OptionalLabelRef<false>;
using WithLabel = OptionalLabelRef<true>;

template <typename FunctorType, typename TagType,
          bool HasTag = !std::is_void_v<TagType>>
struct ParallelConstructName;

template <typename FunctorType, typename TagType>
struct ParallelConstructName<FunctorType, TagType, true> {
  ParallelConstructName(std::string const& label) : label_ref(label) {
    if (label.empty()) {
#ifdef KOKKOS_ENABLE_IMPL_TYPEINFO
      default_name =
          std::string(TypeInfo<std::remove_const_t<FunctorType>>::name()) +
          "/" + std::string(TypeInfo<TagType>::name());
#else
      default_name = std::string(typeid(FunctorType).name()) + "/" +
                     typeid(TagType).name();
#endif
    }
  }
  std::string const& get() {
    return (label_ref.empty()) ? default_name : label_ref;
  }
  std::string const& label_ref;
  std::string default_name;
};

template <typename FunctorType, typename TagType>
struct ParallelConstructName<FunctorType, TagType, false> {
  ParallelConstructName(std::string const& label) : label_ref(label) {
    if (label.empty()) {
#ifdef KOKKOS_ENABLE_IMPL_TYPEINFO
      default_name = TypeInfo<std::remove_const_t<FunctorType>>::name();
#else
      default_name = typeid(FunctorType).name();
#endif
    }
  }
  std::string const& get() {
    return (label_ref.empty()) ? default_name : label_ref;
  }
  std::string const& label_ref;
  std::string default_name;
};

}  // namespace Impl

}  // namespace Kokkos

namespace Kokkos {

namespace Impl {

template <class PatternTag, class... Args>
struct PatternImplSpecializationFromTag;

template <class... Args>
struct PatternImplSpecializationFromTag<Kokkos::ParallelForTag, Args...>
    : std::type_identity<ParallelFor<Args...>> {};

template <class... Args>
struct PatternImplSpecializationFromTag<Kokkos::ParallelReduceTag, Args...>
    : std::type_identity<ParallelReduce<Args...>> {};

template <class... Args>
struct PatternImplSpecializationFromTag<Kokkos::ParallelScanTag, Args...>
    : std::type_identity<ParallelScan<Args...>> {};

template <class PatternImpl>
struct PatternTagFromImplSpecialization;

template <class... Args>
struct PatternTagFromImplSpecialization<ParallelFor<Args...>>
    : std::type_identity<ParallelForTag> {};

template <class... Args>
struct PatternTagFromImplSpecialization<ParallelReduce<Args...>>
    : std::type_identity<ParallelReduceTag> {};

template <class... Args>
struct PatternTagFromImplSpecialization<ParallelScan<Args...>>
    : std::type_identity<ParallelScanTag> {};

}  // end namespace Impl

}  // namespace Kokkos

namespace Kokkos {
namespace Impl {

// Specialization of RangePolicy for defining work over a range of an integral
// type, split up among all resources of a thread team
template <TeamHandle Handle, class... Properties>
class ImplRangePolicy<Handle, Properties...>
    : public Impl::TeamVectorRangeBoundariesStruct<
          typename Impl::PolicyTraits<Properties...>::index_type, Handle> {
  using base_t = typename Impl::TeamVectorRangeBoundariesStruct<
      typename Impl::PolicyTraits<Properties...>::index_type, Handle>;

 public:
  using traits = typename Impl::PolicyTraits<Properties...>;
  static_assert(std::same_as<typename traits::execution_type, Handle>);

  using execution_policy = Kokkos::RangePolicy<Properties...>;
  using work_tag         = typename traits::work_tag;
  using member_type      = typename traits::index_type;
  using index_type       = typename traits::index_type;
  using base_t::base_t;

  KOKKOS_INLINE_FUNCTION Handle const& space() const {
    return static_cast<const base_t*>(this)->member;
  }

  KOKKOS_INLINE_FUNCTION member_type begin() const {
    return static_cast<const base_t*>(this)->start;
  }
  KOKKOS_INLINE_FUNCTION member_type end() const {
    return static_cast<const base_t*>(this)->end;
  }

  KOKKOS_INLINE_FUNCTION member_type chunk_size() const {
    // Chunk size has no meaning currently in this specialization.
    // Returning 1 to allow self-similar code using RangePolicy<ExecSpace> to
    // query chunk_size.
    return 1;
  }
};

template <Kokkos::Experimental::ThreadHandle Handle, class... Properties>
class ImplRangePolicy<Handle, Properties...>
    : public Impl::ThreadVectorRangeBoundariesStruct<
          typename Impl::PolicyTraits<Properties...>::index_type,
          typename Handle::member_type> {
  using base_t = typename Impl::ThreadVectorRangeBoundariesStruct<
      typename Impl::PolicyTraits<Properties...>::index_type,
      typename Handle::member_type>;

 public:
  using traits = typename Impl::PolicyTraits<Properties...>;
  static_assert(std::same_as<typename traits::execution_type, Handle>);

  using execution_policy = Kokkos::RangePolicy<Properties...>;
  using work_tag         = typename traits::work_tag;
  using member_type      = typename traits::index_type;
  using index_type       = typename traits::index_type;

 private:
  Handle m_handle;

 public:
  template <typename IndexType1, typename IndexType2>
  KOKKOS_INLINE_FUNCTION ImplRangePolicy(Handle const& handle,
                                         IndexType1 work_begin,
                                         IndexType2 work_end)
      : base_t(handle.team_member, static_cast<index_type>(work_begin),
               static_cast<index_type>(work_end)),
        m_handle(handle) {}

  template <typename IndexType>
  KOKKOS_INLINE_FUNCTION ImplRangePolicy(Handle const& handle,
                                         IndexType work_count)
      : base_t(handle.team_member, static_cast<index_type>(work_count)),
        m_handle(handle) {}

  KOKKOS_INLINE_FUNCTION Handle const& space() const { return m_handle; }

  KOKKOS_INLINE_FUNCTION member_type begin() const {
    return static_cast<const base_t*>(this)->start;
  }
  KOKKOS_INLINE_FUNCTION member_type end() const {
    return static_cast<const base_t*>(this)->end;
  }

  KOKKOS_INLINE_FUNCTION member_type chunk_size() const {
    // Same rationale as ImplRangePolicy<TeamHandle, ...>::chunk_size().
    return 1;
  }
};
}  // namespace Impl

/** \brief  Execution policy for work over a range of an integral type.
 *
 * RangePolicy has partial specializations for an execution space, a team
 * handle, and a thread handle: they parallelize over an execution space, over
 * a thread team (TeamVectorRange), and within a team thread
 * (ThreadVectorRange), respectively.
 *
 * Valid template argument options:
 *
 *  With a specified execution space:
 *    < ExecSpace , WorkTag , { IntConst | IntType } >
 *    < ExecSpace , WorkTag , void >
 *    < ExecSpace , { IntConst | IntType } , void >
 *    < ExecSpace , void , void >
 *
 * With a specified team handle:
 *    < TeamHandle , void >
 *    < TeamHandle , IntType >
 *
 *  Without specifying an execution type, default behavior is using
 * DefaultExecutionSpace with the following template arguments: < WorkTag , {
 * IntConst | IntType } , void > < WorkTag , void , void > < { IntConst |
 * IntType } , void , void > < void , void , void >
 *
 *  IntType  is a fundamental integral type
 *  IntConst is an Impl::integral_constant< IntType , Blocking >
 *
 *  Blocking is the granularity of partitioning the range among threads.
 */
template <typename... Properties>
class RangePolicy
    : public Impl::ImplRangePolicy<
          typename Impl::PolicyTraits<Properties...>::execution_type,
          Properties...> {
 public:
  using execution_type =
      typename Impl::PolicyTraits<Properties...>::execution_type;
  using base_t = Impl::ImplRangePolicy<execution_type, Properties...>;
  using base_t::base_t;

  // Set chunk size and return the policy. For team handle specialization, this
  // is a no-op.
  KOKKOS_INLINE_FUNCTION RangePolicy& set_chunk_size(
      [[maybe_unused]] int chunk_size) {
    if constexpr (ExecutionSpace<execution_type>) {
      KOKKOS_IF_ON_HOST(
          static_cast<base_t*>(this)->impl_set_chunk_size(chunk_size);)
    }
    return *this;
  }
};

namespace Impl {
// Helper concept for capturing exec space and all handle types
template <class ExecType>
concept ExecutionTypeConcept =
    ExecutionSpace<ExecType> || TeamHandle<ExecType> ||
    Kokkos::Experimental::ThreadHandle<ExecType>;
}  // namespace Impl

// Deduction guide

// Instances for the execution space specialization
RangePolicy() -> RangePolicy<>;
RangePolicy(int64_t, int64_t) -> RangePolicy<>;
RangePolicy(int64_t, int64_t, ChunkSize const&) -> RangePolicy<>;
RangePolicy(const DefaultExecutionSpace&, int64_t, int64_t, ChunkSize const&)
    -> RangePolicy<DefaultExecutionSpace>;
template <Impl::ExecutionTypeConcept Exec>
RangePolicy(const Exec&, int64_t, int64_t, ChunkSize const&)
    -> RangePolicy<Exec>;

// Instances for both execution space and team handle specializations.
// Must be callable on device.
KOKKOS_DEDUCTION_GUIDE RangePolicy(const DefaultExecutionSpace&, int64_t,
                                   int64_t)
    -> RangePolicy<DefaultExecutionSpace>;
template <Impl::ExecutionTypeConcept Exec>
KOKKOS_DEDUCTION_GUIDE RangePolicy(const Exec&, int64_t, int64_t)
    -> RangePolicy<Exec>;

}  // namespace Kokkos

#endif /* #define KOKKOS_EXECPOLICY_HPP */
