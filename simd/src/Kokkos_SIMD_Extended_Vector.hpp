// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_SIMD_EXTENDED_VECTOR_HPP
#define KOKKOS_SIMD_EXTENDED_VECTOR_HPP

#include <Kokkos_SIMD_Common.hpp>
#include <Kokkos_SIMD_Scalar.hpp>
#include <Kokkos_Macros.hpp>

#if defined(KOKKOS_ARCH_AVX) && !defined(__AVX__)
#error "__AVX__ must be defined for KOKKOS_ARCH_AVX"
#endif

#if defined(KOKKOS_ARCH_AVX2)
#if !defined(__AVX2__)
#error "__AVX2__ must be defined for KOKKOS_ARCH_AVX2"
#endif
#include <Kokkos_SIMD_AVX2.hpp>
#endif

#if defined(KOKKOS_ARCH_AVX512XEON)
#if !defined(__AVX512F__)
#error "__AVX512F__ must be defined for KOKKOS_ARCH_AVX512XEON"
#endif
#include <Kokkos_SIMD_AVX512.hpp>
#endif

#if defined(KOKKOS_ARCH_ARM_SVE)
#if !defined(__ARM_FEATURE_SVE_BITS) || !defined(__ARM_NEON)
#error \
    "Both __ARM_FEATURE_SVE_BITS and __ARM_NEON must be definded for KOKKOS_ARCH_ARM_SVE"
#endif
#include <Kokkos_SIMD_SVE.hpp>
#endif

#if defined(KOKKOS_ARCH_ARM_NEON)
#if !defined(__ARM_NEON)
#error "__ARM_NEON must be defined for KOKKOS_ARCH_ARM_NEON"
#endif
#include <Kokkos_SIMD_NEON.hpp>
#endif

namespace Kokkos {
namespace Experimental {
namespace simd_abi {
template <class Abi, Kokkos::Experimental::Impl::simd_size_t N>
class extended_abi {
 public:
  using underlying_abi = Abi;
};
}  // namespace simd_abi

namespace Impl {
template <int It, int Limit, class Functor>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void unroll(Functor&& f) {
  if constexpr (It < Limit) {
    f(std::integral_constant<int, It>{});
    unroll<It + 1, Limit>(f);
  }
}

template <int Limit, class Functor>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void unroll(Functor&& f) {
  unroll<0, Limit>(f);
}
}  // namespace Impl

template <class T, class Abi, Impl::simd_size_t N>
class basic_simd_mask<T, simd_abi::extended_abi<Abi, N>> {
 private:
  using mask_type = basic_simd_mask<T, Abi>;
  static_assert(N % mask_type::size() == 0,
                "Extended simd's size must be a multiple of the underlying "
                "simd type's size");

  static constexpr int nb_vectors = N / mask_type::size();
  std::array<mask_type, nb_vectors> m_values;

 protected:
  using implementation_type = std::array<mask_type, nb_vectors>;

 public:
  using value_type = bool;
  using abi_type   = simd_abi::extended_abi<Abi, N>;

  static constexpr std::integral_constant<Impl::simd_size_t,
                                          nb_vectors * mask_type::size()>
      size{};

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask() noexcept = default;

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      value_type value) noexcept {
    for (int i = 0; i < nb_vectors; i++) {
      m_values[i] = mask_type(value);
    }
  }
  template <class U>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      basic_simd_mask<U, abi_type> const& other) noexcept
      : basic_simd_mask([&](Impl::simd_size_t i) { return other[i]; }) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      std::array<mask_type, nb_vectors> const& value_in) noexcept
      : m_values(value_in) {}
  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      G&& gen) noexcept {
    Impl::unroll<nb_vectors>(
        [&]<int VecIdx>(std::integral_constant<int, VecIdx>) {
          m_values[VecIdx] =
              mask_type([&]<Impl::simd_size_t I>(
                            std::integral_constant<Impl::simd_size_t, I>) {
                return gen(
                    std::integral_constant<Impl::simd_size_t,
                                           I + VecIdx * mask_type::size()>{});
              });
        });
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](Impl::simd_size_t i) const {
    return m_values[i / mask_type::size()][i % mask_type::size()];
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator!() const noexcept {
    return operator~();
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit
  operator std::array<mask_type, nb_vectors>() const noexcept {
    return m_values;
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator~() const noexcept {
    std::array<mask_type, nb_vectors> res;
    Impl::unroll<nb_vectors>(
        [&]<int I>(std::integral_constant<int, I>) { res[I] = ~m_values[I]; });
    return basic_simd_mask(res);
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend basic_simd_mask operator&&(
      basic_simd_mask const& lhs, basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(lhs & rhs);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend basic_simd_mask operator||(
      basic_simd_mask const& lhs, basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(lhs | rhs);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend basic_simd_mask operator&(
      basic_simd_mask const& lhs, basic_simd_mask const& rhs) noexcept {
    std::array<mask_type, nb_vectors> res;
    Impl::unroll<nb_vectors>([&]<int I>(std::integral_constant<int, I>) {
      res[I] = lhs.m_values[I] & rhs.m_values[I];
    });
    return basic_simd_mask(res);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend basic_simd_mask operator|(
      basic_simd_mask const& lhs, basic_simd_mask const& rhs) noexcept {
    std::array<mask_type, nb_vectors> res;
    Impl::unroll<nb_vectors>([&]<int I>(std::integral_constant<int, I>) {
      res[I] = lhs.m_values[I] | rhs.m_values[I];
    });
    return basic_simd_mask(res);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend basic_simd_mask operator^(
      basic_simd_mask const& lhs, basic_simd_mask const& rhs) noexcept {
    std::array<mask_type, nb_vectors> res;
    Impl::unroll<nb_vectors>([&]<int I>(std::integral_constant<int, I>) {
      res[I] = lhs.m_values[I] ^ rhs.m_values[I];
    });
    return basic_simd_mask(res);
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend basic_simd_mask operator==(
      basic_simd_mask const& lhs, basic_simd_mask const& rhs) noexcept {
    std::array<mask_type, nb_vectors> res;
    Impl::unroll<nb_vectors>([&]<int I>(std::integral_constant<int, I>) {
      res[I] = lhs.m_values[I] == rhs.m_values[I];
    });
    return basic_simd_mask(res);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend basic_simd_mask operator!=(
      basic_simd_mask const& lhs, basic_simd_mask const& rhs) noexcept {
    return !operator==(lhs, rhs);
  }
};

template <class T, class Abi, Impl::simd_size_t N>
class basic_simd<T, simd_abi::extended_abi<Abi, N>> {
 private:
  using simd_type = basic_simd<T, Abi>;
  static_assert(N % simd_type::size() == 0,
                "Extended simd's size must be a multiple of the underlying "
                "simd type's size");

  static constexpr int nb_vectors = N / simd_type::size();
  std::array<simd_type, nb_vectors> m_values;

 protected:
  using implementation_type = std::array<simd_type, nb_vectors>;

 public:
  using value_type = T;
  using abi_type   = simd_abi::extended_abi<Abi, N>;
  using mask_type  = basic_simd_mask<T, abi_type>;

  static constexpr std::integral_constant<Impl::simd_size_t,
                                          nb_vectors * simd_type::size()>
      size{};

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd() noexcept = default;
  template <class U>
    requires std::convertible_to<U, value_type>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd(const U& value) noexcept {
    Impl::unroll<nb_vectors>([&]<int I>(std::integral_constant<int, I>) {
      m_values[I] = simd_type(value);
    });
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      std::array<simd_type, nb_vectors> const& value_in) noexcept
      : m_values(value_in) {}
  template <typename U>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION explicit(
      Impl::needs_explicit_conversion_v<U, value_type>)
      basic_simd(basic_simd<U, abi_type> const& other) noexcept {
    using other_simd_type = basic_simd<U, Abi>;

    const std::array<other_simd_type, nb_vectors> other_vecs(other);
    Impl::unroll<nb_vectors>([&]<int I>(std::integral_constant<int, I>) {
      m_values[I] = simd_type(other_vecs[I]);
    });
  }
  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      G&& gen) noexcept {
    Impl::unroll<nb_vectors>(
        [&]<int VecIdx>(std::integral_constant<int, VecIdx>) {
          m_values[VecIdx] =
              simd_type([&]<Impl::simd_size_t I>(
                            std::integral_constant<Impl::simd_size_t, I>) {
                return gen(
                    std::integral_constant<Impl::simd_size_t,
                                           I + VecIdx * simd_type::size()>{});
              });
        });
  }
  template <typename... Flags>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, simd_flags<Flags...> flags = {}) noexcept {
    Impl::unroll<nb_vectors>([&]<int I>(std::integral_constant<int, I>) {
      m_values[I] = simd_type(ptr + I * simd_type::size(), flags);
    });
  }
  template <typename... Flags>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, mask_type const& mask,
      simd_flags<Flags...> flag = {}) noexcept {
    const std::array<typename simd_type::mask_type, nb_vectors> masks(mask);
    Impl::unroll<nb_vectors>([&]<int I>(std::integral_constant<int, I>) {
      m_values[I] = simd_type(ptr + I * simd_type::size(), masks[I], flag);
    });
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](Impl::simd_size_t i) const {
    return m_values[i / simd_type::size()][i % simd_type::size()];
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit
  operator std::array<simd_type, nb_vectors>() const noexcept {
    return m_values;
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd operator-() const noexcept {
    std::array<simd_type, nb_vectors> res;
    Impl::unroll<nb_vectors>(
        [&]<int I>(std::integral_constant<int, I>) { res[I] = -m_values[I]; });
    return basic_simd(res);
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd operator~()

      const noexcept
    requires(std::is_integral_v<value_type>)
  {
    std::array<simd_type, nb_vectors> res;
    Impl::unroll<nb_vectors>(
        [&]<int I>(std::integral_constant<int, I>) { res[I] = ~m_values[I]; });
    return basic_simd(res);
  }

#define KOKKOS_IMPL_EXTENDED_SIMD_BINARY_OPERATOR(OP, SYMBOL, REQUIREMENTS) \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend basic_simd OP(               \
      basic_simd const& lhs, basic_simd const& rhs) noexcept REQUIREMENTS { \
    std::array<simd_type, nb_vectors> res;                                  \
    Impl::unroll<nb_vectors>([&]<int I>(std::integral_constant<int, I>) {   \
      res[I] = lhs.m_values[I] SYMBOL rhs.m_values[I];                      \
    });                                                                     \
    return basic_simd(res);                                                 \
  }

  KOKKOS_IMPL_EXTENDED_SIMD_BINARY_OPERATOR(operator+, +, )
  KOKKOS_IMPL_EXTENDED_SIMD_BINARY_OPERATOR(operator-, -, )
  KOKKOS_IMPL_EXTENDED_SIMD_BINARY_OPERATOR(operator*, *, )
  KOKKOS_IMPL_EXTENDED_SIMD_BINARY_OPERATOR(operator/, /, )
  KOKKOS_IMPL_EXTENDED_SIMD_BINARY_OPERATOR(operator&, &,
                                            requires(
                                                std::is_integral_v<value_type>))
  KOKKOS_IMPL_EXTENDED_SIMD_BINARY_OPERATOR(operator|, |,
                                            requires(
                                                std::is_integral_v<value_type>))
  KOKKOS_IMPL_EXTENDED_SIMD_BINARY_OPERATOR(operator^, ^,
                                            requires(
                                                std::is_integral_v<value_type>))
  KOKKOS_IMPL_EXTENDED_SIMD_BINARY_OPERATOR(operator<<, <<,
                                            requires(
                                                std::is_integral_v<value_type>))
  KOKKOS_IMPL_EXTENDED_SIMD_BINARY_OPERATOR(operator>>, >>,
                                            requires(
                                                std::is_integral_v<value_type>))
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend basic_simd operator<<(
      basic_simd const& lhs, Impl::simd_size_t rhs) noexcept
    requires(std::is_integral_v<value_type>)
  {
    std::array<simd_type, nb_vectors> res;
    Impl::unroll<nb_vectors>([&]<int I>(std::integral_constant<int, I>) {
      res[I] = lhs.m_values[I] << rhs;
    });
    return basic_simd(res);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend basic_simd operator>>(
      basic_simd const& lhs, Impl::simd_size_t rhs) noexcept
    requires(std::is_integral_v<value_type>)
  {
    std::array<simd_type, nb_vectors> res;
    Impl::unroll<nb_vectors>([&]<int I>(std::integral_constant<int, I>) {
      res[I] = lhs.m_values[I] >> rhs;
    });
    return basic_simd(res);
  }

#define KOKKOS_IMPL_EXTENDED_SIMD_MASK_BINARY_OPERATOR(OP, SYMBOL)        \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type OP(              \
      basic_simd const& lhs, basic_simd const& rhs) noexcept {            \
    std::array<typename simd_type::mask_type, nb_vectors> res;            \
    Impl::unroll<nb_vectors>([&]<int I>(std::integral_constant<int, I>) { \
      res[I] = lhs.m_values[I] SYMBOL rhs.m_values[I];                    \
    });                                                                   \
    return mask_type(res);                                                \
  }
  KOKKOS_IMPL_EXTENDED_SIMD_MASK_BINARY_OPERATOR(operator==, ==)
  KOKKOS_IMPL_EXTENDED_SIMD_MASK_BINARY_OPERATOR(operator!=, !=)
  KOKKOS_IMPL_EXTENDED_SIMD_MASK_BINARY_OPERATOR(operator>=, >=)
  KOKKOS_IMPL_EXTENDED_SIMD_MASK_BINARY_OPERATOR(operator<=, <=)
  KOKKOS_IMPL_EXTENDED_SIMD_MASK_BINARY_OPERATOR(operator>, >)
  KOKKOS_IMPL_EXTENDED_SIMD_MASK_BINARY_OPERATOR(operator<, <)

#undef KOKKOS_IMPL_EXTENDED_SIMD_BINARY_OPERATOR
#undef KOKKOS_IMPL_EXTENDED_SIMD_MASK_BINARY_OPERATOR
};
}  // namespace Experimental

#define KOKKOS_IMPL_EXTENDED_SIMD_UNARY_OPERATOR(func)                           \
  template <class T, class Abi, Experimental::Impl::simd_size_t N>               \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::basic_simd<                \
      decltype(Kokkos::func(std::declval<T>())),                                 \
      Experimental::simd_abi::extended_abi<Abi, N>>                              \
  func(Experimental::basic_simd<                                                 \
       T, Experimental::simd_abi::extended_abi<Abi, N>> const& a) noexcept {     \
    using result_value_type = decltype(Kokkos::func(std::declval<T>()));         \
    using vector_type       = Experimental::basic_simd<                          \
        result_value_type, Experimental::simd_abi::extended_abi<Abi, N>>;  \
    using simd_type          = Experimental::basic_simd<T, Abi>;                 \
    using result_simd_type   = Experimental::basic_simd<result_value_type, Abi>; \
    constexpr int nb_vectors = vector_type::size() / simd_type::size();          \
                                                                                 \
    std::array<result_simd_type, nb_vectors> res;                                \
    const std::array<simd_type, nb_vectors> vec_a(a);                            \
    Experimental::Impl::unroll<nb_vectors>(                                      \
        [&]<int I>(std::integral_constant<int, I>) {                             \
          res[I] = Kokkos::func(vec_a[I]);                                       \
        });                                                                      \
    return vector_type(res);                                                     \
  }

#define KOKKOS_IMPL_EXTENDED_SIMD_BINARY_OPERATOR(func)                 \
  template <class T, class Abi, Experimental::Impl::simd_size_t N>      \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::basic_simd<       \
      T, Experimental::simd_abi::extended_abi<Abi, N>>                  \
  func(Experimental::basic_simd<                                        \
           T, Experimental::simd_abi::extended_abi<Abi, N>> const& a,   \
       Experimental::basic_simd<                                        \
           T, Experimental::simd_abi::extended_abi<Abi, N>> const&      \
           b) noexcept {                                                \
    using vector_type = Experimental::basic_simd<                       \
        T, Experimental::simd_abi::extended_abi<Abi, N>>;               \
    using simd_type          = Experimental::basic_simd<T, Abi>;        \
    constexpr int nb_vectors = vector_type::size() / simd_type::size(); \
                                                                        \
    std::array<simd_type, nb_vectors> res;                              \
    const std::array<simd_type, nb_vectors> vec_a(a);                   \
    const std::array<simd_type, nb_vectors> vec_b(b);                   \
    Experimental::Impl::unroll<nb_vectors>(                             \
        [&]<int I>(std::integral_constant<int, I>) {                    \
          res[I] = Kokkos::func(vec_a[I], vec_b[I]);                    \
        });                                                             \
    return vector_type(res);                                            \
  }

#define KOKKOS_IMPL_EXTENDED_SIMD_TERNARY_OPERATOR(func)                \
  template <class T, class Abi, Experimental::Impl::simd_size_t N>      \
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::basic_simd<       \
      T, Experimental::simd_abi::extended_abi<Abi, N>>                  \
  func(Experimental::basic_simd<                                        \
           T, Experimental::simd_abi::extended_abi<Abi, N>> const& a,   \
       Experimental::basic_simd<                                        \
           T, Experimental::simd_abi::extended_abi<Abi, N>> const& b,   \
       Experimental::basic_simd<                                        \
           T, Experimental::simd_abi::extended_abi<Abi, N>> const&      \
           c) noexcept {                                                \
    using vector_type = Experimental::basic_simd<                       \
        T, Experimental::simd_abi::extended_abi<Abi, N>>;               \
    using simd_type          = Experimental::basic_simd<T, Abi>;        \
    constexpr int nb_vectors = vector_type::size() / simd_type::size(); \
                                                                        \
    std::array<simd_type, nb_vectors> res;                              \
    const std::array<simd_type, nb_vectors> vec_a(a);                   \
    const std::array<simd_type, nb_vectors> vec_b(b);                   \
    const std::array<simd_type, nb_vectors> vec_c(c);                   \
    Experimental::Impl::unroll<nb_vectors>(                             \
        [&]<int I>(std::integral_constant<int, I>) {                    \
          res[I] = Kokkos::func(vec_a[I], vec_b[I], vec_c[I]);          \
        });                                                             \
    return vector_type(res);                                            \
  }

KOKKOS_IMPL_EXTENDED_SIMD_BINARY_OPERATOR(copysign)
KOKKOS_IMPL_EXTENDED_SIMD_UNARY_OPERATOR(abs)
// KOKKOS_IMPL_EXTENDED_SIMD_UNARY_OPERATOR(floor)
// KOKKOS_IMPL_EXTENDED_SIMD_UNARY_OPERATOR(ceil)
// KOKKOS_IMPL_EXTENDED_SIMD_UNARY_OPERATOR(round)
// KOKKOS_IMPL_EXTENDED_SIMD_UNARY_OPERATOR(trunc)
KOKKOS_IMPL_EXTENDED_SIMD_UNARY_OPERATOR(sqrt)
KOKKOS_IMPL_EXTENDED_SIMD_TERNARY_OPERATOR(fma)
KOKKOS_IMPL_EXTENDED_SIMD_BINARY_OPERATOR(max)
KOKKOS_IMPL_EXTENDED_SIMD_BINARY_OPERATOR(min)

#undef KOKKOS_IMPL_EXTENDED_SIMD_UNARY_OPERATOR
#undef KOKKOS_IMPL_EXTENDED_SIMD_BINARY_OPERATOR
#undef KOKKOS_IMPL_EXTENDED_SIMD_TERNARY_OPERATOR

namespace Experimental {
namespace Impl {
template <class T>
constexpr bool is_extended_abi_v = false;

template <class Abi, simd_size_t N>
constexpr bool
    is_extended_abi_v<Kokkos::Experimental::simd_abi::extended_abi<Abi, N>> =
        true;
}  // namespace Impl

template <typename SimdType, typename... Flags>
  requires Impl::is_extended_abi_v<typename SimdType::abi_type>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION SimdType
simd_unchecked_load(const typename SimdType::value_type* ptr,
                    simd_flags<Flags...> flag = simd_flag_default) {
  return SimdType(ptr, flag);
}

template <class T, class Abi, Impl::simd_size_t N, typename... Flags>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<T, simd_abi::extended_abi<Abi, N>>
    simd_unchecked_load(
        const T* ptr,
        basic_simd_mask<T, simd_abi::extended_abi<Abi, N>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<T, simd_abi::extended_abi<Abi, N>>(ptr, mask, flag);
}

template <typename SimdType, typename... Flags>
  requires Impl::is_extended_abi_v<typename SimdType::abi_type>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION SimdType
simd_unchecked_load(const typename SimdType::value_type* ptr,
                    typename SimdType::mask_type const& mask,
                    simd_flags<Flags...> flag = simd_flag_default) {
  return SimdType(ptr, mask, flag);
}

template <class T, class Abi, Impl::simd_size_t N, typename... Flags>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<T, simd_abi::extended_abi<Abi, N>>
    simd_partial_load(
        const T* ptr,
        basic_simd_mask<T, simd_abi::extended_abi<Abi, N>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<T, simd_abi::extended_abi<Abi, N>>(ptr, mask, flag);
}

template <typename SimdType, typename... Flags>
  requires Impl::is_extended_abi_v<typename SimdType::abi_type>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION SimdType
simd_partial_load(const typename SimdType::value_type* ptr,
                  typename SimdType::mask_type const& mask,
                  simd_flags<Flags...> flag = simd_flag_default) {
  return SimdType(ptr, mask, flag);
}

template <class T, class Abi, Impl::simd_size_t N, typename... Flags>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void simd_unchecked_store(
    basic_simd<T, simd_abi::extended_abi<Abi, N>> const& simd, T* ptr,
    simd_flags<Flags...> flag = simd_flag_default) {
  using vector_type        = basic_simd<T, simd_abi::extended_abi<Abi, N>>;
  using simd_type          = basic_simd<T, Abi>;
  constexpr int nb_vectors = vector_type::size() / simd_type::size();

  const std::array<simd_type, nb_vectors> vectors(simd);
  Impl::unroll<nb_vectors>([&]<int I>(std::integral_constant<int, I>) {
    simd_unchecked_store(vectors[I], ptr + I * simd_type::size(), flag);
  });
}

template <class T, class Abi, Impl::simd_size_t N, typename... Flags>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void simd_unchecked_store(
    basic_simd<T, simd_abi::extended_abi<Abi, N>> const& simd, T* ptr,
    basic_simd_mask<T, simd_abi::extended_abi<Abi, N>> const& mask,
    simd_flags<Flags...> flag = simd_flag_default) {
  using vector_type        = basic_simd<T, simd_abi::extended_abi<Abi, N>>;
  using simd_type          = basic_simd<T, Abi>;
  using mask_type          = simd_type::mask_type;
  constexpr int nb_vectors = vector_type::size() / simd_type::size();

  const std::array<simd_type, nb_vectors> vectors(simd);
  const std::array<mask_type, nb_vectors> masks(mask);
  Impl::unroll<nb_vectors>([&]<int I>(std::integral_constant<int, I>) {
    simd_unchecked_store(vectors[I], ptr + I * simd_type::size(), masks[I],
                         flag);
  });
}

template <class T, class Abi, Impl::simd_size_t N, typename... Flags>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void simd_partial_store(
    basic_simd<T, simd_abi::extended_abi<Abi, N>> const& simd, T* ptr,
    basic_simd_mask<T, simd_abi::extended_abi<Abi, N>> const& mask,
    simd_flags<Flags...> flag = simd_flag_default) {
  using vector_type        = basic_simd<T, simd_abi::extended_abi<Abi, N>>;
  using simd_type          = basic_simd<T, Abi>;
  using mask_type          = simd_type::mask_type;
  constexpr int nb_vectors = vector_type::size() / simd_type::size();

  const std::array<simd_type, nb_vectors> vectors(simd);
  const std::array<mask_type, nb_vectors> masks(mask);
  Impl::unroll<nb_vectors>([&]<int I>(std::integral_constant<int, I>) {
    simd_unchecked_store(vectors[I], ptr + I * simd_type::size(), masks[I],
                         flag);
  });
}

template <class T, class Abi, Impl::simd_size_t N>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<T, simd_abi::extended_abi<Abi, N>>
    condition(basic_simd_mask<T, simd_abi::extended_abi<Abi, N>> const& a,
              basic_simd<T, simd_abi::extended_abi<Abi, N>> const& b,
              basic_simd<T, simd_abi::extended_abi<Abi, N>> const& c) {
  using vector_type        = basic_simd<T, simd_abi::extended_abi<Abi, N>>;
  using simd_type          = basic_simd<T, Abi>;
  using mask_type          = simd_type::mask_type;
  constexpr int nb_vectors = vector_type::size() / simd_type::size();

  std::array<simd_type, nb_vectors> res;
  const std::array<simd_type, nb_vectors> vec_b(b);
  const std::array<simd_type, nb_vectors> vec_c(c);
  const std::array<mask_type, nb_vectors> masks(a);
  Impl::unroll<nb_vectors>([&]<int I>(std::integral_constant<int, I>) {
    res[I] = condition(masks[I], vec_b[I], vec_c[I]);
  });

  return vector_type(res);
}

template <Impl::SimdVecType V, Impl::Ranges::contiguous_range R,
          Impl::SimdIntegral I, typename... Flags>
  requires Impl::Ranges::sized_range<R> &&
           Impl::is_extended_abi_v<typename V::abi_type> &&
           std::is_same_v<typename V::abi_type, typename I::abi_type>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr V unchecked_gather_from(
    R&& in, const I& indices, simd_flags<Flags...> flag = simd_flag_default) {
  using simd_type =
      basic_simd<typename V::value_type, typename V::abi_type::underlying_abi>;
  using simd_index_type =
      basic_simd<typename I::value_type, typename I::abi_type::underlying_abi>;
  constexpr int nb_vectors = V::size() / simd_type::size();

  std::array<simd_type, nb_vectors> res;
  const std::array<simd_index_type, nb_vectors> vec_indices(indices);
  Impl::unroll<nb_vectors>([&]<int Idx>(std::integral_constant<int, Idx>) {
    res[Idx] = unchecked_gather_from<simd_type>(in, vec_indices[Idx], flag);
  });

  return V(res);
}

template <Impl::SimdVecType V, Impl::Ranges::contiguous_range R,
          Impl::SimdIntegral I, typename... Flags>
  requires Impl::Ranges::sized_range<R> &&
           Impl::is_extended_abi_v<typename V::abi_type> &&
           std::is_same_v<typename V::abi_type, typename I::abi_type>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr V partial_gather_from(
    R&& in, const I& indices, simd_flags<Flags...> flag = simd_flag_default) {
  return unchecked_gather_from<V>(in, indices, flag);
}

template <Impl::SimdVecType V, Impl::Ranges::contiguous_range R,
          Impl::SimdIntegral I, typename... Flags>
  requires Impl::Ranges::sized_range<R> &&
           Impl::is_extended_abi_v<typename V::abi_type> &&
           std::is_same_v<typename V::abi_type, typename I::abi_type>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr V unchecked_gather_from(
    R&& in, const typename I::mask_type& mask, const I& indices,
    simd_flags<Flags...> flag = simd_flag_default) {
  using simd_type =
      basic_simd<typename V::value_type, typename V::abi_type::underlying_abi>;
  using simd_index_type =
      basic_simd<typename I::value_type, typename I::abi_type::underlying_abi>;
  using mask_type          = simd_index_type::mask_type;
  constexpr int nb_vectors = V::size() / simd_type::size();

  std::array<simd_type, nb_vectors> res;
  const std::array<simd_index_type, nb_vectors> vec_indices(indices);
  const std::array<mask_type, nb_vectors> vec_masks(mask);
  Impl::unroll<nb_vectors>([&]<int Idx>(std::integral_constant<int, Idx>) {
    res[Idx] = unchecked_gather_from<simd_type>(in, vec_masks[Idx],
                                                vec_indices[Idx], flag);
  });

  return V(res);
}

template <Impl::SimdVecType V, Impl::Ranges::contiguous_range R,
          Impl::SimdIntegral I, typename... Flags>
  requires Impl::Ranges::sized_range<R> &&
           Impl::is_extended_abi_v<typename V::abi_type> &&
           std::is_same_v<typename V::abi_type, typename I::abi_type>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr V partial_gather_from(
    R&& in, const typename I::mask_type& mask, const I& indices,
    simd_flags<Flags...> flag = simd_flag_default) {
  return unchecked_gather_from<V>(in, mask, indices, flag);
}

template <Impl::SimdVecType V, Impl::Ranges::contiguous_range R,
          Impl::SimdIntegral I, typename... Flags>
  requires Impl::Ranges::sized_range<R> &&
           Impl::is_extended_abi_v<typename V::abi_type> &&
           std::is_same_v<typename V::abi_type, typename I::abi_type>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr void unchecked_scatter_to(
    const V& v, R&& out, const I& indices,
    simd_flags<Flags...> flag = simd_flag_default) {
  using simd_type =
      basic_simd<typename V::value_type, typename V::abi_type::underlying_abi>;
  using simd_index_type =
      basic_simd<typename I::value_type, typename I::abi_type::underlying_abi>;
  constexpr int nb_vectors = V::size() / simd_type::size();

  const std::array<simd_type, nb_vectors> vec_v(v);
  const std::array<simd_index_type, nb_vectors> vec_indices(indices);
  Impl::unroll<nb_vectors>([&]<int Idx>(std::integral_constant<int, Idx>) {
    unchecked_scatter_to(vec_v[Idx], out, vec_indices[Idx], flag);
  });
}

template <Impl::SimdVecType V, Impl::Ranges::contiguous_range R,
          Impl::SimdIntegral I, typename... Flags>
  requires Impl::Ranges::sized_range<R> &&
           Impl::is_extended_abi_v<typename V::abi_type> &&
           std::is_same_v<typename V::abi_type, typename I::abi_type>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr void partial_scatter_to(
    const V& v, R&& out, const I& indices,
    simd_flags<Flags...> flag = simd_flag_default) {
  unchecked_scatter_to(v, out, indices, flag);
}

template <Impl::SimdVecType V, Impl::Ranges::contiguous_range R,
          Impl::SimdIntegral I, typename... Flags>
  requires Impl::Ranges::sized_range<R> &&
           Impl::is_extended_abi_v<typename V::abi_type> &&
           std::is_same_v<typename V::abi_type, typename I::abi_type>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr void unchecked_scatter_to(
    const V& v, R&& out, const typename I::mask_type& mask, const I& indices,
    simd_flags<Flags...> flag = simd_flag_default) {
  using simd_type =
      basic_simd<typename V::value_type, typename V::abi_type::underlying_abi>;
  using simd_index_type =
      basic_simd<typename I::value_type, typename I::abi_type::underlying_abi>;
  using mask_type          = simd_index_type::mask_type;
  constexpr int nb_vectors = V::size() / simd_type::size();

  const std::array<simd_type, nb_vectors> vec_v(v);
  const std::array<simd_index_type, nb_vectors> vec_indices(indices);
  const std::array<mask_type, nb_vectors> vec_masks(mask);
  Impl::unroll<nb_vectors>([&]<int Idx>(std::integral_constant<int, Idx>) {
    unchecked_scatter_to(vec_v[Idx], out, vec_masks[Idx], vec_indices[Idx],
                         flag);
  });
}

template <Impl::SimdVecType V, Impl::Ranges::contiguous_range R,
          Impl::SimdIntegral I, typename... Flags>
  requires Impl::Ranges::sized_range<R> &&
           Impl::is_extended_abi_v<typename V::abi_type> &&
           std::is_same_v<typename V::abi_type, typename I::abi_type>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr void partial_scatter_to(
    const V& v, R&& out, const typename I::mask_type& mask, const I& indices,
    simd_flags<Flags...> flag = simd_flag_default) {
  unchecked_scatter_to(v, out, mask, indices, flag);
}

}  // namespace Experimental

}  // namespace Kokkos

#endif
