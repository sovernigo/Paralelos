#pragma once

namespace nbody {

#ifdef NBODY_USE_FLOAT
using Real = float;
inline constexpr bool kUsingFloat = true;
#else
using Real = double;
inline constexpr bool kUsingFloat = false;
#endif

}  // namespace nbody