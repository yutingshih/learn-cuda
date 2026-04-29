#pragma once

inline auto div_ceil(int a, int b) -> int
{
    return 1 + (a - 1) / b;
}
