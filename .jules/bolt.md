## 2025-10-26 - [Vectorized Simulation Loop]
**Learning:** Pre-calculating daily flux arrays (P/ET) outside the main simulation loop significantly improved performance (~4x speedup). The previous implementation performed repeated pandas Series lookups and DOY conversions inside the loop, which was a bottleneck.
**Action:** When working with iterative time-step simulations in Python/Pandas, always vectorize input series preparation before entering the sequential loop.
