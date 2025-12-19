## 2025-12-19 - Pandas Copy Optimization
**Learning:** Frequent full dataframe copies (e.g., `df.copy()`) just for minor operations like column renaming can be a significant performance bottleneck in data-heavy applications.
**Action:** Use targeted column access with case-insensitive logic instead of copying the whole dataframe when possible.
