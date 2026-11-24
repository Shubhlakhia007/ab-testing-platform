# Data Quality Log

Audit log documenting all data cleaning and standardization operations applied to the dataset.

| Issue | Count | Action |
|:---|---:|:---|
| Exact duplicate rows | 0 | No exact duplicate rows found; no rows dropped |
| Missing values in group column 'test group' | 0 | No missing group values detected; no rows dropped |
| Missing values in conversion column 'converted' | 0 | No missing conversion values detected; no rows dropped |
| Whitespace in group column 'test group' | 1 | Stripped leading/trailing whitespace from 1 values |
| Conversion format standardization for 'converted' | 588,829 | Normalized boolean/text representations to binary integers (0 and 1) |
| Duplicate user IDs in 'user id' | 156,517 | Identified 156,517 duplicate user ID rows (retained for user-level aggregation if needed) |
