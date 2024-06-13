import numpy as np
from scipy.stats import wilcoxon

# 示例数据：每个查询的倒数排名（RR）
query_rrs = np.array([1/1, 1/2, 1/3, 1/4, 1/5])  # 替换为实际数据

# 计算MRR
mrr_observed = np.mean(query_rrs)

# 基准MRR值（例如随机检索结果的期望MRR）
mrr_baseline = 0.2  # 替换为实际基准值 0.6824

# 计算每个查询的RR与基准MRR的差异
rr_diffs = query_rrs - mrr_baseline

# 进行Wilcoxon符号秩检验
statistic, p_value = wilcoxon(rr_diffs)

print(f"MRR (Observed): {mrr_observed}")
print(f"p-value: {p_value}")

# 判断结果是否显著
alpha = 0.05
if p_value < alpha:
    print("结果在统计上显著 (拒绝零假设)")
else:
    print("结果在统计上不显著 (不能拒绝零假设)")
