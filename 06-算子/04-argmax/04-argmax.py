

# 参考:
# https://blog.csdn.net/qq_36098284/article/details/128653733?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171155671416800192283848%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171155671416800192283848&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-4-128653733-null-null.142^v100^pc_search_result_base7&utm_term=argmax&spm=1018.2226.3001.4187

# https://zhuanlan.zhihu.com/p/677737249


def argmax(lst):
    if not lst:  # 确保列表非空
        raise ValueError("Cannot find argmax in an empty list")

    max_value = lst[0]
    max_index = 0

    for index, value in enumerate(lst):
        if value > max_value:
            max_value = value
            max_index = index

    return max_index

# 示例
data = [4, 2, 9, 6, 11, 3]
max_index = argmax(data)
print("最大索引值:", max_index)


import torch
# 假设我们有一个张量
my_tensor = torch.tensor(data)

# 计算张量中每个元素的最大值索引
# 如果你想获取每行的最大值索引
max_index = torch.argmax(my_tensor, dim=-1)

# 输出每行最大值的索引
print("最大索引值:", int(max_index))