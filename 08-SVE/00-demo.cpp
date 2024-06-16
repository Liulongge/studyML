#include <arm_sve.h>  
#include <stdio.h>  
  
int main() {  
    // 假设我们有两个 16 个元素的整数向量  
    svint32_t vec1 = svdup_s32(1);  // 创建一个元素全为 1 的 SVE 向量  
    svint32_t vec2 = svdup_s32(2);  // 创建一个元素全为 2 的 SVE 向量  
  
    // 使用 SVE 指令进行向量加法  
    svint32_t sum = svadd_s32_z(svptrue_b32(), vec1, vec2);  
  
    // 打印结果向量的前几个元素（为了简化，这里只打印前 4 个元素）  
    for (int i = 0; i < 4; ++i) {  
        // printf("%d ", svext_s32(sum, i));  
    }  
    printf("\n");  
  
    return 0;  
}