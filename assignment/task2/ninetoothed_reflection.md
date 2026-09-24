# NineToothed Vector Add 实验总结

NineToothed 将算子的分块组织和分块内计算拆分为 arrangement 与 application 两部分。arrangement 使用 BLOCK_SIZE 对输入和输出进行切块，application 只需要描述每个分块内的逐元素加法。对于不能整除 BLOCK_SIZE 的向量长度，NineToothed 能够自动处理尾块，因此测试规模 98432 也能正确运行。开启自动调优后部分规模略有提升，最大规模耗时从 0.7820 ms 降至 0.7676 ms，但本次实验中整体仍略慢于 PyTorch。
