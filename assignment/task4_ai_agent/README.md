# 任务四：AI Agent 辅助算子开发、验证与优化

## 任务目标

选择 Add 或 Softmax（也可自己设计 ReLU、LayerNorm、GELU、RMSNorm），让 AI Agent 实质性参与至少一个环节，并用可复现实验证据证明最终结果正确。

如何在模力方舟上配置AI编程助手,可见[开发工具配置](https://ai.gitee.com/docs/integrations/Development-Tools/Codex),
(不局限于Codex,其他也均可以),报名领取的算力卷也可用来购买沐曦专属的Token资源包,用于进行API调用

## 必须完成的四个阶段

1. **需求分析**：向 Agent 提供算子公式、shape/dtype、TileLang 版本，请它列出算法步骤、边界条件和风险。
2. **代码或测试生成**：让 Agent 生成 kernel 初稿或测试用例；
3. **验证与调试**：运行对应任务的 pytest 和 benchmark。如果失败日志发回 Agent，请它提出修复。
4. **性能优化**：让 Agent 基于实测数据提出至少两个优化假设（如 BLOCK 大小、fragment、归约布局、访存融合），逐一实测并记录是否有效。

## Prompt 要求

每轮 Prompt 必须包含：算子契约、目标硬件、当前代码/错误、验收标准。可直接从 `prompts.md` 模板开始。禁止只提交一句“帮我优化”，必须让 Agent 给出可验证的具体建议。

## 提交

最终prompts.md文件的截图展示

