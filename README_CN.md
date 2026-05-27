# ScholarLens 学术透镜

ScholarLens 是一个面向中文学生阅读英文论文和课件的本地优先学术阅读助手。它保留原文阅读界面，用可追溯 evidence 回答问题，支持翻译、学习简报，并为低质量解析结果提供 OCR 和可选 Vision 增强。

[English README](README.md)

## 功能亮点

- Evidence-first QA：回答必须基于检索片段，并使用 `[1]` 这类 evidence 编号引用。
- 论文与课件双入口：论文入口用于 PDF 论文；课件入口支持 PDF 和 PPTX。
- 双语学习支持：中文解释中保留关键 English terms、公式、模型名和数据集名。
- 解析质量诊断：通过启发式质量评分判断哪些页面需要 OCR 或 Vision 增强。
- OCR/Vision 增强：上传后默认使用 RapidOCR 处理推荐 OCR 的页面；Vision 可在 Config 面板中开启，作为更强的升级路径。
- 公式感知文本层：规范化公式和公式类文本，提升检索与解释效果。
- 语义化切片与混合检索：结合章节/页面结构、BM25/token overlap、可选向量检索、reranking、上下文扩展和 memory-aware retrieval hints。
- 学习工具：流式 Chat、章节翻译、Study Brief、evidence 卡片和学习记忆面板。

## 未来更新方向

- 优化 PPTX 理解能力，支持整页 slide 渲染后再进行 OCR/Vision，提升截图型课件、复杂图示和图文混排页面的解析质量。
- 增强表格和公式处理能力，在当前公式感知文本层之外，进一步提升表格数值问答和公式解释的可靠性。
- 将解析、OCR、Vision 和索引等耗时任务改为后台任务，并提供更清晰的进度反馈，改善大文件上传体验。
- 持续优化中文或中英混合问题检索英文论文的效果，提升跨语言检索质量。

## 技术架构

```text
scholar-lens/
├── scholar_lens/
│   ├── api/          FastAPI 应用、路由、请求/响应 schema、Chat 服务和文档分析接口
│   │   └── routes/   文档上传/解析/增强、配置、Chat、Study Brief、Memory API
│   ├── parsers/      PDF/PPTX 解析、质量评估、语义切片、OCR/Vision 增强和公式文本规范化
│   ├── rag/          本地文档存储、BM25/token overlap 检索、向量索引、reranker 和上下文扩展
│   ├── memory/       学习记忆、概念状态、当前位置、近期行为和检索提示
│   ├── agents/       文档理解、解释、辅导和校验相关的 LLM agent 封装
│   └── core/         配置、模型工厂、异常、熔断器和 token 统计等基础设施
├── web/
│   └── src/          React/Vite 前端：侧边栏、阅读器、Chat、翻译、Study Brief、Memory 和配置面板
├── scripts/          本地调试、真实文件 smoke、memory smoke 和评估辅助脚本
├── tests/            单元测试、集成测试和前端工具函数测试
└── data/             本地运行数据目录，保存上传文件、解析结果、chunks、索引、memory 和评估输出
```

## 快速启动

```bash
conda create -n scholar_lens python=3.11 -y
conda activate scholar_lens

pip install -e ".[rag,parsers,dev]"

cd web
npm install
cd ..

cp .env.example .env
```

可以在 `.env` 或网页 Config 面板中配置模型。需要 LLM 的功能在未配置模型时会显示不可用状态，不再生成低质量规则兜底内容。

启动后端：

```bash
python -m uvicorn scholar_lens.api.main:create_app --factory --reload
```

启动前端：

```bash
cd web
npm run dev
```

Vite 会输出本地前端地址，通常是：

```text
http://localhost:5173
```

## 推荐 Demo 流程

使用论文 PDF，或课件 PDF/PPTX。

1. 通过论文入口上传论文，或通过课件入口上传课件。
2. 等待文档状态变为 `ready`。
3. 质量诊断推荐的 OCR 页面会自动增强并应用；需要更强增强时，可在 Config 中开启 Vision 和 LLM 质量评估。
4. 在 Reader 中选择章节。
5. 在 Chat 中提问，例如：`这个 attention 公式里的 Q、K、V 分别表示什么？`
6. 查看中文流式回答，并展开 evidence 验证引用。
7. 使用 Translate 翻译选中章节或粘贴文本。
8. 使用 Study Brief 查看结构化学习摘要。
9. 打开 Memory 查看近期行为和概念状态；必要时清空会话或当前文档记忆。


## 评估说明

ScholarLens 使用一套 30 题高置信度的自建 QA 集进行本地评估，覆盖 5 份文档：3 篇公开论文（Transformer、BERT 和 GNN survey）以及 2 份课件文件（1 份 PDF、1 份 PPTX）。题型覆盖事实检索、方法理解、对比分析、公式解释、跨语言提问，以及课件结构或图表理解。

| 指标 | 结果 |
| --- | ---: |
| 生成成功率 | 100% |
| Judge 成功率 | 100% |
| 引用率 | 96.7% |
| 空上下文率 | 0.0% |
| 正确性 | 3.37 / 5 |
| 忠实性 | 4.10 / 5 |
| 证据质量 | 3.67 / 5 |
| 完整性 | 3.03 / 5 |
| Retrieval hit@5 | 16.7% |
| Context hit@5 | 16.7% |
| MRR@5 | 0.15 |

## 许可证

MIT
