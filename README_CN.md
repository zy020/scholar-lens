# ScholarLens 学术透镜

ScholarLens 是一个面向中文学生阅读英文论文和课件的本地优先学术阅读助手。它保留原文阅读界面，基于检索到的文档上下文回答问题，支持翻译、文档分析，并为低质量解析结果提供 OCR 和可选 Vision 增强。

[English README](README.md)

## 功能亮点

- 上下文约束问答：回答基于检索到的文档片段，同时允许在明确区分的情况下补充教学背景。
- 显式文档类型选择：单个 PDF 上传入口，上传前由用户选择论文或课件。
- 双语学习支持：中文解释中保留关键 English terms、公式、模型名和数据集名。
- 解析质量诊断：通过启发式质量评分判断哪些页面需要 OCR 或 Vision 增强。
- 解析质量增强：GPU RapidOCR 可用时，上传后自动处理推荐 OCR 的页面；前端可在配置面板启用 LLM 质量评估和 Vision，并通过“解析增强”手动处理疑难页面。
- 公式感知文本层：对公式和公式类文本进行规范化与标记，提升检索与解释效果。
- 语义化切片与混合检索：结合章节/页面结构，默认使用关键词/BM25检索，并支持可选向量检索、可选模型 reranking、上下文扩展和 memory-aware retrieval hints。
- 学习工具：流式 Chat、论文章节翻译、课件选中文本翻译、文档学习分析，以及作为后端个性化组件运行的学习记忆。

## 未来更新方向

- 优化 PDF 课件解析能力，提升截图型课件、复杂图示、表格、公式和图文混排页面的解析质量。
- 将解析、OCR、Vision 和索引等耗时任务改为后台任务，并提供更清晰的进度反馈，改善大文件上传体验。
- 持续优化中文或中英混合问题检索英文论文的效果，提升跨语言检索质量。
- 进一步增强公式、图表和表格理解能力；当前公式主要是文本层规范化，复杂视觉语义仍依赖 Vision。
- 增强模型调用稳定性、重试和任务恢复能力，降低外部 API 波动对长流程的影响。

## 支持的文档

- 论文 PDF。
- 课件 PDF。
- 不直接支持 PPT、PPTX、Word 或图片文件。请先将课件或文档导出为 PDF 后上传。

论文和课件使用同一个上传控件，但上传前需要由用户显式选择文档类型。ScholarLens 会据此采用不同的解析、切片、检索、翻译和分析策略。

## 技术架构

```text
scholar-lens/
├── scholar_lens/
│   ├── api/          FastAPI 应用、路由、请求/响应 schema、Chat 服务和文档分析接口
│   │   └── routes/   文档上传/解析/增强、配置、Chat、文档分析、Memory API
│   ├── parsers/      PDF 解析、质量评估、语义切片、OCR/Vision 增强和公式文本规范化
│   ├── rag/          本地文档存储、BM25/token overlap 检索、向量索引、reranker 和上下文扩展
│   ├── memory/       学习记忆、概念状态、当前位置、近期行为和检索提示
│   ├── agents/       文档理解、解释、辅导和校验相关的 LLM agent 封装
│   └── core/         配置、模型工厂、异常、熔断器和 token 统计等基础设施
├── web/
│   └── src/          React/Vite 前端：侧边栏、阅读器、Chat、翻译、文档分析和配置面板
├── tests/            单元测试、集成测试和前端工具函数测试
└── data/             本地运行数据目录，保存上传文件、解析结果、chunks、索引、memory 和缓存
```

## 环境要求

- Python 3.11。
- Node.js 和 npm，用于 React/Vite 前端。
- Chat、翻译、文档学习分析和可选 LLM 解析质量评估需要配置 LLM。
- 推荐配置 embedding 模型以启用向量检索；未配置 embedding 时，系统仍可使用关键词/BM25检索兜底。
- GPU OCR 需要 RapidOCR 与 ONNX Runtime CUDA 支持。CUDA OCR 不可用时，OCR 会暂停；手动解析增强仍可在配置模型后使用 LLM 质量评估和 Vision 增强。

## 模型配置

模型配置可以写入 `.env`，也可以在网页 Config 面板中更新。默认情况下，LLM、embedding、reranker 和 Vision 会继承共享的 `API_KEY` 和 `BASE_URL`，也可以为单个模型配置独立的 key 或 base URL。

常用配置包括：

- `LLM__MODEL`：用于 Chat、翻译、文档学习分析和可选 LLM 质量评估。
- `EMBEDDING__MODEL`：用于向量索引和检索。
- `RERANKER__MODEL`：用于可选的模型 reranking。
- `VISION__MODEL`：用于可选的 Vision 解析增强。

## 数据与隐私

ScholarLens 默认将上传文件、解析结果、chunks、索引、缓存分析和 memory 数据保存在本地 `data/` 目录。启用 LLM、embedding、reranker 或 Vision 功能时，相关文本、问题、检索上下文或选中的页面图片可能会发送到你配置的模型服务商。

## 当前限制

- 不支持直接上传 PPT/PPTX；请先导出为 PDF。
- 截图型课件 PDF、复杂图示、复杂表格和公式密集页面，可能需要 Vision 增强才能获得更好的理解效果。
- 公式理解目前主要依赖文本层规范化、LaTeX/符号保留和检索增强，并不是完整的公式 OCR 或自动推导系统。
- 图表和表格解析采用轻量结构识别与文本增强；复杂视觉关系仍需要 Vision 模型辅助。
- 大型 PDF 的解析、增强和索引可能耗时较长，因为耗时任务目前仍在请求流程中执行。
- 当前更适合作为本地单用户学习工具，尚未提供多用户权限隔离、任务队列和云端部署治理。
- 本地评估集包含私有课件样例，因此不随项目公开。

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
http://localhost:3000
```

## 评估说明

ScholarLens 使用一套 30 题高置信度的自建 QA 集进行 RAG 评估，覆盖 5 份文档：3 篇公开论文（Transformer、BERT 和 GNN survey）以及 2 份课件 PDF。题型覆盖事实检索、方法理解、对比分析、公式解释、跨语言提问，以及课件结构或图表理解。该评估集包含私有课件，因此不随项目公开。

| 指标 | 结果 |
| --- | ---: |
| 生成成功率 | 100% |
| Judge 成功率 | 100% |
| 空上下文率 | 0.0% |
| 正确性 | 3.37 / 5 |
| 忠实性 | 4.10 / 5 |
| 完整性 | 3.03 / 5 |

## 许可证

MIT
