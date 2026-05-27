import assert from 'node:assert/strict'
import test from 'node:test'

import { briefSourceLabel, formatBriefMarkdown } from './studyBriefUtils.js'

const sampleBrief = {
  title: 'paper.pdf',
  source: 'not_generated',
  tldr: ['第一句', '第二句'],
  problem: '问题',
  motivation: '动机',
  contributions: [{ claim: '贡献', why_it_matters: '重要', evidence: { section_title: 'Method', quote: 'quote' } }],
  method_walkthrough: [{ title: 'Step 1', explanation: '解释' }],
  key_terms: [{ term: 'self-attention', explanation_zh: '自注意力', keep_english: true }],
  reading_focus: [{ section_title: 'Introduction', reason: '先看问题' }],
  review_questions: [{ question: '问题是什么？', level: 'basic', expected_answer_hint: '看引言' }],
  limitations: ['需要模型生成。'],
}

test('brief source label is explicit', () => {
  assert.equal(briefSourceLabel('not_generated'), '尚未生成')
  assert.equal(briefSourceLabel('llm'), '模型生成')
  assert.equal(briefSourceLabel('cached'), '已缓存')
})

test('formatBriefMarkdown exports structured content', () => {
  const md = formatBriefMarkdown(sampleBrief)

  assert.match(md, /# 论文理解简报/)
  assert.match(md, /## 核心速览/)
  assert.match(md, /## 核心贡献/)
  assert.match(md, /self-attention/)
  assert.match(md, /复习问题/)
})

test('formatBriefMarkdown uses lecture title for lecture briefs', () => {
  const md = formatBriefMarkdown({
    ...sampleBrief,
    brief_type: 'lecture',
    title: 'lecture.pdf',
    text_quality: 'good',
    ocr_needed: false,
  })

  assert.match(md, /# 课件学习简报: lecture\.pdf/)
  assert.match(md, /文本质量：良好/)
  assert.match(md, /## 本讲主题与学习目标/)
  assert.match(md, /## 知识点脉络/)
  assert.match(md, /## 重点 Slide/)
  assert.match(md, /## 自测问题/)
  assert.doesNotMatch(md, /## 核心贡献/)
})
