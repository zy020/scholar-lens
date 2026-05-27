import assert from 'node:assert/strict'
import test from 'node:test'

import { briefFocusLabel, briefSourceLabel, formatBriefMarkdown } from './studyBriefUtils.js'

const sampleBrief = {
  title: 'paper.pdf',
  source: 'not_generated',
  tldr: ['第一句', '第二句'],
  problem: '问题',
  motivation: '动机',
  contributions: [{ claim: '贡献', why_it_matters: '重要', evidence: { section_title: 'Method', quote: 'quote' } }],
  method_walkthrough: [{ title: 'Step 1', explanation: '解释' }],
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

  assert.match(md, /# 论文学习分析/)
  assert.match(md, /## 核心速览/)
  assert.match(md, /## 核心贡献/)
  assert.doesNotMatch(md, /关键术语/)
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

  assert.match(md, /# 课件学习分析: lecture\.pdf/)
  assert.match(md, /文本质量：良好/)
  assert.match(md, /## 本讲主题与学习目标/)
  assert.match(md, /## 知识点脉络/)
  assert.match(md, /## 重点 Slide/)
  assert.match(md, /## 自测问题/)
  assert.doesNotMatch(md, /## 核心贡献/)
})

test('briefFocusLabel shows slide page number for lecture focus items', () => {
  assert.equal(
    briefFocusLabel({ section_id: 'slide_3', section_title: 'Convolution', reason: '理解卷积' }, { brief_type: 'lecture' }),
    'Slide 4：Convolution',
  )
  assert.equal(
    briefFocusLabel({ section_title: 'Introduction', reason: '理解问题' }, { brief_type: 'paper' }),
    'Introduction',
  )
})
