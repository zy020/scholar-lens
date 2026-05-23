import assert from 'node:assert/strict'
import test from 'node:test'

import { briefSourceLabel, formatBriefMarkdown } from './studyBriefUtils.js'

const sampleBrief = {
  title: 'paper.pdf',
  source: 'fallback',
  tldr: ['第一句', '第二句'],
  problem: '问题',
  motivation: '动机',
  contributions: [{ claim: '贡献', why_it_matters: '重要', evidence: { section_title: 'Method', quote: 'quote' } }],
  method_walkthrough: [{ title: 'Step 1', explanation: '解释' }],
  key_terms: [{ term: 'self-attention', explanation_zh: '自注意力', keep_english: true }],
  reading_focus: [{ section_title: 'Introduction', reason: '先看问题' }],
  review_questions: [{ question: '问题是什么？', level: 'basic', expected_answer_hint: '看引言' }],
  limitations: ['fallback'],
}

test('brief source label is explicit', () => {
  assert.equal(briefSourceLabel('fallback'), 'Fallback Brief')
  assert.equal(briefSourceLabel('llm'), 'LLM Brief')
  assert.equal(briefSourceLabel('cached'), 'Cached Brief')
})

test('formatBriefMarkdown exports structured content', () => {
  const md = formatBriefMarkdown(sampleBrief)

  assert.match(md, /# Paper Understanding Brief/)
  assert.match(md, /## 核心速览/)
  assert.match(md, /## Contribution Map/)
  assert.match(md, /self-attention/)
  assert.match(md, /Review Questions/)
})

test('formatBriefMarkdown uses lecture title for lecture briefs', () => {
  const md = formatBriefMarkdown({
    ...sampleBrief,
    brief_type: 'lecture',
    title: 'lecture.pdf',
    text_quality: 'good',
    ocr_needed: false,
  })

  assert.match(md, /# Lecture Study Brief: lecture\.pdf/)
  assert.match(md, /Text Quality: good/)
})

test('formatBriefMarkdown exports low text warning clearly', () => {
  const md = formatBriefMarkdown({
    ...sampleBrief,
    brief_type: 'low_text',
    title: 'scanned.pdf',
    text_quality: 'image_based',
    ocr_needed: true,
    problem: '文本抽取不足，无法可靠识别本材料。',
    contributions: [],
    method_walkthrough: [],
    limitations: ['OCR 和 Vision 解析未在本阶段实现。'],
  })

  assert.match(md, /# Low Text Study Brief: scanned\.pdf/)
  assert.match(md, /OCR Needed: yes/)
  assert.match(md, /文本抽取不足/)
})
