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
  assert.match(md, /## TL;DR/)
  assert.match(md, /## Contribution Map/)
  assert.match(md, /self-attention/)
  assert.match(md, /Review Questions/)
})
