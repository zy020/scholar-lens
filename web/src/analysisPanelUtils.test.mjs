import test from 'node:test'
import assert from 'node:assert/strict'
import {
  analysisStatusLabel,
  formatSectionSummaryLabel,
  parseQualityPageItems,
  parseQualityStatusLabel,
  sectionSummaryItems,
  termPreview,
} from './analysisPanelUtils.js'

test('analysisStatusLabel formats missing parser unavailable and llm states', () => {
  assert.equal(analysisStatusLabel({ status: 'missing' }), '未生成')
  assert.equal(analysisStatusLabel({ status: 'available', source: 'parse_quality' }), '解析质量')
  assert.equal(analysisStatusLabel({ status: 'available', source: 'parser' }), '解析结构')
  assert.equal(analysisStatusLabel({ status: 'available', source: 'unavailable' }), '需要配置模型')
  assert.equal(analysisStatusLabel({ status: 'available', source: 'llm' }), 'LLM 增强')
})

test('parse quality helpers format parser diagnostics', () => {
  assert.equal(parseQualityStatusLabel('enhanced_completed'), '已完成解析增强')
  assert.equal(parseQualityStatusLabel('needs_enhancement'), '需要增强解析')
  assert.deepEqual(
    parseQualityPageItems({
      parse_quality_pages: [
        { page: 2, page_label: '第 3 页', quality: 'weak', recommended_action: 'ocr', overall_score: 0.34 },
      ],
    }),
    [{ key: '2-ocr', pageLabel: '第 3 页', quality: 'weak', action: '低文本页', score: '0.34', preview: '' }],
  )
  assert.equal(parseQualityPageItems({ parse_quality_pages: [{ page: 17, recommended_action: 'ocr', overall_score: 0 }] })[0].score, '')
})

test('termPreview formats bilingual terms', () => {
  assert.deepEqual(
    termPreview([{ english: 'RAG', chinese: '检索增强生成' }, { english: 'LLM', chinese: '' }]),
    ['RAG（检索增强生成）', 'LLM'],
  )
})

test('sectionSummaryItems converts summary maps into display rows', () => {
  assert.deepEqual(
    sectionSummaryItems({ intro: 'Intro summary', method: 'Method summary' }, 1),
    [{ sectionId: 'intro', label: 'intro', summary: 'Intro summary' }],
  )
})

test('formatSectionSummaryLabel hides generated ids', () => {
  assert.equal(formatSectionSummaryLabel('slide_2'), 'Slide 3')
  assert.equal(formatSectionSummaryLabel('01HZX9J2S2Q8V1A9B3C4D5E6F7', 3), '章节 4')
})
