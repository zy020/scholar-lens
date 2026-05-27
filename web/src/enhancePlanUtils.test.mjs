import test from 'node:test'
import assert from 'node:assert/strict'
import { enhanceDecisionItems, enhancePlanSummary, enhanceReasonItems, formatPageLabel, ocrCapabilityLabel, visionPlanLabel } from './enhancePlanUtils.js'

test('enhancePlanSummary formats planned pages', () => {
  assert.equal(enhancePlanSummary({ status: 'planned', estimated_ocr_pages: 3 }), '建议增强 3 页')
})

test('enhancePlanSummary formats skipped state', () => {
  assert.equal(enhancePlanSummary({ status: 'skipped', estimated_ocr_pages: 0 }), '当前无需增强解析')
})

test('visionPlanLabel reflects availability and possible escalation', () => {
  assert.equal(visionPlanLabel({ vision_available: false }), 'Vision 未配置')
  assert.equal(
    visionPlanLabel({ vision_available: true, vision_enhancement_enabled: false, vision_possible: false }),
    'Vision 已配置，需在配置中启用增强',
  )
  assert.equal(
    visionPlanLabel({ vision_available: true, vision_enhancement_enabled: true, vision_possible: false }),
    'Vision 已配置，当前无需调用',
  )
  assert.equal(
    visionPlanLabel({ vision_available: true, vision_enhancement_enabled: true, vision_possible: true }),
    'Vision 可用于疑难页',
  )
})

test('enhanceReasonItems maps OCR reason codes', () => {
  assert.deepEqual(
    enhanceReasonItems({
      ocr_recommendation_reasons: {
        2: 'text_low_parser_visuals',
        4: 'document_image_based',
      },
    }),
    [
      { page: '2', pageLabel: '第 3 页', label: '文本少且含视觉元素' },
      { page: '4', pageLabel: '第 5 页', label: '疑似图片型文档' },
    ],
  )
})

test('formatPageLabel formats internal zero-based pages for display', () => {
  assert.equal(formatPageLabel(0), '第 1 页')
  assert.equal(formatPageLabel('bad'), '相关页面')
})

test('ocrCapabilityLabel formats gpu cpu and vision choices', () => {
  assert.equal(ocrCapabilityLabel({ ocr_installed: true, ocr_gpu_available: true }), 'GPU OCR 可用')
  assert.equal(
    ocrCapabilityLabel({
      ocr_installed: true,
      ocr_gpu_available: false,
      ocr_cpu_available: false,
      available_actions: ['vision'],
    }),
    'GPU OCR 不可用，可使用 Vision',
  )
  assert.equal(ocrCapabilityLabel({ ocr_installed: false, available_actions: ['vision'] }), 'RapidOCR 未安装，可使用 Vision')
})

test('enhanceDecisionItems formats page-level enhancement actions and reasons', () => {
  assert.deepEqual(
    enhanceDecisionItems({
      page_decisions: [
        { page: 0, action: 'use_original', reason: 'sparse_but_structured' },
        { page: 2, action: 'apply_ocr', reason: 'ocr_readable_gain' },
        { page: 3, action: 'apply_vision', reason: 'visual_semantics_need_vision' },
        { page: 4, action: 'apply_ocr_then_maybe_vision', reason: 'ocr_first_for_visual_text' },
      ],
    }),
    [
      { key: '0-use_original', pageLabel: '第 1 页', action: '无需增强', reason: '内容少但结构完整' },
      { key: '2-apply_ocr', pageLabel: '第 3 页', action: '建议 OCR', reason: 'OCR 可读文本明显提升' },
      { key: '3-apply_vision', pageLabel: '第 4 页', action: '建议 Vision', reason: '需要理解图表/公式等视觉语义' },
      { key: '4-apply_ocr_then_maybe_vision', pageLabel: '第 5 页', action: '先 OCR，必要时 Vision', reason: '优先提取图片文字' },
    ],
  )
})
