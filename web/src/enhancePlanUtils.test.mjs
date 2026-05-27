import test from 'node:test'
import assert from 'node:assert/strict'
import { enhancePlanSummary, enhanceReasonItems, formatPageLabel, ocrCapabilityLabel, visionPlanLabel } from './enhancePlanUtils.js'

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
      ocr_cpu_available: true,
      available_actions: ['cpu_ocr', 'vision'],
    }),
    'GPU OCR 不可用，可选择 CPU OCR 或 Vision',
  )
  assert.equal(ocrCapabilityLabel({ ocr_installed: false, available_actions: ['vision'] }), 'RapidOCR 未安装，可使用 Vision')
})
