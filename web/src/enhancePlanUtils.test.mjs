import test from 'node:test'
import assert from 'node:assert/strict'
import { shouldShowEnhancePlan } from './enhancePlanUtils.js'

test('shouldShowEnhancePlan hides plans without actionable enhancement', () => {
  assert.equal(shouldShowEnhancePlan(null), false)
  assert.equal(shouldShowEnhancePlan({ status: 'skipped', page_decisions: [] }), false)
  assert.equal(shouldShowEnhancePlan({ status: 'planned', ocr_recommendation_reasons: { 0: 'parser_recommends_ocr' } }), false)
  assert.equal(shouldShowEnhancePlan({ status: 'planned', page_decisions: [{ page: 0, action: 'use_original' }] }), false)
  assert.equal(shouldShowEnhancePlan({ status: 'planned', estimated_ocr_pages: 1 }), true)
  assert.equal(shouldShowEnhancePlan({ status: 'planned', recommended_ocr_pages: [0] }), true)
  assert.equal(shouldShowEnhancePlan({ status: 'planned', page_decisions: [{ page: 1, action: 'apply_vision' }] }), true)
  assert.equal(shouldShowEnhancePlan({ status: 'planned', vision_possible: true }), true)
})
