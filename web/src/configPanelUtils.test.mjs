import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { mergeConfigResponse, saveStatusText } from './configPanelUtils.js'

test('mergeConfigResponse fills editable model and endpoint fields', () => {
  const current = { api_key: '', llm_model: '', reranker_enabled: false }
  const merged = mergeConfigResponse(current, {
    llm_model: 'gpt-4o-mini',
    llm_base_url: 'https://llm.example/v1',
    embedding_model: 'text-embedding-3-small',
    embedding_base_url: 'https://emb.example/v1',
    reranker_available: true,
    reranker_active: false,
    reranker_model: 'rerank',
    reranker_base_url: 'https://rerank.example/v1',
    reranker_use_separate: true,
    vision_available: false,
    vision_use_separate: true,
    auto_ocr_enabled: true,
    llm_quality_enabled: true,
    vision_enhancement_enabled: false,
    memory_llm_compression_enabled: true,
  })

  assert.equal(merged.llm_model, 'gpt-4o-mini')
  assert.equal(merged.llm_base_url, 'https://llm.example/v1')
  assert.equal(merged.embedding_model, 'text-embedding-3-small')
  assert.equal(merged.reranker_enabled, false)
  assert.equal(merged.reranker_model, 'rerank')
  assert.equal(merged.reranker_use_separate, true)
  assert.equal(merged.vision_enabled, false)
  assert.equal(merged.vision_use_separate, true)
  assert.equal(merged.auto_ocr_enabled, true)
  assert.equal(merged.llm_quality_enabled, true)
  assert.equal(merged.vision_enhancement_enabled, false)
  assert.equal(merged.memory_llm_compression_enabled, true)
  assert.equal(merged.api_key, '')
})

test('saveStatusText distinguishes live updates from restart-required updates', () => {
  assert.equal(saveStatusText({ requires_restart: false }), '已保存，当前会话已生效')
  assert.equal(saveStatusText({ requires_restart: true }), '已保存，重启后完全生效')
})

test('ConfigPanel broadcasts config save without touching documents directly', () => {
  const source = readFileSync(new URL('./ConfigPanel.jsx', import.meta.url), 'utf8')
  assert.match(source, /scholarlens-config-saved/)
  assert.doesNotMatch(source, /enhanceVision|enhanceOcr|applyEnhancement/)
})
