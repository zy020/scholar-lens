import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'

const source = readFileSync(new URL('./NotesPanel.jsx', import.meta.url), 'utf8')

test('NotesPanel loads study brief from an effect instead of render state', () => {
  assert.match(source, /import\s+\{[^}]*useCallback[^}]*useEffect[^}]*useState[^}]*\}\s+from\s+['"]react['"]/)
  assert.match(source, /useEffect\s*\(/)
  assert.doesNotMatch(source, /if\s*\(\s*!loaded\s*&&\s*docId\s*\)/)
  assert.doesNotMatch(source, /setLoaded\s*\(/)
})

test('NotesPanel avoids rendering empty optional brief sections', () => {
  assert.match(source, /brief\.method_walkthrough\?\.length > 0/)
  assert.match(source, /brief\.reading_focus\?\.length > 0/)
  assert.doesNotMatch(source, /brief\.key_terms/)
})

test('NotesPanel loads cached study brief on initial document load without force generation', () => {
  assert.match(source, /getStudyBrief\(docId,\s*false\)/)
  assert.doesNotMatch(source, /loadBrief\(true\)[\s\S]*useEffect/)
  assert.match(source, /生成文档学习分析/)
})

test('NotesPanel labels parse quality and keeps generated learning analysis separate', () => {
  assert.match(source, /解析质量：/)
  assert.match(source, /文档学习分析/)
  assert.match(source, /解析增强/)
  assert.match(source, /evaluateParseQuality\(docId,\s*true\)/)
  assert.match(source, /enhanceOcr\(docId,\s*'auto'\)/)
  assert.match(source, /enhanceVision\(docId\)/)
  assert.doesNotMatch(source, /文本：/)
  assert.doesNotMatch(source, /生成文档理解分析/)
})

test('NotesPanel defaults to parse quality analysis instead of mixed document understanding', () => {
  assert.match(source, /文档解析质量分析/)
  assert.doesNotMatch(source, />文档分析<\/h4>/)
  assert.doesNotMatch(source, /难度:/)
  assert.doesNotMatch(source, /预计阅读/)
  assert.doesNotMatch(source, /analysisTerms/)
  assert.doesNotMatch(source, /analysisSummaries/)
  assert.doesNotMatch(source, /mermaid_map/)
})

test('NotesPanel refreshes current enhance plan after config save without running enhancement', () => {
  assert.match(source, /scholarlens-config-saved/)
  assert.match(source, /refreshEnhancePlan\(\)/)
  const listenerBlock = source.match(/window\.addEventListener\('scholarlens-config-saved'[\s\S]*?window\.removeEventListener\('scholarlens-config-saved'[\s\S]*?\)/)?.[0] || ''
  assert.doesNotMatch(listenerBlock, /enhanceVision|enhanceOcr|applyEnhancement/)
})

test('NotesPanel combines Vision enhancement and apply into one user action', () => {
  assert.match(source, /hasUsableEnhancementText\(vision,\s*'vision_quality'\)/)
  assert.match(source, /const applied = await applyEnhancement\(docId\)/)
  assert.doesNotMatch(source, />应用增强结果<\//)
  assert.doesNotMatch(source, /执行并应用 Vision/)
})

test('NotesPanel shows enhancement completion without exposing internal enhancement plan details', () => {
  assert.match(source, /enhancementCompleted/)
  assert.match(source, /解析增强已完成/)
  assert.doesNotMatch(source, /增强解析计划/)
  assert.doesNotMatch(source, /enhance-plan-row/)
  assert.doesNotMatch(source, />执行 OCR<\//)
  assert.doesNotMatch(source, /enhanceReasonItems|enhanceDecisionItems|enhancePlanSummary/)
  assert.match(source, /!enhancementCompleted && !analysisEnhancedCompleted && parseQualityPages\.length > 0/)
})
