import { useState, useRef, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { explainText, getSectionText } from './api'
import { buildCoursewareTranslateHint, buildSectionCacheKey, buildSectionTranslationPrompt, isCoursewareDocument } from './translationUtils'
import SectionsPanel from './SectionsPanel'
import 'katex/dist/katex.min.css'

const EXPLAIN_TIMEOUT_MS = 120000

function MarkdownBlock({ text }) {
  if (!text) return null
  return (
    <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
      {text}
    </ReactMarkdown>
  )
}

export default function TranslatePanel({ doc, sections, activeSectionId, onSelectSection }) {
  const [translation, setTranslation] = useState('')
  const [translateInput, setTranslateInput] = useState('')
  const [translating, setTranslating] = useState(false)
  const [showSections, setShowSections] = useState(true)
  const cacheRef = useRef({})

  const currentSection = sections.find(s => s.section_id === activeSectionId)
  const courseware = isCoursewareDocument(doc)

  const doExplain = useCallback(async (text, mode, explicitCacheKey = '', explicitSectionId = activeSectionId) => {
    const sectionId = explicitSectionId || ''
    const cacheKey = explicitCacheKey || `${doc.doc_id || ''}|${sectionId}|${text.slice(0, 80)}|${text.length}|${mode}`
    if (cacheRef.current[cacheKey]) {
      setTranslation(cacheRef.current[cacheKey])
      return
    }
    setTranslating(true)
    setTranslation('')
    const timer = setTimeout(() => {
      setTranslating(false)
      setTranslation('请求超时，请重试')
    }, EXPLAIN_TIMEOUT_MS)
    try {
      const data = await explainText({
        message: text,
        doc_id: doc.doc_id || '',
        section_id: sectionId,
        mode,
      })
      clearTimeout(timer)
      const result = data.content || (typeof data === 'string' ? data : JSON.stringify(data))
      setTranslation(result)
      cacheRef.current[cacheKey] = result
    } catch (err) {
      clearTimeout(timer)
      setTranslation(`翻译失败: ${err.message}`)
    }
    setTranslating(false)
  }, [activeSectionId, doc.doc_id])

  const translateText = useCallback((text) => doExplain(text, 'translate'), [doExplain])

  const translateSection = useCallback(async () => {
    if (!currentSection) return
    setTranslating(true)
    setTranslation('')
    try {
      const data = await getSectionText(doc.doc_id, currentSection.section_id)
      const sectionText = data.text || currentSection.gist || ''
      if (!sectionText.trim()) {
        setTranslation('该章节无可翻译的文本内容')
        return
      }
      const cacheKey = buildSectionCacheKey(doc.doc_id, currentSection.section_id, 'translate-section', sectionText)
      const prompt = buildSectionTranslationPrompt(currentSection, sectionText)
      await doExplain(prompt, 'translate', cacheKey, currentSection.section_id)
    } catch (err) {
      setTranslation(`翻译失败: ${err.message}`)
    } finally {
      setTranslating(false)
    }
  }, [currentSection, doExplain, doc.doc_id])

  const translatePastedText = useCallback(() => {
    const text = translateInput.trim()
    if (!text) return
    translateText([
      'Translate the following selected academic text into clear Chinese.',
      'Preserve model names, dataset names, acronyms, formulas, and domain-specific terms in English when needed.',
      'Only output the translation. Do not append related terms or glossary.',
      '',
      text,
    ].join('\n'))
  }, [translateInput, translateText])

  if (!doc.name) {
    return <div className="translate-panel"><p className="empty">请先上传文档</p></div>
  }

  return (
    <div className="translate-panel">
      <h3>翻译</h3>
      {!courseware && (
        <div className="translate-sections">
          <button className="translate-sections-toggle" onClick={() => setShowSections(v => !v)}>
            {showSections ? '隐藏章节' : '显示章节'}
          </button>
          {showSections && (
            <SectionsPanel
              key={sections.length}
              sections={sections}
              activeSectionId={activeSectionId}
              onSelectSection={onSelectSection || (() => {})}
              maxInitialLevel={1}
              compact
              doc={doc}
            />
          )}
        </div>
      )}
      {courseware && (
        <p className="brief-muted">{buildCoursewareTranslateHint()}</p>
      )}
      {!courseware && currentSection && (
        <div className="translate-actions">
          <button onClick={translateSection} disabled={translating}>
            翻译当前章节
          </button>
        </div>
      )}
      <textarea value={translateInput} onChange={e => setTranslateInput(e.target.value)}
                rows={5} placeholder="或在此粘贴文本翻译..." />
      <button onClick={translatePastedText} disabled={translating || !translateInput.trim()}>翻译选中文本</button>
      {translating && <p className="empty translate-status">翻译中...</p>}
      {translation && (
        <div className="translation-result">
          <MarkdownBlock text={translation} />
        </div>
      )}
    </div>
  )
}
