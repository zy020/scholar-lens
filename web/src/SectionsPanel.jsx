import { useState, useMemo } from 'react'

function numberSections(sections) {
  const counters = []
  return sections.map((section) => {
    const level = Math.max(1, Number(section.level || 1))
    counters[level - 1] = (counters[level - 1] || 0) + 1
    counters.length = level
    return {
      ...section,
      displayNumber: counters.slice(0, level).join('.'),
    }
  })
}

function buildTree(sections) {
  if (!sections || sections.length === 0) return []
  const nodes = sections.map((s, i) => ({
    ...s,
    index: i,
    isParent: false,
  }))
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      if (nodes[j].level <= nodes[i].level) break
      nodes[i].isParent = true
      break
    }
  }
  return nodes
}

export default function SectionsPanel({
  sections,
  activeSectionId,
  onSelectSection,
  maxInitialLevel = Infinity,
  compact = false,
}) {
  const numbered = useMemo(() => numberSections(sections), [sections])
  const tree = useMemo(() => buildTree(numbered), [numbered])

  const initialCollapsed = useMemo(() => {
    const next = {}
    for (const node of tree) {
      if (node.level >= maxInitialLevel && node.isParent) {
        next[node.index] = true
      }
    }
    return next
  }, [tree, maxInitialLevel])

  const [collapsed, setCollapsed] = useState(() => ({ ...initialCollapsed }))

  const toggleCollapse = (idx) => {
    setCollapsed(prev => ({ ...prev, [idx]: !prev[idx] }))
  }

  const visibleNodes = (() => {
    if (!tree.length) return []
    const result = []
    let hidingUntilLevel = Infinity
    for (let i = 0; i < tree.length; i++) {
      const node = tree[i]
      if (node.level <= hidingUntilLevel) hidingUntilLevel = Infinity
      const isVisible = hidingUntilLevel === Infinity
      result.push({ ...node, visible: isVisible })
      if (node.isParent && collapsed[node.index]) {
        hidingUntilLevel = Math.min(hidingUntilLevel, node.level)
      }
    }
    return result.filter(n => n.visible)
  })()

  if (sections.length === 0) {
    return <div className={`sections-panel ${compact ? 'compact' : ''}`}><p className="empty">本文档无可用章节</p></div>
  }

  return (
    <div className={`sections-panel ${compact ? 'compact' : ''}`}>
      <div className="section-list">
        {visibleNodes.map(s => (
          <button key={s.section_id}
                  className={`section-btn ${activeSectionId === s.section_id ? 'active' : ''}`}
                  onClick={() => onSelectSection(s)}
                  style={{ paddingLeft: 8 + s.level * 14 }}>
            {s.isParent && (
              <span className="section-toggle" onClick={(e) => { e.stopPropagation(); toggleCollapse(s.index) }}>
                {collapsed[s.index] ? '▸' : '▾'}
              </span>
            )}
            {!s.isParent && <span className="section-toggle-spacer" />}
            <span className="section-number">{s.displayNumber || `§${s.index + 1}`}</span>
            <span className="section-title-text">{s.title}</span>
            {s.page_start != null && <span className="section-page">第 {Number(s.page_start) + 1} 页</span>}
          </button>
        ))}
      </div>
    </div>
  )
}
