export default function ReaderPanel({ doc, activeSectionId, sections }) {
  const pdfUrl = doc.file_url || null

  if (!doc.name) {
    return <div className="reader-panel"><p className="empty">请先上传文档</p></div>
  }

  const currentSection = sections?.find(s => s.section_id === activeSectionId)
  const requestedPage = currentSection?.page_start ?? null
  const pdfSrc = pdfUrl && requestedPage ? `${pdfUrl}#page=${requestedPage}` : pdfUrl

  return (
    <div className="reader-panel">
      <div className="reader-header">
        <span>{doc.name}</span>
        <div className="reader-actions">
          {pdfUrl && (
            <a href={pdfSrc || pdfUrl} target="_blank" rel="noreferrer" className="pdf-link">新窗口打开 PDF</a>
          )}
        </div>
      </div>
      {pdfSrc && <iframe src={pdfSrc} className="pdf-viewer" title="PDF Viewer" />}
      {!pdfSrc && <div className="pdf-placeholder">PDF 不可用</div>}
    </div>
  )
}
