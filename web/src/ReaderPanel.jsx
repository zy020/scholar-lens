import { canInlinePreview, readerOpenLabel, readerPreviewUrl } from './readerPanelUtils'

export default function ReaderPanel({ doc }) {
  const fileUrl = doc.file_url || null

  if (!doc.name) {
    return <div className="reader-panel"><p className="empty">请先上传文档</p></div>
  }

  const inlinePreview = canInlinePreview(doc)
  const previewUrl = readerPreviewUrl(doc)

  return (
    <div className="reader-panel">
      <div className="reader-header">
        <span>{doc.name}</span>
        <div className="reader-actions">
          {fileUrl && (
            <a href={fileUrl} target="_blank" rel="noreferrer" className="pdf-link">{readerOpenLabel(doc)}</a>
          )}
        </div>
      </div>
      {fileUrl && inlinePreview && <iframe src={previewUrl} className="pdf-viewer" title="文档预览" />}
      {fileUrl && !inlinePreview && <div className="pdf-placeholder">当前文档不可内嵌预览，原文件请通过上方按钮打开。</div>}
      {!fileUrl && <div className="pdf-placeholder">原文件不可用</div>}
    </div>
  )
}
