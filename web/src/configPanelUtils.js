export function mergeConfigResponse(current, data) {
  return {
    ...current,
    llm_model: data.llm_model && data.llm_model !== 'not configured' ? data.llm_model : '',
    llm_base_url: data.llm_base_url || current.llm_base_url || '',
    embedding_model: data.embedding_model && data.embedding_model !== 'not configured' ? data.embedding_model : '',
    embedding_base_url: data.embedding_base_url || current.embedding_base_url || '',
    reranker_enabled: Object.hasOwn(data, 'reranker_active') ? Boolean(data.reranker_active) : Boolean(data.reranker_available),
    reranker_model: data.reranker_model || '',
    reranker_base_url: data.reranker_base_url || current.reranker_base_url || '',
    reranker_use_separate: Boolean(data.reranker_use_separate),
    vision_enabled: Boolean(data.vision_available),
    vision_model: data.vision_model || '',
    vision_base_url: data.vision_base_url || current.vision_base_url || '',
    vision_use_separate: Boolean(data.vision_use_separate),
    auto_ocr_enabled: Object.hasOwn(data, 'auto_ocr_enabled') ? Boolean(data.auto_ocr_enabled) : true,
    llm_quality_enabled: Boolean(data.llm_quality_enabled),
    vision_enhancement_enabled: Boolean(data.vision_enhancement_enabled),
    memory_llm_compression_enabled: Boolean(data.memory_llm_compression_enabled),
  }
}

export function saveStatusText(data) {
  return data?.requires_restart ? '已保存，重启后完全生效' : '已保存，当前会话已生效'
}
