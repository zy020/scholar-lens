import { useEffect, useState } from 'react'
import { getConfig, updateConfig } from './api'
import { mergeConfigResponse, saveStatusText } from './configPanelUtils'

function ModelSection({ label, prefix, showModel, settings, onChange }) {
  return (
    <fieldset className="config-card">
      <legend>{label}</legend>
      {showModel && (
        <div className="config-field">
          <label>模型</label>
          <input value={settings[`${prefix}_model`] || ''} onChange={onChange(`${prefix}_model`)} placeholder="模型名称" />
        </div>
      )}
      <label className="checkbox-label">
        <input type="checkbox" checked={settings[`${prefix}_use_separate`] || false} onChange={onChange(`${prefix}_use_separate`)} />
        {' '}使用单独配置
      </label>
      {settings[`${prefix}_use_separate`] && (
        <>
          <div className="config-field">
            <label>API Key</label>
            <input value={settings[`${prefix}_api_key`] || ''} onChange={onChange(`${prefix}_api_key`)} type="password" placeholder="密钥" />
          </div>
          <div className="config-field">
            <label>Base URL</label>
            <input value={settings[`${prefix}_base_url`] || ''} onChange={onChange(`${prefix}_base_url`)} placeholder="https://api.openai.com/v1" />
          </div>
        </>
      )}
    </fieldset>
  )
}

export default function ConfigPanel() {
  const [settings, setSettings] = useState({
    api_key: '', base_url: '',
    llm_model: '', llm_use_separate: false, llm_api_key: '', llm_base_url: '',
    embedding_model: '', embedding_use_separate: false, embedding_api_key: '', embedding_base_url: '',
    reranker_enabled: true, reranker_model: '', reranker_use_separate: false, reranker_api_key: '', reranker_base_url: '',
    vision_enabled: true, vision_model: '', vision_use_separate: false, vision_api_key: '', vision_base_url: '',
    auto_ocr_enabled: true, llm_quality_enabled: false, vision_enhancement_enabled: false,
    memory_llm_compression_enabled: false,
  })
  const [status, setStatus] = useState('')

  useEffect(() => {
    let ignore = false
    getConfig().then(data => {
      if (!ignore) setSettings(current => mergeConfigResponse(current, data))
    }).catch(err => {
      if (!ignore) setStatus(`加载配置失败: ${err.message}`)
    })
    return () => { ignore = true }
  }, [])

  const set = (field) => (e) => {
    const val = e.target.type === 'checkbox' ? e.target.checked : e.target.value
    setSettings(s => ({ ...s, [field]: val }))
  }

  const handleSave = async () => {
    setStatus('')
    try {
      const data = await updateConfig(settings)
      setSettings(current => mergeConfigResponse(current, data))
      setStatus(saveStatusText(data))
      window.dispatchEvent(new CustomEvent('scholarlens-config-saved', { detail: data }))
    } catch (err) {
      setStatus(`保存失败: ${err.message}`)
    }
  }

  return (
    <div className="config-panel">
      <h3>模型配置</h3>

      <fieldset className="config-card">
        <legend>全局凭据</legend>
        <div className="config-field">
          <label>API Key</label>
          <input value={settings.api_key || ''} onChange={set('api_key')} type="password" placeholder="密钥" />
        </div>
        <div className="config-field">
          <label>Base URL</label>
          <input value={settings.base_url || ''} onChange={set('base_url')} placeholder="https://api.openai.com/v1" />
        </div>
      </fieldset>

      <ModelSection label="LLM" prefix="llm" showModel settings={settings} onChange={set} />

      <ModelSection label="Embedding" prefix="embedding" showModel settings={settings} onChange={set} />

      <fieldset className="config-card">
        <legend>解析增强</legend>
        <label className="checkbox-label">
          <input type="checkbox" checked={settings.auto_ocr_enabled !== false} disabled readOnly />
          {' '}上传后自动使用 GPU OCR 处理推荐页
        </label>
        <label className="checkbox-label">
          <input type="checkbox" checked={settings.llm_quality_enabled || false} onChange={set('llm_quality_enabled')} />
          {' '}手动解析增强时启用 LLM 解析质量评估
        </label>
        <label className="checkbox-label">
          <input type="checkbox" checked={settings.vision_enhancement_enabled || false} onChange={set('vision_enhancement_enabled')} />
          {' '}手动解析增强时启用 Vision 处理疑难页面
        </label>
      </fieldset>

      <fieldset className="config-card">
        <legend>学习记忆</legend>
        <label className="checkbox-label">
          <input type="checkbox" checked={settings.memory_llm_compression_enabled || false} onChange={set('memory_llm_compression_enabled')} />
          {' '}使用 LLM 压缩连续学习记忆
        </label>
      </fieldset>

      <fieldset className="config-card">
        <legend>Reranker</legend>
        <label className="checkbox-label">
          <input type="checkbox" checked={settings.reranker_enabled || false} onChange={set('reranker_enabled')} />
          {' '}保存 Reranker 配置
        </label>
        {settings.reranker_enabled && (
          <>
            <div className="config-field">
              <label>模型</label>
              <input value={settings.reranker_model || ''} onChange={set('reranker_model')} placeholder="Qwen3-Reranker-0.6B" />
            </div>
            <label className="checkbox-label">
              <input type="checkbox" checked={settings.reranker_use_separate || false} onChange={set('reranker_use_separate')} />
              {' '}使用单独配置
            </label>
            {settings.reranker_use_separate && (
              <>
                <div className="config-field">
                  <label>API Key</label>
                  <input value={settings.reranker_api_key || ''} onChange={set('reranker_api_key')} type="password" placeholder="密钥" />
                </div>
                <div className="config-field">
                  <label>Base URL</label>
                  <input value={settings.reranker_base_url || ''} onChange={set('reranker_base_url')} placeholder="https://api.openai.com/v1" />
                </div>
              </>
            )}
          </>
        )}
      </fieldset>

      <fieldset className="config-card">
        <legend>Vision / OCR</legend>
        <label className="checkbox-label">
          <input type="checkbox" checked={settings.vision_enabled || false} onChange={set('vision_enabled')} />
          {' '}保存 Vision 配置
        </label>
        {settings.vision_enabled && (
          <>
            <div className="config-field">
              <label>模型</label>
              <input value={settings.vision_model || ''} onChange={set('vision_model')} placeholder="Qwen3.6-27B" />
            </div>
            <label className="checkbox-label">
              <input type="checkbox" checked={settings.vision_use_separate || false} onChange={set('vision_use_separate')} />
              {' '}使用单独配置
            </label>
            {settings.vision_use_separate && (
              <>
                <div className="config-field">
                  <label>API Key</label>
                  <input value={settings.vision_api_key || ''} onChange={set('vision_api_key')} type="password" placeholder="密钥" />
                </div>
                <div className="config-field">
                  <label>Base URL</label>
                  <input value={settings.vision_base_url || ''} onChange={set('vision_base_url')} placeholder="https://api.openai.com/v1" />
                </div>
              </>
            )}
          </>
        )}
      </fieldset>

      <button className="save-btn" onClick={handleSave} style={{ marginTop: 8 }}>保存</button>
      {status && <p className="empty" style={{ fontSize: '.75rem', marginTop: 4 }}>{status}</p>}
    </div>
  )
}
