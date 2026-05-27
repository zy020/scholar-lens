import assert from 'node:assert/strict'
import test from 'node:test'
import config from './vite.config.js'

test('vite config splits large vendor libraries into separate chunks', () => {
  const manualChunks = config.build?.rollupOptions?.output?.manualChunks

  assert.equal(typeof manualChunks, 'function')
  assert.equal(manualChunks('/project/node_modules/react/index.js'), 'react')
  assert.equal(manualChunks('/project/node_modules/react-dom/index.js'), 'react')
  assert.equal(manualChunks('/project/node_modules/katex/dist/katex.mjs'), 'katex')
  assert.equal(manualChunks('/project/node_modules/react-katex/dist/react-katex.js'), 'katex')
  assert.equal(manualChunks('/project/node_modules/lucide-react/dist/esm/icons.js'), 'icons')
  assert.equal(manualChunks('/project/node_modules/other-lib/index.js'), 'vendor')
  assert.equal(manualChunks('/project/src/App.jsx'), undefined)
})
