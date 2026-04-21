/*
 * web/viz/umap_browser.js — interactive UMAP embedding browser.
 *
 * Exposes a single global entry point, `window.renderUMAP(points, container)`,
 * expected to be called by the bootstrap script in umap_browser.html once the
 * dataset has been fetched. The dataset schema matches the output of
 * `scripts/viz/project_embeddings_umap.py`:
 *
 *   [
 *     { bibcode, x, y, community_id, resolution },
 *     ...
 *   ]
 *
 * Style notes: 2-space indent, single quotes, semicolons at end of statements,
 * vanilla ES2020, no module system. Relies on the `deck` global loaded from
 * https://unpkg.com/deck.gl@9/dist.min.js.
 */
;(function () {
  'use strict'

  // 20-color categorical palette — d3.schemeCategory10 + d3.schemeTableau10
  // flattened. Indexed by community_id % PALETTE.length. Stored as [r,g,b]
  // triples because deck.gl's ScatterplotLayer expects numeric color arrays.
  const PALETTE = [
    [31, 119, 180],
    [255, 127, 14],
    [44, 160, 44],
    [214, 39, 40],
    [148, 103, 189],
    [140, 86, 75],
    [227, 119, 194],
    [127, 127, 127],
    [188, 189, 34],
    [23, 190, 207],
    [78, 121, 167],
    [242, 142, 43],
    [225, 87, 89],
    [118, 183, 178],
    [89, 161, 79],
    [237, 201, 72],
    [176, 122, 161],
    [255, 157, 167],
    [156, 117, 95],
    [186, 176, 172],
  ]
  const FALLBACK_COLOR = [160, 160, 160]

  function _colorForCommunity(cid) {
    if (cid == null || Number.isNaN(Number(cid))) {
      return FALLBACK_COLOR
    }
    const idx = ((Number(cid) % PALETTE.length) + PALETTE.length) % PALETTE.length
    return PALETTE[idx]
  }

  function _resolveContainer(container) {
    if (typeof container === 'string') {
      return document.querySelector(container)
    }
    return container
  }

  function _mountError(container, message) {
    const div = document.createElement('div')
    div.className = 'umap-error'
    div.style.padding = '12px'
    div.style.color = '#b00020'
    div.textContent = message
    container.appendChild(div)
  }

  function _computeBounds(points) {
    let minX = Infinity
    let maxX = -Infinity
    let minY = Infinity
    let maxY = -Infinity
    for (let i = 0; i < points.length; i += 1) {
      const p = points[i]
      const x = Number(p.x)
      const y = Number(p.y)
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y
    }
    if (minX === Infinity) {
      return { minX: -1, maxX: 1, minY: -1, maxY: 1 }
    }
    return { minX: minX, maxX: maxX, minY: minY, maxY: maxY }
  }

  function _initialViewState(bounds, width, height) {
    const cx = (bounds.minX + bounds.maxX) / 2
    const cy = (bounds.minY + bounds.maxY) / 2
    const spanX = Math.max(1e-6, bounds.maxX - bounds.minX)
    const spanY = Math.max(1e-6, bounds.maxY - bounds.minY)
    // Zoom so the longer axis fills ~80% of the canvas.
    // With OrthographicView, zoom=Z means 1 data-unit == 2^Z pixels.
    const pxPerUnit = Math.min(width / spanX, height / spanY) * 0.8
    const zoom = Math.log2(Math.max(1, pxPerUnit))
    return { target: [cx, cy, 0], zoom: zoom }
  }

  // Per-container cache of title lookups so we don't hammer /viz/api/paper/
  // while the user idles their cursor on a single dot.
  function _makeTitleCache() {
    const cache = new Map()
    return function lookupTitle(bibcode, onResolved) {
      if (cache.has(bibcode)) {
        onResolved(cache.get(bibcode))
        return
      }
      cache.set(bibcode, null) // mark in-flight
      fetch('/viz/api/paper/' + encodeURIComponent(bibcode), { cache: 'no-store' })
        .then(function (resp) {
          if (!resp.ok) return null
          return resp.json()
        })
        .then(function (body) {
          const title = body && body.title ? String(body.title) : ''
          cache.set(bibcode, title)
          onResolved(title)
        })
        .catch(function () {
          cache.delete(bibcode)
          onResolved('')
        })
    }
  }

  function _showTooltip(tooltipEl, x, y, text) {
    if (!tooltipEl) return
    tooltipEl.style.left = x + 8 + 'px'
    tooltipEl.style.top = y + 8 + 'px'
    tooltipEl.textContent = text
    tooltipEl.style.display = 'block'
  }

  function _hideTooltip(tooltipEl) {
    if (!tooltipEl) return
    tooltipEl.style.display = 'none'
  }

  function _updatePanel(bibcode, title, communityId) {
    const panel = document.getElementById('umap-panel')
    if (!panel) return
    const safeTitle = title || '(title unavailable)'
    const cidText = communityId != null ? String(communityId) : '—'
    panel.innerHTML =
      '<h2>Selected paper</h2>' +
      '<div><strong>bibcode:</strong> <code>' +
      String(bibcode).replace(/[<>&]/g, '') +
      '</code></div>' +
      '<div><strong>community:</strong> ' +
      cidText +
      '</div>' +
      '<div style="margin-top:6px">' +
      String(safeTitle).replace(/[<>]/g, '') +
      '</div>'
  }

  function _populateLegend(communityCounts, activeCommunityRef, onPick) {
    const grid = document.getElementById('umap-legend-grid')
    if (!grid) return
    grid.innerHTML = ''
    const ids = Array.from(communityCounts.keys()).sort(function (a, b) {
      return Number(a) - Number(b)
    })
    ids.forEach(function (cid) {
      const rgb = _colorForCommunity(cid)
      const chip =
        '<span class="chip" style="background:rgb(' +
        rgb[0] +
        ',' +
        rgb[1] +
        ',' +
        rgb[2] +
        ')"></span>'
      const count = communityCounts.get(cid)
      const label =
        'c' +
        cid +
        ' <span style="color:#888">(' +
        count.toLocaleString() +
        ')</span>'
      const el = document.createElement('div')
      el.className = 'legend-swatch'
      el.dataset.cid = String(cid)
      el.innerHTML = chip + '<span>' + label + '</span>'
      el.addEventListener('click', function () {
        const newPick = activeCommunityRef.value === cid ? null : cid
        activeCommunityRef.value = newPick
        grid.querySelectorAll('.legend-swatch').forEach(function (sw) {
          sw.classList.toggle('active', newPick != null && Number(sw.dataset.cid) === newPick)
        })
        onPick(newPick)
      })
      grid.appendChild(el)
    })
  }

  function _updateStats(n, activeCid, communityCounts) {
    const el = document.getElementById('umap-stats')
    if (!el) return
    if (activeCid == null) {
      el.textContent = n.toLocaleString() + ' papers · 20 communities'
    } else {
      const shown = communityCounts.get(activeCid) || 0
      el.textContent =
        shown.toLocaleString() +
        ' / ' +
        n.toLocaleString() +
        ' papers · community ' +
        activeCid
    }
  }

  function renderUMAP(points, container) {
    const node = _resolveContainer(container)
    if (!node) {
      throw new Error('renderUMAP: container not found')
    }
    if (!Array.isArray(points)) {
      _mountError(node, 'renderUMAP: expected an array of {bibcode,x,y,community_id}')
      return null
    }
    if (typeof deck === 'undefined' || typeof deck.Deck !== 'function') {
      _mountError(node, 'renderUMAP: deck.gl failed to load from CDN')
      return null
    }

    const tooltipEl = node.querySelector('#umap-tooltip')
    const lookupTitle = _makeTitleCache()

    const rect = node.getBoundingClientRect ? node.getBoundingClientRect() : null
    const width = rect && rect.width > 0 ? rect.width : 1100
    const height = rect && rect.height > 0 ? rect.height : 600

    const bounds = _computeBounds(points)
    const initialViewState = _initialViewState(bounds, width, height)

    // Tally per-community counts for the legend + stats line.
    const communityCounts = new Map()
    for (let i = 0; i < points.length; i += 1) {
      const cid = points[i].community_id
      if (cid == null) continue
      communityCounts.set(cid, (communityCounts.get(cid) || 0) + 1)
    }

    // Mutable ref so legend clicks can update the scatter layer's color fn.
    const activeCommunity = { value: null }

    const scatter = new deck.ScatterplotLayer({
      id: 'umap-scatter',
      data: points,
      pickable: true,
      radiusUnits: 'pixels',
      getPosition: function (d) {
        return [Number(d.x) || 0, Number(d.y) || 0]
      },
      getRadius: 2.5,
      getFillColor: function (d) {
        const base = _colorForCommunity(d.community_id)
        if (activeCommunity.value == null) return base
        if (Number(d.community_id) === Number(activeCommunity.value)) return base
        return [210, 210, 210] // dim non-selected
      },
      updateTriggers: {
        getFillColor: [activeCommunity],
      },
      opacity: 0.75,
    })

    const instance = new deck.Deck({
      parent: node,
      width: '100%',
      height: '100%',
      views: new deck.OrthographicView({ id: 'ortho' }),
      controller: true,
      initialViewState: initialViewState,
      layers: [scatter],
      onHover: function (info) {
        if (!info || !info.object) {
          _hideTooltip(tooltipEl)
          return
        }
        const p = info.object
        // Show bibcode immediately; resolve title asynchronously.
        _showTooltip(tooltipEl, info.x, info.y, String(p.bibcode || ''))
        if (p.bibcode) {
          lookupTitle(p.bibcode, function (title) {
            // Only overwrite if the tooltip is still showing this point.
            if (tooltipEl && tooltipEl.style.display !== 'none' && title) {
              _showTooltip(tooltipEl, info.x, info.y, title)
            }
          })
        }
      },
      onClick: function (info) {
        if (!info || !info.object) return
        const p = info.object
        // eslint-disable-next-line no-console
        console.log('[umap] click', {
          bibcode: p.bibcode,
          community_id: p.community_id,
          x: p.x,
          y: p.y,
        })
        _updatePanel(p.bibcode, '', p.community_id)
        if (p.bibcode) {
          lookupTitle(p.bibcode, function (title) {
            _updatePanel(p.bibcode, title, p.community_id)
          })
        }
      },
    })

    _populateLegend(communityCounts, activeCommunity, function (newPick) {
      // Force deck.gl to re-evaluate getFillColor by swapping the layer.
      instance.setProps({
        layers: [
          scatter.clone({
            updateTriggers: {
              getFillColor: [{ value: newPick }],
            },
            getFillColor: function (d) {
              const base = _colorForCommunity(d.community_id)
              if (newPick == null) return base
              if (Number(d.community_id) === Number(newPick)) return base
              return [215, 215, 215]
            },
          }),
        ],
      })
      _updateStats(points.length, newPick, communityCounts)
    })
    _updateStats(points.length, null, communityCounts)

    return instance
  }

  // Public export. Assigned to window so tests and the bootstrap script can
  // locate the symbol by name.
  if (typeof window !== 'undefined') {
    window.renderUMAP = renderUMAP
  }
})()
