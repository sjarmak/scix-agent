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

  // Community-label cache — filled by _loadCommunityLabels() before first
  // render; maps community_id (number) -> {terms: [...], n_sampled: N}.
  let _communityLabels = {}

  function _labelForCommunity(cid) {
    if (cid == null) return 'community —'
    const entry = _communityLabels[cid]
    if (entry && entry.terms && entry.terms.length) {
      return entry.terms.slice(0, 3).join(' / ')
    }
    return 'community ' + cid
  }

  function _loadCommunityLabels() {
    return fetch('/viz/community_labels.json', { cache: 'no-store' })
      .then(function (resp) {
        return resp.ok ? resp.json() : null
      })
      .then(function (payload) {
        if (!payload || !Array.isArray(payload.communities)) return
        _communityLabels = {}
        payload.communities.forEach(function (c) {
          if (c && c.community_id != null) _communityLabels[c.community_id] = c
        })
      })
      .catch(function () {})
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
    const safeBib = String(bibcode).replace(/[<>&]/g, '')
    const egoHref =
      './ego.html?bibcode=' + encodeURIComponent(String(bibcode))
    panel.innerHTML =
      '<h2>Selected paper</h2>' +
      '<div><strong>bibcode:</strong> <code>' +
      safeBib +
      '</code></div>' +
      '<div><strong>community:</strong> ' +
      cidText +
      '</div>' +
      '<div style="margin-top:6px">' +
      String(safeTitle).replace(/[<>]/g, '') +
      '</div>' +
      '<div style="margin-top:10px">' +
      '<a href="' +
      egoHref +
      '" target="_blank" rel="noopener" class="ego-link">' +
      'Open citation ego network →' +
      '</a>' +
      '</div>'
  }

  function _populateLegend(communityCounts, activeCommunityRef, onPick) {
    const grid = document.getElementById('umap-legend-grid')
    if (!grid) return
    grid.innerHTML = ''
    const ids = Array.from(communityCounts.keys()).sort(function (a, b) {
      // Order by paper count desc so biggest communities come first.
      return communityCounts.get(b) - communityCounts.get(a)
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
      const name = _labelForCommunity(cid)
      const label =
        '<span class="legend-name">' +
        name.replace(/[<>]/g, '') +
        '</span> <span style="color:#888">(' +
        count.toLocaleString() +
        ')</span>'
      const el = document.createElement('div')
      el.className = 'legend-swatch'
      el.dataset.cid = String(cid)
      el.innerHTML = chip + '<span>' + label + '</span>'
      el.title = 'c' + cid + ' · ' + count.toLocaleString() + ' papers in sample'
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

  function _communityCentroids(points) {
    const acc = new Map()
    for (let i = 0; i < points.length; i += 1) {
      const p = points[i]
      const cid = p.community_id
      if (cid == null) continue
      const e = acc.get(cid) || { sx: 0, sy: 0, n: 0 }
      e.sx += Number(p.x) || 0
      e.sy += Number(p.y) || 0
      e.n += 1
      acc.set(cid, e)
    }
    const out = []
    acc.forEach(function (e, cid) {
      if (e.n > 0) {
        out.push({
          community_id: cid,
          x: e.sx / e.n,
          y: e.sy / e.n,
          n: e.n,
          label: _labelForCommunity(cid),
        })
      }
    })
    return out
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
      getRadius: 2,
      getFillColor: function (d) {
        const base = _colorForCommunity(d.community_id)
        if (activeCommunity.value == null) return base
        if (Number(d.community_id) === Number(activeCommunity.value)) return base
        return [210, 210, 210] // dim non-selected
      },
      updateTriggers: {
        getFillColor: [activeCommunity],
      },
      opacity: 0.7,
    })

    // Compute centroids for label overlay. Top-N (by paper count) are
    // rendered as halo labels directly on the canvas so the clusters are
    // legible without hovering.
    const centroids = _communityCentroids(points)
    centroids.sort(function (a, b) {
      return b.n - a.n
    })
    // Fewer labels at default zoom so they don't overlap; we'll reveal more
    // as the user zooms in (handled in onViewStateChange below).
    const topCentroids = centroids.slice(0, Math.min(8, centroids.length))
    const allCentroids = centroids.slice(0, Math.min(16, centroids.length))

    let labelLayer = null
    function _makeLabelLayer(dataOverride, sizeOverride) {
      if (!deck.TextLayer) return null
      return new deck.TextLayer({
        id: 'umap-labels',
        data: dataOverride || topCentroids,
        pickable: false,
        getPosition: function (d) {
          return [d.x, d.y]
        },
        getText: function (d) {
          return d.label
        },
        getColor: [15, 15, 15, 255],
        getSize: sizeOverride || 18,
        sizeUnits: 'pixels',
        sizeScale: 1,
        fontWeight: 800,
        fontSettings: { sdf: true },
        background: true,
        getBackgroundColor: [255, 255, 255, 235],
        backgroundPadding: [8, 5],
        getBorderColor: [0, 0, 0, 120],
        getBorderWidth: 1.25,
        outlineColor: [255, 255, 255, 255],
        outlineWidth: 3,
        getTextAnchor: 'middle',
        getAlignmentBaseline: 'center',
        billboard: true,
      })
    }
    labelLayer = _makeLabelLayer()

    // Track the latest view state so label selection can react to zoom.
    let _currentZoom = initialViewState.zoom
    function _rebuildLabelsForZoom() {
      if (!labelLayer) return null
      // At low zoom show only the top-8 centroids with bigger font; as the
      // user zooms in, reveal more labels at a smaller font so text fits
      // the denser regions without piling up.
      const zoom = _currentZoom
      const baseZoom = initialViewState.zoom
      const delta = zoom - baseZoom
      let count, size
      if (delta < 0.5) {
        count = Math.min(8, topCentroids.length)
        size = 18
      } else if (delta < 2) {
        count = Math.min(12, allCentroids.length)
        size = 16
      } else {
        count = allCentroids.length
        size = 14
      }
      return _makeLabelLayer(allCentroids.slice(0, count), size)
    }

    const instance = new deck.Deck({
      parent: node,
      width: '100%',
      height: '100%',
      views: new deck.OrthographicView({ id: 'ortho' }),
      controller: true,
      initialViewState: initialViewState,
      onViewStateChange: function (evt) {
        if (evt && evt.viewState && typeof evt.viewState.zoom === 'number') {
          const prev = _currentZoom
          _currentZoom = evt.viewState.zoom
          if (Math.abs(prev - _currentZoom) > 0.3) {
            // Only rebuild the labels layer on meaningful zoom changes.
            labelLayer = _rebuildLabelsForZoom()
            const existing = (instance.props.layers || []).filter(function (l) {
              return l && l.id !== 'umap-labels'
            })
            instance.setProps({ layers: labelLayer ? existing.concat([labelLayer]) : existing })
          }
        }
      },
      layers: labelLayer ? [scatter, labelLayer] : [scatter],
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
      const nextScatter = scatter.clone({
        updateTriggers: {
          getFillColor: [{ value: newPick }],
        },
        getFillColor: function (d) {
          const base = _colorForCommunity(d.community_id)
          if (newPick == null) return base
          if (Number(d.community_id) === Number(newPick)) return base
          return [215, 215, 215]
        },
      })
      // When isolating one community, show only its label.
      const nextLabels = labelLayer
        ? labelLayer.clone({
            data:
              newPick == null
                ? topCentroids
                : topCentroids.filter(function (c) {
                    return Number(c.community_id) === Number(newPick)
                  }),
          })
        : null
      instance.setProps({ layers: nextLabels ? [nextScatter, nextLabels] : [nextScatter] })
      _updateStats(points.length, newPick, communityCounts)
    })
    _updateStats(points.length, null, communityCounts)

    // Labels load asynchronously; once they arrive, re-populate the legend
    // with the named communities AND re-run the zoom-reactive label selector
    // so centroids don't freeze on the pre-label "c5" names.
    _loadCommunityLabels().then(function () {
      const refreshedCentroids = _communityCentroids(points).sort(function (a, b) {
        return b.n - a.n
      })
      // Mutate the outer arrays in place so _rebuildLabelsForZoom picks up
      // the new labels without needing a second closure.
      topCentroids.length = 0
      allCentroids.length = 0
      refreshedCentroids.slice(0, Math.min(8, refreshedCentroids.length)).forEach(
        function (c) { topCentroids.push(c) },
      )
      refreshedCentroids.slice(0, Math.min(16, refreshedCentroids.length)).forEach(
        function (c) { allCentroids.push(c) },
      )
      _populateLegend(communityCounts, activeCommunity, function (newPick) {
        const nextScatter = scatter.clone({
          updateTriggers: { getFillColor: [{ value: newPick }] },
          getFillColor: function (d) {
            const base = _colorForCommunity(d.community_id)
            if (newPick == null) return base
            if (Number(d.community_id) === Number(newPick)) return base
            return [215, 215, 215]
          },
        })
        const base = _rebuildLabelsForZoom() || labelLayer
        const nextLabels = base
          ? base.clone({
              data:
                newPick == null
                  ? base.props.data
                  : allCentroids.filter(function (c) {
                      return Number(c.community_id) === Number(newPick)
                    }),
            })
          : null
        instance.setProps({ layers: nextLabels ? [nextScatter, nextLabels] : [nextScatter] })
        _updateStats(points.length, newPick, communityCounts)
      })
      labelLayer = _rebuildLabelsForZoom()
      if (labelLayer) {
        const otherLayers = (instance.props.layers || []).filter(function (l) {
          return l && l.id !== 'umap-labels'
        })
        instance.setProps({ layers: otherLayers.concat([labelLayer]) })
      }
    })

    return instance
  }

  // Public export. Assigned to window so tests and the bootstrap script can
  // locate the symbol by name.
  if (typeof window !== 'undefined') {
    window.renderUMAP = renderUMAP
  }
})()
