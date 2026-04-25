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

  // Color comes from window.scixViz.colorForCommunity in shared.js — a
  // resolution-aware generator that returns the hand-tuned 20-color palette
  // for coarse and a golden-ratio HSL walk for medium/fine. Keeping it
  // central keeps the UMAP and ego views in sync without "keep this list in
  // sync" comments. We fall back to neutral grey if shared.js isn't loaded.
  const FALLBACK_COLOR = [160, 160, 160]

  // Ego-proximity palette — used when an "ego overlay" is active. The hue
  // ramp goes warm (citation neighbour) → cool (2-hop) → grey (outside the
  // neighbourhood) so the eye can read distance at a glance even when the
  // surrounding community palette is busy.
  const EGO_COLORS = {
    CENTER: [255, 199, 44], // bright gold — the pinned paper itself
    DIRECT: [240, 124, 60], // warm orange — direct refs / cites (1 hop)
    SECOND: [82, 132, 199], // cool blue — 2-hop neighbours
    OUTSIDE: [225, 225, 225], // light grey — out-of-neighbourhood
  }

  function _colorForCommunity(cid) {
    var scx = (typeof window !== 'undefined' && window.scixViz) || null
    if (scx && typeof scx.colorForCommunity === 'function') {
      return scx.colorForCommunity(cid)
    }
    if (cid == null || Number.isNaN(Number(cid))) return FALLBACK_COLOR
    return FALLBACK_COLOR
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
    // Use the shared resolution config when available so switching to
    // medium/fine picks the matching label bundle; fall back to the
    // historical hardcoded coarse URL for backwards compatibility.
    var scx = (typeof window !== 'undefined' && window.scixViz) || null
    if (scx && typeof scx.resolutionFiles === 'function' && typeof scx.fetchFirstAvailable === 'function') {
      var cfg = scx.resolutionFiles()
      return scx
        .fetchFirstAvailable(cfg.labels)
        .then(function (payload) {
          if (!payload || !Array.isArray(payload.communities)) return
          _communityLabels = {}
          payload.communities.forEach(function (c) {
            if (c && c.community_id != null) _communityLabels[c.community_id] = c
          })
        })
        .catch(function () {})
    }
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

  // Compute a {bibcode -> distance} map from a /viz/api/ego/ payload.
  // Distance 1 = direct refs and direct cites; distance 2 = second-hop
  // neighbours. The center bibcode is handled by the caller (its hop
  // distance of zero is always implicit).
  function _distancesFromEgoPayload(payload) {
    const distances = Object.create(null)
    function _set(node, dist) {
      if (!node || !node.bibcode) return
      const bib = String(node.bibcode)
      // Don't downgrade an existing shorter distance if the same bibcode
      // shows up at two hop counts (defensive — the API shouldn't return
      // the center inside refs/cites, but second_hop_sample can overlap
      // with direct neighbours in rare graph topologies).
      if (distances[bib] != null && distances[bib] <= dist) return
      distances[bib] = dist
    }
    if (payload && Array.isArray(payload.direct_refs)) {
      payload.direct_refs.forEach(function (n) { _set(n, 1) })
    }
    if (payload && Array.isArray(payload.direct_cites)) {
      payload.direct_cites.forEach(function (n) { _set(n, 1) })
    }
    if (payload && Array.isArray(payload.second_hop_sample)) {
      payload.second_hop_sample.forEach(function (n) { _set(n, 2) })
    }
    return distances
  }

  // Render or hide the small distance-key legend that appears beneath the
  // Selected-paper panel while an ego overlay is active.
  function _renderEgoLegend(payload) {
    const panel = document.getElementById('umap-panel')
    if (!panel) return
    const existing = panel.querySelector('#ego-distance-legend')
    if (!payload) {
      if (existing) existing.remove()
      return
    }
    const counts = {
      center: 1,
      direct:
        ((payload.counts && payload.counts.direct_refs) || 0) +
        ((payload.counts && payload.counts.direct_cites) || 0),
      second: (payload.counts && payload.counts.second_hop) || 0,
    }
    const html =
      '<div id="ego-distance-legend" style="margin-top:10px;font-size:11px;color:#444">' +
      '<div style="font-weight:600;margin-bottom:4px">Citation distance</div>' +
      '<div style="display:flex;gap:10px;flex-wrap:wrap">' +
      '<span><span class="chip" style="background:rgb(' +
      EGO_COLORS.CENTER.join(',') + ')"></span> pinned paper</span>' +
      '<span><span class="chip" style="background:rgb(' +
      EGO_COLORS.DIRECT.join(',') + ')"></span> 1 hop (' + counts.direct + ')</span>' +
      '<span><span class="chip" style="background:rgb(' +
      EGO_COLORS.SECOND.join(',') + ')"></span> 2 hops (' + counts.second + ')</span>' +
      '<span><span class="chip" style="background:rgb(' +
      EGO_COLORS.OUTSIDE.join(',') + ')"></span> outside</span>' +
      '</div></div>'
    if (existing) {
      existing.outerHTML = html
    } else {
      panel.insertAdjacentHTML('beforeend', html)
    }
  }

  // Fetch the ego network for ``bibcode`` and ask the live UMAP instance to
  // recolor by hop distance. Surfaces failures inline on the side panel so
  // the user knows when the prod DB hasn't seen the paper.
  function _activateEgoOverlay(bibcode) {
    const inst = (typeof window !== 'undefined' && window.umapInstance) || null
    if (!inst || typeof inst.setEgoOverlay !== 'function') return
    const status = document.getElementById('ego-overlay-status')
    if (status) status.textContent = 'loading…'
    const url = '/viz/api/ego/' + encodeURIComponent(bibcode)
    fetch(url, { cache: 'no-store' })
      .then(function (resp) {
        if (!resp.ok) {
          throw new Error('HTTP ' + resp.status)
        }
        return resp.json()
      })
      .then(function (payload) {
        const distances = _distancesFromEgoPayload(payload)
        inst.setEgoOverlay({
          center: String(bibcode),
          distances: distances,
          counts: payload && payload.counts,
        })
        if (status) status.textContent = ''
      })
      .catch(function (err) {
        if (status) {
          status.textContent =
            'Could not load neighborhood (' + (err.message || String(err)) + ')'
        }
      })
  }

  function _resetEgoOverlay() {
    const inst = (typeof window !== 'undefined' && window.umapInstance) || null
    if (!inst || typeof inst.clearEgoOverlay !== 'function') return
    inst.clearEgoOverlay()
    const status = document.getElementById('ego-overlay-status')
    if (status) status.textContent = ''
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
      '<div style="margin-top:10px;display:flex;gap:6px;flex-wrap:wrap">' +
      '<button type="button" id="ego-overlay-btn" class="ego-overlay-btn">' +
      'Show neighborhood' +
      '</button>' +
      '<button type="button" id="ego-reset-btn" class="ego-overlay-btn">' +
      'Reset colors' +
      '</button>' +
      '</div>' +
      '<div id="ego-overlay-status" style="margin-top:6px;font-size:11px;color:#777;min-height:14px"></div>' +
      '<div style="margin-top:10px">' +
      '<a href="' +
      egoHref +
      '" target="_blank" rel="noopener" class="ego-link">' +
      'Open citation ego network →' +
      '</a>' +
      '</div>'

    const showBtn = panel.querySelector('#ego-overlay-btn')
    if (showBtn) {
      showBtn.addEventListener('click', function () {
        _activateEgoOverlay(bibcode)
      })
    }
    const resetBtn = panel.querySelector('#ego-reset-btn')
    if (resetBtn) {
      resetBtn.addEventListener('click', _resetEgoOverlay)
    }
  }

  // How many top communities to show by name in the legend. The remainder
  // collapse into a single "other (N communities)" row that expands inline
  // when clicked — at medium/fine resolution most communities are tiny but
  // the long tail is exactly where cross-disciplinary clusters live, so we
  // make it discoverable rather than truncating it.
  const LEGEND_TOP_N = 20

  function _renderSwatch(cid, count, activeCommunityRef, onPick) {
    const rgb = _colorForCommunity(cid)
    const chip =
      '<span class="chip" style="background:rgb(' +
      rgb[0] + ',' + rgb[1] + ',' + rgb[2] +
      ')"></span>'
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
      // Update active class across all swatches (including expanded tail).
      const grid = el.parentElement
      if (grid) {
        grid.querySelectorAll('.legend-swatch').forEach(function (sw) {
          sw.classList.toggle('active', newPick != null && Number(sw.dataset.cid) === newPick)
        })
      }
      onPick(newPick)
    })
    return el
  }

  function _populateLegend(communityCounts, activeCommunityRef, onPick) {
    const grid = document.getElementById('umap-legend-grid')
    if (!grid) return
    grid.innerHTML = ''

    const ids = Array.from(communityCounts.keys()).sort(function (a, b) {
      // Order by paper count desc so biggest communities come first.
      return communityCounts.get(b) - communityCounts.get(a)
    })
    const topIds = ids.slice(0, LEGEND_TOP_N)
    const tailIds = ids.slice(LEGEND_TOP_N)

    topIds.forEach(function (cid) {
      grid.appendChild(_renderSwatch(cid, communityCounts.get(cid), activeCommunityRef, onPick))
    })

    if (tailIds.length === 0) return

    const tailCount = tailIds.reduce(function (s, c) { return s + communityCounts.get(c) }, 0)
    const toggle = document.createElement('div')
    toggle.className = 'legend-other-toggle'
    toggle.style.cssText =
      'grid-column:1 / -1;cursor:pointer;color:#666;padding:4px 4px;' +
      'border-top:1px solid #eee;margin-top:4px;font-size:11px;' +
      'user-select:none;display:flex;align-items:center;gap:6px'
    toggle.innerHTML =
      '<span class="caret" aria-hidden="true">▸</span>' +
      '<span>other (' + tailIds.length.toLocaleString() + ' communities, ' +
      tailCount.toLocaleString() + ' papers)</span>'
    grid.appendChild(toggle)

    const tailWrap = document.createElement('div')
    tailWrap.className = 'legend-other-grid'
    tailWrap.style.cssText =
      'grid-column:1 / -1;display:none;grid-template-columns:repeat(2, 1fr);' +
      'gap:4px 10px;max-height:240px;overflow-y:auto;padding-top:4px'
    tailIds.forEach(function (cid) {
      tailWrap.appendChild(_renderSwatch(cid, communityCounts.get(cid), activeCommunityRef, onPick))
    })
    grid.appendChild(tailWrap)

    toggle.addEventListener('click', function () {
      const open = tailWrap.style.display !== 'none'
      tailWrap.style.display = open ? 'none' : 'grid'
      const caret = toggle.querySelector('.caret')
      if (caret) caret.textContent = open ? '▸' : '▾'
    })
  }

  function _updateStats(n, activeCid, communityCounts) {
    const el = document.getElementById('umap-stats')
    if (!el) return
    if (activeCid == null) {
      el.textContent =
        n.toLocaleString() + ' papers · ' +
        communityCounts.size.toLocaleString() + ' communities'
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

  function _updateLegendHeader() {
    const header = document.querySelector('#umap-legend h3')
    if (!header) return
    var scx = (typeof window !== 'undefined' && window.scixViz) || null
    var res = scx && typeof scx.getResolution === 'function' ? scx.getResolution() : 'coarse'
    header.textContent = 'Communities (' + res + ')'
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

    // Mutable refs so legend clicks (community-isolation) and the
    // Show-neighborhood / Reset-colors buttons (ego overlay) can each
    // update the scatter layer's color fn without stomping on the other.
    const activeCommunity = { value: null }
    const egoOverlay = { value: null } // null | { center, distances, counts? }

    function _colorForPoint(d) {
      // Ego overlay takes priority — it's the explicit, scoped action.
      if (egoOverlay.value) {
        const bib = String(d.bibcode || '')
        if (bib === egoOverlay.value.center) return EGO_COLORS.CENTER
        const dist =
          egoOverlay.value.distances && egoOverlay.value.distances[bib]
        if (dist === 1) return EGO_COLORS.DIRECT
        if (dist === 2) return EGO_COLORS.SECOND
        return EGO_COLORS.OUTSIDE
      }
      const base = _colorForCommunity(d.community_id)
      if (activeCommunity.value == null) return base
      if (Number(d.community_id) === Number(activeCommunity.value)) return base
      return [210, 210, 210] // dim non-selected community
    }

    const scatter = new deck.ScatterplotLayer({
      id: 'umap-scatter',
      data: points,
      pickable: true,
      radiusUnits: 'pixels',
      getPosition: function (d) {
        return [Number(d.x) || 0, Number(d.y) || 0]
      },
      getRadius: 2,
      getFillColor: _colorForPoint,
      updateTriggers: {
        getFillColor: [activeCommunity, egoOverlay],
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

    // Helper that produces a fresh scatter layer reflecting the current
    // (activeCommunity, egoOverlay) state. Used by the legend onPick and
    // by setEgoOverlay/clearEgoOverlay below — single source of truth for
    // the scatter recolor path.
    function _scatterWithCurrentColors() {
      return scatter.clone({
        updateTriggers: {
          getFillColor: [
            { ac: activeCommunity.value, eo: egoOverlay.value },
          ],
        },
        getFillColor: _colorForPoint,
      })
    }

    _populateLegend(communityCounts, activeCommunity, function (newPick) {
      const nextScatter = _scatterWithCurrentColors()
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
    _updateLegendHeader()

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
        const nextScatter = _scatterWithCurrentColors()
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

    // Public ego-overlay surface — drives recoloring from the side panel
    // buttons added by _updatePanel. Mutates the shared egoOverlay ref then
    // swaps the scatter layer for one with refreshed updateTriggers so
    // deck.gl actually re-runs getFillColor across the whole point cloud.
    function _swapScatter(nextScatter) {
      const otherLayers = (instance.props.layers || []).filter(function (l) {
        return l && l.id !== 'umap-scatter'
      })
      instance.setProps({ layers: [nextScatter].concat(otherLayers) })
    }
    instance.setEgoOverlay = function (payload) {
      egoOverlay.value = payload || null
      _swapScatter(_scatterWithCurrentColors())
      _renderEgoLegend(payload || null)
    }
    instance.clearEgoOverlay = function () {
      egoOverlay.value = null
      _swapScatter(_scatterWithCurrentColors())
      _renderEgoLegend(null)
    }

    return instance
  }

  // Public export. Assigned to window so tests and the bootstrap script can
  // locate the symbol by name.
  if (typeof window !== 'undefined') {
    window.renderUMAP = renderUMAP
  }
})()
