/*
 * web/viz/heatmap.js — community × community citation topology.
 *
 * Exposes window.renderHeatmap(data, labels, container, options).
 *   data    — output of scripts/viz/build_citation_heatmap.py
 *   labels  — output of scripts/viz/compute_community_labels.py (nullable)
 *   options — { resolution: 'coarse'|'medium', defaultMode: 'absolute'|'row'|'lift' }
 *
 * Three color modes:
 *   - absolute:  raw cell count (linear)
 *   - row:       cell / row_total   (what does community X cite?)
 *   - lift:      (cell/row_total) / (col_total/grand)   (observed vs expected)
 *
 * For small n (≤25) we render a labelled D3 SVG. For large n we switch to
 * a canvas-based zoomable matrix (d3-zoom for pan+scroll-zoom, per-cell
 * hover tooltip, click-pin into the side panel).
 *
 * 2-space indent, single quotes, vanilla ES2020, relies on d3@7 global.
 */
;(function () {
  'use strict'

  const SVG_THRESHOLD = 25

  // --- data helpers ---------------------------------------------------------

  function _labelFor(cid, labels) {
    if (!labels || !Array.isArray(labels.communities)) return 'c' + cid
    const entry = labels.communities.find(function (c) {
      return c.community_id === cid
    })
    if (!entry || !entry.terms || !entry.terms.length) return 'c' + cid
    return entry.terms.slice(0, 2).join(' / ')
  }

  function _fullLabelFor(cid, labels) {
    if (!labels || !Array.isArray(labels.communities)) return 'c' + cid
    const entry = labels.communities.find(function (c) {
      return c.community_id === cid
    })
    if (!entry || !entry.terms) return 'c' + cid
    return 'c' + cid + ' · ' + entry.terms.join(', ')
  }

  function _termsFor(cid, labels) {
    if (!labels || !Array.isArray(labels.communities)) return []
    const entry = labels.communities.find(function (c) {
      return c.community_id === cid
    })
    return entry && entry.terms ? entry.terms : []
  }

  function _cellMatrix(data) {
    const cids = data.communities.slice()
    const idx = new Map(cids.map(function (c, i) { return [c, i] }))
    const n = cids.length
    const m = []
    for (let i = 0; i < n; i += 1) {
      m.push(new Array(n).fill(0))
    }
    data.cells.forEach(function (c) {
      const si = idx.get(c.src)
      const ti = idx.get(c.tgt)
      if (si == null || ti == null) return
      m[si][ti] = c.n
    })
    return { matrix: m, order: cids, rowTotals: data.row_totals, colTotals: data.col_totals, grand: data.grand_total }
  }

  function _valueFn(mode) {
    if (mode === 'row') {
      return function (m, i, j, rowTotals) {
        const r = rowTotals[i] || 1
        return m[i][j] / r
      }
    }
    if (mode === 'lift') {
      return function (m, i, j, rowTotals, colTotals, grand) {
        const r = rowTotals[i] || 1
        const expected = ((colTotals[j] || 1) / grand) * r
        if (expected <= 0) return 0
        return m[i][j] / expected
      }
    }
    return function (m, i, j) {
      return m[i][j]
    }
  }

  function _scaleFor(mode, values) {
    if (mode === 'absolute') {
      const max = d3.max(values) || 1
      return d3.scaleSequentialLog().domain([1, max]).interpolator(d3.interpolateBlues)
    }
    if (mode === 'row') {
      const max = d3.max(values) || 1
      return d3.scaleSequential().domain([0, max]).interpolator(d3.interpolateBlues)
    }
    // lift — diverging around 1.0
    const max = d3.max(values) || 2
    const domain = [0, 1, Math.max(2, max)]
    return d3.scaleDiverging().domain(domain).interpolator(d3.interpolateRdBu).clamp(true)
  }

  function _colorFor(v, scale, mode) {
    if (mode === 'absolute') {
      return v > 0 ? scale(v) : '#f9f9f9'
    }
    return scale(v || 0)
  }

  function _updateLegend(mode, values) {
    const legendMin = document.getElementById('hm-legend-min')
    const legendMax = document.getElementById('hm-legend-max')
    const grad = document.getElementById('hm-legend-grad')
    if (grad) {
      grad.classList.toggle('diverging', mode === 'lift')
    }
    if (!legendMin || !legendMax) return
    if (mode === 'absolute') {
      legendMin.textContent = '1'
      legendMax.textContent = (d3.max(values) || 0).toLocaleString()
    } else if (mode === 'row') {
      legendMin.textContent = '0%'
      legendMax.textContent = Math.round(100 * (d3.max(values) || 0)) + '%'
    } else {
      legendMin.textContent = '0× (under)'
      legendMax.textContent = (d3.max(values) || 0).toFixed(1) + '× (over)'
    }
  }

  function _paintPanel(d, ctx) {
    const { mode, labels, rowTotals, colTotals } = ctx
    const panel = document.getElementById('heatmap-panel')
    if (!panel) return
    const srcName = _fullLabelFor(d.srcCid, labels)
    const tgtName = _fullLabelFor(d.tgtCid, labels)
    const srcTerms = _termsFor(d.srcCid, labels)
    const tgtTerms = _termsFor(d.tgtCid, labels)
    const rowShare = rowTotals[d.i] ? ((d.raw / rowTotals[d.i]) * 100).toFixed(2) : '0'
    const colShare = colTotals[d.j] ? ((d.raw / colTotals[d.j]) * 100).toFixed(2) : '0'
    panel.innerHTML =
      '<h2>Cited by → Cites</h2>' +
      '<div><span class="pill">source</span> ' + srcName + '</div>' +
      (srcTerms.length ? '<div class="term-list">top terms: ' + srcTerms.join(', ') + '</div>' : '') +
      '<div style="margin-top:8px"><span class="pill">target</span> ' + tgtName + '</div>' +
      (tgtTerms.length ? '<div class="term-list">top terms: ' + tgtTerms.join(', ') + '</div>' : '') +
      '<div style="margin-top:14px"><strong>' + d.raw.toLocaleString() + '</strong> citation edges</div>' +
      '<div style="color:#555;margin-top:4px">' + rowShare + '% of source’s outbound citations</div>' +
      '<div style="color:#555">' + colShare + '% of target’s inbound citations</div>' +
      (mode === 'lift' && d.lift != null
        ? '<div style="color:#444;margin-top:8px;font-size:12px">Lift <strong>' + d.lift.toFixed(2) + '×</strong> — observed / uniform-expected.</div>'
        : '')
  }

  // --- SVG renderer (small n) -----------------------------------------------

  function renderSvg(data, labels, container, opts) {
    const CELL_SIZE = 32
    const AXIS_LEFT = 210
    const AXIS_TOP = 160
    const MARGIN = { top: 18, right: 100, bottom: 18, left: 18 }

    const { matrix, order, rowTotals, colTotals, grand } = _cellMatrix(data)
    const n = order.length
    const width = AXIS_LEFT + CELL_SIZE * n + MARGIN.right
    const height = AXIS_TOP + CELL_SIZE * n + MARGIN.bottom

    // Preserve the tooltip node (it's reused by canvas path too) but clear svg.
    const tooltip = document.getElementById('heatmap-tooltip')
    container.querySelectorAll('svg').forEach(function (n) { n.remove() })
    const svg = d3
      .select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', '0 0 ' + width + ' ' + height)

    function draw(mode) {
      const vfn = _valueFn(mode)
      const values = []
      for (let i = 0; i < n; i += 1) {
        for (let j = 0; j < n; j += 1) {
          values.push(vfn(matrix, i, j, rowTotals, colTotals, grand))
        }
      }
      const scale = _scaleFor(mode, values)
      _updateLegend(mode, values)

      const cells = svg.selectAll('.hm-cell').data(
        values.map(function (v, k) {
          const i = Math.floor(k / n)
          const j = k % n
          return {
            i: i, j: j, v: v, raw: matrix[i][j],
            lift: mode === 'lift' ? v : null,
            srcCid: order[i], tgtCid: order[j],
          }
        }),
      )
      cells
        .enter()
        .append('rect')
        .attr('class', 'hm-cell')
        .merge(cells)
        .attr('x', function (d) { return AXIS_LEFT + d.j * CELL_SIZE })
        .attr('y', function (d) { return AXIS_TOP + d.i * CELL_SIZE })
        .attr('width', CELL_SIZE - 1)
        .attr('height', CELL_SIZE - 1)
        .attr('rx', 2)
        .attr('fill', function (d) { return _colorFor(d.v, scale, mode) })
        .on('mouseenter', function (ev, d) {
          d3.select(this).attr('stroke', '#000').attr('stroke-width', 2)
          if (tooltip) {
            tooltip.style.display = 'block'
            tooltip.innerHTML =
              '<strong>' + _labelFor(d.srcCid, labels) + ' → ' + _labelFor(d.tgtCid, labels) + '</strong>' +
              '<br/>' + d.raw.toLocaleString() + ' edges' +
              (mode === 'lift' ? '<br/>lift ' + (d.lift || 0).toFixed(2) + '×' : '')
          }
        })
        .on('mousemove', function (ev) {
          if (!tooltip) return
          const rect = container.getBoundingClientRect()
          tooltip.style.left = ev.clientX - rect.left + 14 + 'px'
          tooltip.style.top = ev.clientY - rect.top + 14 + 'px'
        })
        .on('mouseleave', function () {
          d3.select(this).attr('stroke', 'transparent')
          if (tooltip) tooltip.style.display = 'none'
        })
        .on('click', function (_ev, d) {
          _paintPanel(d, { mode: mode, labels: labels, rowTotals: rowTotals, colTotals: colTotals })
        })
    }

    // Row labels (source = Y-axis).
    svg
      .append('g')
      .attr('class', 'hm-axis')
      .selectAll('text')
      .data(order)
      .join('text')
      .attr('x', AXIS_LEFT - 8)
      .attr('y', function (_d, i) { return AXIS_TOP + i * CELL_SIZE + CELL_SIZE / 2 })
      .attr('text-anchor', 'end')
      .attr('dominant-baseline', 'middle')
      .text(function (d) { return _labelFor(d, labels) })
      .append('title')
      .text(function (d) { return _fullLabelFor(d, labels) })

    svg
      .append('g')
      .attr('class', 'hm-axis')
      .selectAll('text')
      .data(order)
      .join('text')
      .attr('x', 0)
      .attr('y', 0)
      .attr('transform', function (_d, i) {
        return (
          'translate(' + (AXIS_LEFT + i * CELL_SIZE + CELL_SIZE / 2) + ', ' + (AXIS_TOP - 10) + ') rotate(-45)'
        )
      })
      .attr('text-anchor', 'start')
      .attr('dominant-baseline', 'middle')
      .text(function (d) { return _labelFor(d, labels) })
      .append('title')
      .text(function (d) { return _fullLabelFor(d, labels) })

    svg
      .append('text')
      .attr('class', 'hm-axis axis-title')
      .attr('x', AXIS_LEFT / 2)
      .attr('y', AXIS_TOP - 80)
      .attr('text-anchor', 'middle')
      .text('Source community (citing)')
    svg
      .append('text')
      .attr('class', 'hm-axis axis-title')
      .attr('transform',
        'translate(' + (AXIS_LEFT / 2 - 80) + ', ' + (AXIS_TOP + (CELL_SIZE * n) / 2) + ') rotate(-90)',
      )
      .attr('text-anchor', 'middle')
      .text('Target community (cited)')

    draw(opts.defaultMode || 'absolute')
    _installModeToggle(opts.defaultMode || 'absolute', draw)
  }

  // --- Canvas renderer (large n, zoom + pan) --------------------------------

  function renderCanvas(data, labels, container, opts) {
    const { matrix, order, rowTotals, colTotals, grand } = _cellMatrix(data)
    const n = order.length

    // Pre-compute every value matrix upfront — modes swap instantly.
    const precomputed = {
      absolute: _fillValues(matrix, n, _valueFn('absolute'), rowTotals, colTotals, grand),
      row: _fillValues(matrix, n, _valueFn('row'), rowTotals, colTotals, grand),
      lift: _fillValues(matrix, n, _valueFn('lift'), rowTotals, colTotals, grand),
    }

    const containerWidth = Math.min(1100, Math.max(600, container.clientWidth - 24))
    const MATRIX_SIZE = containerWidth
    const cellSize = MATRIX_SIZE / n

    // Clear prior renderers; keep the tooltip node.
    const tooltip = document.getElementById('heatmap-tooltip')
    container.querySelectorAll('svg, canvas').forEach(function (n) { n.remove() })

    const dpr = Math.max(1, window.devicePixelRatio || 1)
    const canvas = document.createElement('canvas')
    canvas.id = 'heatmap-canvas'
    canvas.width = MATRIX_SIZE * dpr
    canvas.height = MATRIX_SIZE * dpr
    canvas.style.width = MATRIX_SIZE + 'px'
    canvas.style.height = MATRIX_SIZE + 'px'
    container.appendChild(canvas)

    const ctx2d = canvas.getContext('2d')
    ctx2d.scale(dpr, dpr)

    const zoomHint = document.getElementById('zoom-hint')
    if (zoomHint) {
      zoomHint.textContent = 'Scroll to zoom · drag to pan · click a cell to pin details. ' + n + ' × ' + n + ' = ' + (n * n).toLocaleString() + ' cells.'
    }

    let currentMode = opts.defaultMode || 'lift'
    let transform = d3.zoomIdentity

    function currentScale(mode) {
      return _scaleFor(mode, precomputed[mode])
    }

    function paint() {
      const mode = currentMode
      const values = precomputed[mode]
      const scale = currentScale(mode)
      _updateLegend(mode, values)

      ctx2d.save()
      ctx2d.clearRect(0, 0, MATRIX_SIZE, MATRIX_SIZE)
      ctx2d.translate(transform.x, transform.y)
      ctx2d.scale(transform.k, transform.k)

      // Determine visible (i, j) range to skip off-screen cells when zoomed in.
      const invX0 = (-transform.x) / transform.k
      const invY0 = (-transform.y) / transform.k
      const invX1 = (MATRIX_SIZE - transform.x) / transform.k
      const invY1 = (MATRIX_SIZE - transform.y) / transform.k
      const j0 = Math.max(0, Math.floor(invX0 / cellSize))
      const j1 = Math.min(n - 1, Math.ceil(invX1 / cellSize))
      const i0 = Math.max(0, Math.floor(invY0 / cellSize))
      const i1 = Math.min(n - 1, Math.ceil(invY1 / cellSize))

      // At very low zoom, thin strokes create visual noise; suppress them.
      const stroke = transform.k > 4
      for (let i = i0; i <= i1; i += 1) {
        for (let j = j0; j <= j1; j += 1) {
          const v = values[i * n + j]
          const raw = matrix[i][j]
          const fill = mode === 'absolute' && raw <= 0 ? '#fafafa' : _colorFor(v, scale, mode)
          ctx2d.fillStyle = fill
          ctx2d.fillRect(j * cellSize, i * cellSize, cellSize, cellSize)
          if (stroke && raw > 0) {
            ctx2d.strokeStyle = 'rgba(0,0,0,0.08)'
            ctx2d.lineWidth = 0.5 / transform.k
            ctx2d.strokeRect(j * cellSize, i * cellSize, cellSize, cellSize)
          }
        }
      }

      // Overlay row/col labels when zoomed in enough that individual cells are readable.
      if (transform.k > 6) {
        ctx2d.fillStyle = '#333'
        ctx2d.font = (10 / transform.k) + 'px sans-serif'
        ctx2d.textAlign = 'right'
        ctx2d.textBaseline = 'middle'
        for (let i = i0; i <= i1; i += 1) {
          const y = (i + 0.5) * cellSize
          ctx2d.fillText(_labelFor(order[i], labels), j0 * cellSize - 2 / transform.k, y)
        }
        ctx2d.textAlign = 'left'
        ctx2d.textBaseline = 'alphabetic'
        for (let j = j0; j <= j1; j += 1) {
          const x = (j + 0.5) * cellSize
          ctx2d.save()
          ctx2d.translate(x, i0 * cellSize - 2 / transform.k)
          ctx2d.rotate(-Math.PI / 4)
          ctx2d.fillText(_labelFor(order[j], labels), 0, 0)
          ctx2d.restore()
        }
      }

      ctx2d.restore()

      // Persistent hover ring (drawn in untransformed space so it stays crisp).
      if (hoverCell) {
        ctx2d.save()
        ctx2d.translate(transform.x, transform.y)
        ctx2d.scale(transform.k, transform.k)
        ctx2d.strokeStyle = '#111'
        ctx2d.lineWidth = 1.5 / transform.k
        ctx2d.strokeRect(hoverCell.j * cellSize, hoverCell.i * cellSize, cellSize, cellSize)
        ctx2d.restore()
      }
    }

    let hoverCell = null

    function hit(ev) {
      const rect = canvas.getBoundingClientRect()
      const x = ev.clientX - rect.left
      const y = ev.clientY - rect.top
      const ix = (x - transform.x) / transform.k
      const iy = (y - transform.y) / transform.k
      const j = Math.floor(ix / cellSize)
      const i = Math.floor(iy / cellSize)
      if (i < 0 || i >= n || j < 0 || j >= n) return null
      return { i: i, j: j }
    }

    function cellRecord(hitRes) {
      const mode = currentMode
      const v = precomputed[mode][hitRes.i * n + hitRes.j]
      return {
        i: hitRes.i,
        j: hitRes.j,
        raw: matrix[hitRes.i][hitRes.j],
        v: v,
        lift: mode === 'lift' ? v : precomputed.lift[hitRes.i * n + hitRes.j],
        srcCid: order[hitRes.i],
        tgtCid: order[hitRes.j],
      }
    }

    canvas.addEventListener('mousemove', function (ev) {
      const hitRes = hit(ev)
      if (!hitRes) {
        if (hoverCell) {
          hoverCell = null
          paint()
        }
        if (tooltip) tooltip.style.display = 'none'
        return
      }
      if (!hoverCell || hoverCell.i !== hitRes.i || hoverCell.j !== hitRes.j) {
        hoverCell = hitRes
        paint()
      }
      if (tooltip) {
        const rec = cellRecord(hitRes)
        const mode = currentMode
        tooltip.style.display = 'block'
        tooltip.innerHTML =
          '<strong>' + _labelFor(rec.srcCid, labels) + ' → ' + _labelFor(rec.tgtCid, labels) + '</strong>' +
          '<br/>' + rec.raw.toLocaleString() + ' edges' +
          (mode === 'lift' ? '<br/>lift ' + (rec.lift || 0).toFixed(2) + '×' : '') +
          '<br/><span style="color:#bbb">click to pin</span>'
        const rect = container.getBoundingClientRect()
        tooltip.style.left = ev.clientX - rect.left + 14 + 'px'
        tooltip.style.top = ev.clientY - rect.top + 14 + 'px'
      }
    })

    canvas.addEventListener('mouseleave', function () {
      hoverCell = null
      if (tooltip) tooltip.style.display = 'none'
      paint()
    })

    // Click — d3-zoom swallows mouseup, so listen for `click` instead
    // (the browser only synthesises `click` when down+up happen on the same
    // element without a drag, which is exactly what we want for pin).
    let dragStart = null
    canvas.addEventListener('pointerdown', function (ev) {
      dragStart = { x: ev.clientX, y: ev.clientY }
      canvas.classList.add('dragging')
    })
    canvas.addEventListener('pointerup', function () {
      canvas.classList.remove('dragging')
    })
    canvas.addEventListener('click', function (ev) {
      if (dragStart) {
        const dx = ev.clientX - dragStart.x
        const dy = ev.clientY - dragStart.y
        dragStart = null
        if (Math.abs(dx) > 3 || Math.abs(dy) > 3) return
      }
      const hitRes = hit(ev)
      if (!hitRes) return
      const rec = cellRecord(hitRes)
      _paintPanel(rec, { mode: currentMode, labels: labels, rowTotals: rowTotals, colTotals: colTotals })
    })

    const zoom = d3
      .zoom()
      .scaleExtent([1, 40])
      .translateExtent([
        [0, 0],
        [MATRIX_SIZE, MATRIX_SIZE],
      ])
      .on('zoom', function (ev) {
        transform = ev.transform
        paint()
      })
    d3.select(canvas).call(zoom)

    _installModeToggle(currentMode, function (mode) {
      currentMode = mode
      paint()
    })

    paint()
  }

  function _fillValues(matrix, n, vfn, rowTotals, colTotals, grand) {
    const out = new Array(n * n)
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        out[i * n + j] = vfn(matrix, i, j, rowTotals, colTotals, grand)
      }
    }
    return out
  }

  function _installModeToggle(initialMode, onChange) {
    const btns = document.querySelectorAll('#mode-toggle button')
    btns.forEach(function (b) {
      b.classList.toggle('active', b.dataset.mode === initialMode)
    })
    btns.forEach(function (btn) {
      btn.onclick = function () {
        btns.forEach(function (b) { b.classList.remove('active') })
        btn.classList.add('active')
        onChange(btn.dataset.mode || 'absolute')
      }
    })
  }

  // --- Entry ----------------------------------------------------------------

  function renderHeatmap(data, labels, container, options) {
    if (typeof d3 === 'undefined') {
      container.textContent = 'd3 failed to load'
      return
    }
    const opts = options || {}
    const n = (data.communities || []).length
    if (n <= SVG_THRESHOLD) {
      renderSvg(data, labels, container, opts)
    } else {
      renderCanvas(data, labels, container, opts)
    }
  }

  if (typeof window !== 'undefined') {
    window.renderHeatmap = renderHeatmap
  }
})()
