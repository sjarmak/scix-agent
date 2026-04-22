/*
 * web/viz/heatmap.js — community × community citation topology.
 *
 * Exposes window.renderHeatmap(data, labels, container).
 *   data   — output of scripts/viz/build_citation_heatmap.py
 *   labels — output of scripts/viz/compute_community_labels.py (nullable)
 *
 * Three color modes:
 *   - absolute:  raw cell count (linear)
 *   - row:       cell / row_total   (what does community X cite?)
 *   - lift:      (cell/row_total) / (col_total/grand)   (observed vs expected)
 *
 * 2-space indent, single quotes, vanilla ES2020, relies on d3@7 global.
 */
;(function () {
  'use strict'

  const CELL_SIZE = 32
  const AXIS_LEFT = 210
  const AXIS_TOP = 160
  const MARGIN = { top: 18, right: 100, bottom: 18, left: 18 }

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
      // Log scale so diagonals (self-citation) don't flatten everything else.
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
    if (mode === 'row') {
      return scale(v || 0)
    }
    return scale(v || 0)
  }

  function renderHeatmap(data, labels, container) {
    if (typeof d3 === 'undefined') {
      container.textContent = 'd3 failed to load'
      return
    }
    const { matrix, order, rowTotals, colTotals, grand } = _cellMatrix(data)
    const n = order.length
    const width = AXIS_LEFT + CELL_SIZE * n + MARGIN.right
    const height = AXIS_TOP + CELL_SIZE * n + MARGIN.bottom

    container.innerHTML = ''
    const svg = d3
      .select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', '0 0 ' + width + ' ' + height)

    const panel = document.getElementById('heatmap-panel')
    const legendMin = document.getElementById('hm-legend-min')
    const legendMax = document.getElementById('hm-legend-max')
    let currentMode = 'absolute'

    function draw(mode) {
      currentMode = mode
      const vfn = _valueFn(mode)
      const values = []
      for (let i = 0; i < n; i += 1) {
        for (let j = 0; j < n; j += 1) {
          values.push(vfn(matrix, i, j, rowTotals, colTotals, grand))
        }
      }
      const scale = _scaleFor(mode, values)

      // Legend labels.
      if (legendMin && legendMax) {
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

      const cells = svg.selectAll('.hm-cell').data(
        values.map(function (v, k) {
          const i = Math.floor(k / n)
          const j = k % n
          return { i: i, j: j, v: v, raw: matrix[i][j], srcCid: order[i], tgtCid: order[j] }
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
        .on('mouseenter', function (_ev, d) {
          d3.select(this).attr('stroke', '#000').attr('stroke-width', 2)
        })
        .on('mouseleave', function () {
          d3.select(this).attr('stroke', 'transparent')
        })
        .on('click', function (_ev, d) {
          const srcName = _fullLabelFor(d.srcCid, labels)
          const tgtName = _fullLabelFor(d.tgtCid, labels)
          const rowShare = rowTotals[d.i] ? ((d.raw / rowTotals[d.i]) * 100).toFixed(2) : '0'
          const colShare = colTotals[d.j] ? ((d.raw / colTotals[d.j]) * 100).toFixed(2) : '0'
          panel.innerHTML =
            '<h2>Cited by → Cites</h2>' +
            '<div><span class="pill">source</span> ' + srcName + '</div>' +
            '<div style="margin-top:4px"><span class="pill">target</span> ' + tgtName + '</div>' +
            '<div style="margin-top:14px"><strong>' + d.raw.toLocaleString() + '</strong> citation edges</div>' +
            '<div style="color:#555;margin-top:4px">' + rowShare + '% of source’s outbound citations</div>' +
            '<div style="color:#555">' + colShare + '% of target’s inbound citations</div>' +
            (mode === 'lift'
              ? '<div style="color:#444;margin-top:8px;font-size:12px">Lift = observed / uniform-expected. > 1 means this pair cites each other more than random.</div>'
              : '')
        })
        .append('title')
        .text(function (d) {
          return (
            _labelFor(d.srcCid, labels) +
            '  →  ' +
            _labelFor(d.tgtCid, labels) +
            '\n' +
            d.raw.toLocaleString() +
            ' edges'
          )
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

    // Column labels (target = X-axis), rotated.
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
          'translate(' +
          (AXIS_LEFT + i * CELL_SIZE + CELL_SIZE / 2) +
          ', ' +
          (AXIS_TOP - 10) +
          ') rotate(-45)'
        )
      })
      .attr('text-anchor', 'start')
      .attr('dominant-baseline', 'middle')
      .text(function (d) { return _labelFor(d, labels) })
      .append('title')
      .text(function (d) { return _fullLabelFor(d, labels) })

    // Axis titles.
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
        'translate(' +
        (AXIS_LEFT / 2 - 80) +
        ', ' +
        (AXIS_TOP + (CELL_SIZE * n) / 2) +
        ') rotate(-90)',
      )
      .attr('text-anchor', 'middle')
      .text('Target community (cited)')

    draw('absolute')

    // Mode toggle.
    document.querySelectorAll('#mode-toggle button').forEach(function (btn) {
      btn.addEventListener('click', function () {
        document.querySelectorAll('#mode-toggle button').forEach(function (b) {
          b.classList.remove('active')
        })
        btn.classList.add('active')
        draw(btn.dataset.mode || 'absolute')
      })
    })
  }

  if (typeof window !== 'undefined') {
    window.renderHeatmap = renderHeatmap
  }
})()
