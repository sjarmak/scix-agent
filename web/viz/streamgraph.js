/*
 * web/viz/streamgraph.js — papers-per-community-per-year stacked area.
 *
 * Exposes `window.renderStreamgraph(payload, container)`. Payload schema
 * matches `scripts/viz/build_stream_data.py`:
 *
 *   {
 *     resolution: "coarse" | "medium" | "fine",
 *     year_min: 2005,
 *     year_max: 2024,
 *     years: [2005, ..., 2024],
 *     communities: [
 *       { community_id: 0, label: "...", total: 1234, counts: [..] },
 *       ...
 *     ]
 *   }
 *
 * Style notes: 2-space indent, single quotes, semicolons, ES2020. Only
 * runtime dependency is d3@7 from CDN. Color comes from the shared
 * resolution-aware palette in shared.js.
 */
;(function () {
  'use strict'

  const FALLBACK_COLOR = 'rgb(160,160,160)'
  // Show this many bands by name in the side panel; the rest stack but
  // aren't enumerated unless hovered.
  const PANEL_TOP_N = 12

  function _colorForCommunity(cid) {
    var scx = (typeof window !== 'undefined' && window.scixViz) || null
    if (scx && typeof scx.colorForCommunity === 'function') {
      var rgb = scx.colorForCommunity(cid)
      return 'rgb(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ')'
    }
    return FALLBACK_COLOR
  }

  function _resolveContainer(container) {
    if (typeof container === 'string') return document.querySelector(container)
    return container
  }

  function _showTooltip(tooltipEl, x, y, html) {
    if (!tooltipEl) return
    tooltipEl.style.left = x + 12 + 'px'
    tooltipEl.style.top = y + 12 + 'px'
    tooltipEl.innerHTML = html
    tooltipEl.style.display = 'block'
  }

  function _hideTooltip(tooltipEl) {
    if (tooltipEl) tooltipEl.style.display = 'none'
  }

  function _formatTooltipBody(community, year, count) {
    var safeLabel = String(community.label || '').replace(/[<>&]/g, function (c) {
      return c === '<' ? '&lt;' : c === '>' ? '&gt;' : '&amp;'
    })
    return (
      '<div style="font-weight:600;margin-bottom:4px">' + safeLabel + '</div>' +
      '<div>community ' + community.community_id + ' · ' + year + '</div>' +
      '<div>' + count.toLocaleString() + ' papers</div>'
    )
  }

  function _updatePanel(communities, hovered) {
    var panel = document.getElementById('stream-panel')
    if (!panel) return
    if (hovered) {
      var safeLabel = String(hovered.label || '').replace(/[<>]/g, '')
      var totalsRow =
        '<div><strong>community:</strong> ' + hovered.community_id + '</div>' +
        '<div><strong>label:</strong> ' + safeLabel + '</div>' +
        '<div><strong>total (window):</strong> ' + hovered.total.toLocaleString() + ' papers</div>'
      panel.innerHTML = '<h2>Selected community</h2>' + totalsRow
      return
    }
    // Default panel: top-N by total in the window.
    var rows = communities.slice(0, PANEL_TOP_N).map(function (c) {
      var rgb = _colorForCommunity(c.community_id)
      var safeLabel = String(c.label || '').replace(/[<>]/g, '')
      return (
        '<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">' +
        '<span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:' +
        rgb + ';flex-shrink:0"></span>' +
        '<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' +
        safeLabel + '">' + safeLabel + '</span>' +
        '<span style="color:#888;font-size:12px">' + c.total.toLocaleString() + '</span>' +
        '</div>'
      )
    }).join('')
    panel.innerHTML =
      '<h2>Top ' + Math.min(PANEL_TOP_N, communities.length) + ' communities</h2>' +
      '<div style="font-size:12px;color:#888;margin-bottom:8px">By total papers in the visible window. Hover a band for details.</div>' +
      rows
  }

  function _updateStats(payload) {
    var el = document.getElementById('stream-stats')
    if (!el) return
    var totalPapers = payload.communities.reduce(function (s, c) { return s + c.total }, 0)
    el.textContent =
      totalPapers.toLocaleString() + ' papers · ' +
      payload.communities.length.toLocaleString() + ' communities · ' +
      payload.year_min + '–' + payload.year_max
  }

  function _toRows(payload) {
    // d3.stack expects an array of row objects keyed by series name. Build
    // [{year: 2005, "0": 100, "1": 200, ...}, ...].
    return payload.years.map(function (year, yearIdx) {
      var row = { year: year }
      payload.communities.forEach(function (c) {
        row[String(c.community_id)] = c.counts[yearIdx] || 0
      })
      return row
    })
  }

  function renderStreamgraph(payload, container) {
    var node = _resolveContainer(container)
    if (!node) throw new Error('renderStreamgraph: container not found')
    if (!payload || !Array.isArray(payload.years) || !Array.isArray(payload.communities)) {
      throw new Error('renderStreamgraph: payload missing years/communities')
    }
    if (typeof d3 === 'undefined') {
      throw new Error('renderStreamgraph: d3@7 failed to load from CDN')
    }

    var rect = node.getBoundingClientRect ? node.getBoundingClientRect() : null
    var width = rect && rect.width > 0 ? rect.width : 1100
    var height = rect && rect.height > 0 ? rect.height : 600
    var margin = { top: 20, right: 20, bottom: 36, left: 50 }
    var innerW = Math.max(100, width - margin.left - margin.right)
    var innerH = Math.max(100, height - margin.top - margin.bottom)

    var rows = _toRows(payload)
    var keys = payload.communities.map(function (c) { return String(c.community_id) })
    var commById = {}
    payload.communities.forEach(function (c) { commById[String(c.community_id)] = c })

    var stack = d3.stack()
      .keys(keys)
      .order(d3.stackOrderInsideOut)
      .offset(d3.stackOffsetWiggle)
    var series = stack(rows)

    var x = d3.scaleLinear()
      .domain([payload.year_min, payload.year_max])
      .range([0, innerW])

    var minY = d3.min(series, function (s) { return d3.min(s, function (d) { return d[0] }) })
    var maxY = d3.max(series, function (s) { return d3.max(s, function (d) { return d[1] }) })
    var y = d3.scaleLinear()
      .domain([minY, maxY])
      .range([innerH, 0])
      .nice()

    var area = d3.area()
      .x(function (d) { return x(d.data.year) })
      .y0(function (d) { return y(d[0]) })
      .y1(function (d) { return y(d[1]) })
      .curve(d3.curveCatmullRom.alpha(0.5))

    var svg = d3.select(node).append('svg')
      .attr('viewBox', '0 0 ' + width + ' ' + height)
      .attr('preserveAspectRatio', 'xMidYMid meet')

    var g = svg.append('g')
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')')

    var tooltip = node.querySelector('#stream-tooltip')

    // Click-to-isolate: clicking a band dims everything else.
    var isolated = null

    var bands = g.selectAll('path.stream-band')
      .data(series, function (s) { return s.key })
      .enter()
      .append('path')
      .attr('class', 'stream-band')
      .attr('d', area)
      .attr('fill', function (s) { return _colorForCommunity(Number(s.key)) })
      .on('mousemove', function (evt, s) {
        var community = commById[s.key]
        if (!community) return
        // Map mouse x to the closest year so the tooltip shows that year's count.
        var rectN = node.getBoundingClientRect()
        var xPx = evt.clientX - rectN.left - margin.left
        var year = Math.round(x.invert(Math.max(0, Math.min(innerW, xPx))))
        var yearIdx = year - payload.year_min
        var count = community.counts[yearIdx] || 0
        _showTooltip(
          tooltip,
          evt.clientX - rectN.left,
          evt.clientY - rectN.top,
          _formatTooltipBody(community, year, count),
        )
        _updatePanel(payload.communities, community)
      })
      .on('mouseleave', function () {
        _hideTooltip(tooltip)
        _updatePanel(payload.communities, isolated ? commById[isolated] : null)
      })
      .on('click', function (evt, s) {
        var key = s.key
        isolated = isolated === key ? null : key
        bands.classed('dim', function (d) {
          return isolated != null && d.key !== isolated
        })
      })

    // X axis — integer years.
    var xAxis = d3.axisBottom(x)
      .ticks(Math.min(payload.years.length, 10))
      .tickFormat(d3.format('d'))
    g.append('g')
      .attr('class', 'stream-axis')
      .attr('transform', 'translate(0,' + innerH + ')')
      .call(xAxis)

    _updatePanel(payload.communities, null)
    _updateStats(payload)

    return {
      finalize: function () {
        // Hook for future tear-down (selections, listeners). Currently the
        // SVG is fully removed by the bootstrap before re-render.
      },
    }
  }

  if (typeof window !== 'undefined') {
    window.renderStreamgraph = renderStreamgraph
  }
})()
