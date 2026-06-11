/* V12 Provenance — entity source provenance + precision profile.
 *
 * Renders three D3 charts from data/viz/provenance.json:
 *   1. Source treemap: entities by (source -> entity_type), area = entity count.
 *   2. Precision profile: one stacked horizontal bar per source, segments = share
 *      of the source's links in each precision band (high/medium/low/noisy).
 *      Mirrors scix.extract.ner_quality_profile.precision_band.
 *   3. Ambiguity classes: bars of the classified slice (domain_safe / banned /
 *      unique / homograph); the unclassified NULL majority is shown in the caption.
 *
 * Exposes window.renderProvenance(payload, { treemapRoot, precisionRoot, ambiguityRoot }).
 * Pure render — the page bootstrap (in the HTML) handles fetching.
 */
;(function () {
  'use strict'
  if (typeof window === 'undefined') return

  // Precision bands, ordered best -> worst, with a fixed colour each.
  var BAND_ORDER = ['high', 'medium', 'low', 'noisy']
  var BAND_COLOR = {
    high: '#1a9850',
    medium: '#4e79a7',
    low: '#e0a32e',
    noisy: '#d6604d',
  }
  // Stable per-source colours for the treemap (gliner first / dominant).
  var SOURCE_COLOR = [
    '#0b60b0', '#4e79a7', '#6a9f58', '#b07aa1',
    '#e0843e', '#9c755f', '#d7a0a0', '#8c8c8c',
  ]

  function fmtInt(n) {
    return Number(n).toLocaleString('en-US')
  }
  function fmtCompact(n) {
    if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M'
    if (n >= 1e3) return (n / 1e3).toFixed(0) + 'K'
    return String(n)
  }
  function pct(x) {
    return (x * 100).toFixed(1) + '%'
  }

  // --- Chart 1: source x type treemap -----------------------------------
  function renderTreemap(payload, root) {
    var d3 = window.d3
    var sources = payload.by_source || []
    var width = root.clientWidth || 720
    var height = 420

    var colorFor = {}
    sources.forEach(function (s, i) {
      colorFor[s.source] = SOURCE_COLOR[i % SOURCE_COLOR.length]
    })

    var rootNode = {
      name: 'root',
      children: sources.map(function (s) {
        return {
          name: s.source,
          children: (s.entity_types || []).map(function (t) {
            return { name: t.entity_type, value: t.n, source: s.source }
          }),
        }
      }),
    }

    var hierarchy = d3
      .hierarchy(rootNode)
      .sum(function (d) { return d.value || 0 })
      .sort(function (a, b) { return b.value - a.value })

    d3.treemap().size([width, height]).paddingInner(1).paddingTop(0).round(true)(hierarchy)

    var svg = d3
      .select(root)
      .append('svg')
      .attr('width', '100%')
      .attr('viewBox', '0 0 ' + width + ' ' + height)
      .attr('role', 'img')
      .attr('aria-label', 'Entity counts by source and type')

    var leaf = svg
      .selectAll('g.leaf')
      .data(hierarchy.leaves())
      .enter()
      .append('g')
      .attr('class', 'leaf')
      .attr('transform', function (d) { return 'translate(' + d.x0 + ',' + d.y0 + ')' })

    leaf
      .append('rect')
      .attr('width', function (d) { return Math.max(0, d.x1 - d.x0) })
      .attr('height', function (d) { return Math.max(0, d.y1 - d.y0) })
      .attr('fill', function (d) { return colorFor[d.data.source] })
      .attr('opacity', 0.88)
      .append('title')
      .text(function (d) {
        return d.data.source + ' · ' + d.data.name + ': ' + fmtInt(d.value)
      })

    // Label only tiles big enough to hold text.
    leaf
      .filter(function (d) { return d.x1 - d.x0 > 64 && d.y1 - d.y0 > 26 })
      .append('text')
      .attr('x', 4)
      .attr('y', 15)
      .attr('fill', '#fff')
      .attr('font-size', 12)
      .text(function (d) { return d.data.name })

    leaf
      .filter(function (d) { return d.x1 - d.x0 > 64 && d.y1 - d.y0 > 40 })
      .append('text')
      .attr('x', 4)
      .attr('y', 30)
      .attr('fill', '#eef')
      .attr('font-size', 11)
      .text(function (d) { return fmtCompact(d.value) })
  }

  // --- Chart 2: precision band profile per source -----------------------
  function renderPrecision(payload, root) {
    var d3 = window.d3
    var sources = (payload.by_source || []).filter(function (s) {
      return s.precision_profile
    })
    var width = root.clientWidth || 720
    var rowH = 38
    var margin = { top: 8, right: 16, bottom: 8, left: 110 }
    var height = margin.top + margin.bottom + sources.length * rowH

    var svg = d3
      .select(root)
      .append('svg')
      .attr('width', '100%')
      .attr('viewBox', '0 0 ' + width + ' ' + height)
      .attr('role', 'img')
      .attr('aria-label', 'Precision band distribution per source')

    var x = d3.scaleLinear().domain([0, 1]).range([margin.left, width - margin.right])
    var y = d3
      .scaleBand()
      .domain(sources.map(function (s) { return s.source }))
      .range([margin.top, height - margin.bottom])
      .padding(0.3)

    var g = svg.append('g')

    sources.forEach(function (s) {
      var cursor = 0
      var yy = y(s.source)
      BAND_ORDER.forEach(function (band) {
        var frac = s.precision_profile[band] || 0
        if (frac <= 0) return
        g.append('rect')
          .attr('x', x(cursor))
          .attr('y', yy)
          .attr('width', x(cursor + frac) - x(cursor))
          .attr('height', y.bandwidth())
          .attr('fill', BAND_COLOR[band])
          .append('title')
          .text(s.source + ' · ' + band + ': ' + pct(frac))
        cursor += frac
      })

      // Source label (left gutter), with sampled/constant marker.
      g.append('text')
        .attr('x', margin.left - 10)
        .attr('y', yy + y.bandwidth() / 2)
        .attr('text-anchor', 'end')
        .attr('dominant-baseline', 'central')
        .attr('fill', '#333')
        .attr('font-size', 12)
        .text(s.source)

      g.append('text')
        .attr('x', margin.left - 10)
        .attr('y', yy + y.bandwidth() / 2 + 12)
        .attr('text-anchor', 'end')
        .attr('dominant-baseline', 'central')
        .attr('fill', '#999')
        .attr('font-size', 9)
        .text(s.precision_basis === 'sampled' ? 'sampled' : 'lexical')
    })
  }

  // --- Chart 3: ambiguity classes (classified slice) --------------------
  function renderAmbiguity(payload, root) {
    var d3 = window.d3
    var rows = (payload.ambiguity || []).filter(function (a) {
      return a.ambiguity_class !== null && a.ambiguity_class !== undefined
    })
    rows.sort(function (a, b) { return b.n - a.n })
    var width = root.clientWidth || 720
    var rowH = 40
    var margin = { top: 8, right: 70, bottom: 8, left: 120 }
    var height = margin.top + margin.bottom + Math.max(rows.length, 1) * rowH

    var svg = d3
      .select(root)
      .append('svg')
      .attr('width', '100%')
      .attr('viewBox', '0 0 ' + width + ' ' + height)
      .attr('role', 'img')
      .attr('aria-label', 'Ambiguity class counts (classified entities)')

    var maxN = d3.max(rows, function (d) { return d.n }) || 1
    var x = d3.scaleLinear().domain([0, maxN]).range([margin.left, width - margin.right])
    var y = d3
      .scaleBand()
      .domain(rows.map(function (d) { return d.ambiguity_class }))
      .range([margin.top, height - margin.bottom])
      .padding(0.3)

    var g = svg.append('g')

    g.selectAll('rect')
      .data(rows)
      .enter()
      .append('rect')
      .attr('x', x(0))
      .attr('y', function (d) { return y(d.ambiguity_class) })
      .attr('width', function (d) { return x(d.n) - x(0) })
      .attr('height', y.bandwidth())
      .attr('rx', 3)
      .attr('fill', '#b07aa1')
      .append('title')
      .text(function (d) { return d.ambiguity_class + ': ' + fmtInt(d.n) })

    g.selectAll('text.label')
      .data(rows)
      .enter()
      .append('text')
      .attr('class', 'label')
      .attr('x', margin.left - 10)
      .attr('y', function (d) { return y(d.ambiguity_class) + y.bandwidth() / 2 })
      .attr('text-anchor', 'end')
      .attr('dominant-baseline', 'central')
      .attr('fill', '#333')
      .attr('font-size', 12)
      .text(function (d) { return d.ambiguity_class })

    g.selectAll('text.value')
      .data(rows)
      .enter()
      .append('text')
      .attr('class', 'value')
      .attr('x', function (d) { return x(d.n) + 8 })
      .attr('y', function (d) { return y(d.ambiguity_class) + y.bandwidth() / 2 })
      .attr('dominant-baseline', 'central')
      .attr('fill', '#555')
      .attr('font-size', 11)
      .text(function (d) { return fmtCompact(d.n) })
  }

  function renderProvenance(payload, opts) {
    opts = opts || {}
    if (opts.treemapRoot) {
      while (opts.treemapRoot.firstChild) opts.treemapRoot.removeChild(opts.treemapRoot.firstChild)
      renderTreemap(payload, opts.treemapRoot)
    }
    if (opts.precisionRoot) {
      while (opts.precisionRoot.firstChild) opts.precisionRoot.removeChild(opts.precisionRoot.firstChild)
      renderPrecision(payload, opts.precisionRoot)
    }
    if (opts.ambiguityRoot) {
      while (opts.ambiguityRoot.firstChild) opts.ambiguityRoot.removeChild(opts.ambiguityRoot.firstChild)
      renderAmbiguity(payload, opts.ambiguityRoot)
    }
    return { payload: payload }
  }

  window.renderProvenance = renderProvenance
})()
