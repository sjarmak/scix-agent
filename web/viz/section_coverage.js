/* V11 Sections — full-text section coverage.
 *
 * Renders two D3 charts from data/viz/section_coverage.json:
 *   1. Role presence: horizontal bars, % of full-text papers carrying each
 *      canonical section role (background/method/result/conclusion/other).
 *   2. Decade coverage: stacked bars of full-text vs metadata-only papers per
 *      decade, with the coverage % printed on top of each bar.
 *
 * Exposes window.renderSectionCoverage(payload, { roleRoot, decadeRoot }).
 * Pure render — the page bootstrap (in the HTML) handles fetching.
 */
;(function () {
  'use strict'
  if (typeof window === 'undefined') return

  var FULLTEXT_COLOR = '#0b60b0'
  var METADATA_COLOR = '#d7dee6'
  var ROLE_COLOR = '#4e79a7'

  function pct(x) {
    return (x * 100).toFixed(1) + '%'
  }

  function fmtInt(n) {
    return Number(n).toLocaleString('en-US')
  }

  // --- Chart 1: role presence (horizontal bars) --------------------------
  function renderRoles(payload, root) {
    var d3 = window.d3
    var roles = payload.roles || []
    var width = root.clientWidth || 640
    var rowH = 46
    var margin = { top: 8, right: 70, bottom: 8, left: 170 }
    var height = margin.top + margin.bottom + roles.length * rowH

    var svg = d3
      .select(root)
      .append('svg')
      .attr('width', '100%')
      .attr('viewBox', '0 0 ' + width + ' ' + height)
      .attr('role', 'img')
      .attr('aria-label', 'Section role presence across full-text papers')

    var x = d3
      .scaleLinear()
      .domain([0, 1])
      .range([margin.left, width - margin.right])
    var y = d3
      .scaleBand()
      .domain(roles.map(function (r) { return r.role }))
      .range([margin.top, height - margin.bottom])
      .padding(0.28)

    var g = svg.append('g')

    // Track backgrounds (full-width, light) so partial bars read as fractions.
    g.selectAll('rect.track')
      .data(roles)
      .enter()
      .append('rect')
      .attr('class', 'track')
      .attr('x', x(0))
      .attr('y', function (r) { return y(r.role) })
      .attr('width', x(1) - x(0))
      .attr('height', y.bandwidth())
      .attr('rx', 4)
      .attr('fill', '#f0f3f7')

    g.selectAll('rect.bar')
      .data(roles)
      .enter()
      .append('rect')
      .attr('class', 'bar')
      .attr('x', x(0))
      .attr('y', function (r) { return y(r.role) })
      .attr('width', function (r) { return x(r.present_pct) - x(0) })
      .attr('height', y.bandwidth())
      .attr('rx', 4)
      .attr('fill', ROLE_COLOR)

    // Role labels (left gutter).
    g.selectAll('text.label')
      .data(roles)
      .enter()
      .append('text')
      .attr('class', 'label')
      .attr('x', margin.left - 10)
      .attr('y', function (r) { return y(r.role) + y.bandwidth() / 2 })
      .attr('text-anchor', 'end')
      .attr('dominant-baseline', 'central')
      .attr('fill', '#333')
      .attr('font-size', 13)
      .text(function (r) { return r.label })

    // Value labels (right of each track).
    g.selectAll('text.value')
      .data(roles)
      .enter()
      .append('text')
      .attr('class', 'value')
      .attr('x', x(1) + 8)
      .attr('y', function (r) { return y(r.role) + y.bandwidth() / 2 })
      .attr('dominant-baseline', 'central')
      .attr('fill', '#555')
      .attr('font-size', 12)
      .text(function (r) { return pct(r.present_pct) })
  }

  // --- Chart 2: decade coverage (stacked bars) ---------------------------
  function renderDecades(payload, root) {
    var d3 = window.d3
    var decades = payload.decades || []
    var width = root.clientWidth || 640
    var height = 360
    var margin = { top: 24, right: 16, bottom: 40, left: 64 }

    var svg = d3
      .select(root)
      .append('svg')
      .attr('width', '100%')
      .attr('viewBox', '0 0 ' + width + ' ' + height)
      .attr('role', 'img')
      .attr('aria-label', 'Full-text vs metadata-only papers per decade')

    var x = d3
      .scaleBand()
      .domain(decades.map(function (d) { return d.decade }))
      .range([margin.left, width - margin.right])
      .padding(0.2)
    var yMax = d3.max(decades, function (d) { return d.total }) || 1
    var y = d3
      .scaleLinear()
      .domain([0, yMax])
      .nice()
      .range([height - margin.bottom, margin.top])

    var g = svg.append('g')

    // Metadata-only segment (full bar height = total).
    g.selectAll('rect.total')
      .data(decades)
      .enter()
      .append('rect')
      .attr('class', 'total')
      .attr('x', function (d) { return x(d.decade) })
      .attr('y', function (d) { return y(d.total) })
      .attr('width', x.bandwidth())
      .attr('height', function (d) { return y(0) - y(d.total) })
      .attr('fill', METADATA_COLOR)

    // Full-text segment (overlaid from baseline up to fulltext count).
    g.selectAll('rect.fulltext')
      .data(decades)
      .enter()
      .append('rect')
      .attr('class', 'fulltext')
      .attr('x', function (d) { return x(d.decade) })
      .attr('y', function (d) { return y(d.fulltext) })
      .attr('width', x.bandwidth())
      .attr('height', function (d) { return y(0) - y(d.fulltext) })
      .attr('fill', FULLTEXT_COLOR)
      .append('title')
      .text(function (d) {
        return (
          d.decade +
          's: ' +
          fmtInt(d.fulltext) +
          ' full-text / ' +
          fmtInt(d.total) +
          ' total (' +
          pct(d.fulltext_pct) +
          ')'
        )
      })

    // Coverage % atop each bar.
    g.selectAll('text.cov')
      .data(decades)
      .enter()
      .append('text')
      .attr('class', 'cov')
      .attr('x', function (d) { return x(d.decade) + x.bandwidth() / 2 })
      .attr('y', function (d) { return y(d.total) - 6 })
      .attr('text-anchor', 'middle')
      .attr('fill', '#333')
      .attr('font-size', 11)
      .text(function (d) { return pct(d.fulltext_pct) })

    // X axis (decade labels).
    var xAxis = d3.axisBottom(x).tickFormat(function (d) { return d + 's' })
    g.append('g')
      .attr('class', 'cov-axis')
      .attr('transform', 'translate(0,' + (height - margin.bottom) + ')')
      .call(xAxis)

    // Y axis (paper counts, abbreviated).
    var yAxis = d3.axisLeft(y).ticks(5).tickFormat(d3.format('.2s'))
    g.append('g')
      .attr('class', 'cov-axis')
      .attr('transform', 'translate(' + margin.left + ',0)')
      .call(yAxis)
  }

  function renderSectionCoverage(payload, opts) {
    opts = opts || {}
    if (opts.roleRoot) {
      while (opts.roleRoot.firstChild) opts.roleRoot.removeChild(opts.roleRoot.firstChild)
      renderRoles(payload, opts.roleRoot)
    }
    if (opts.decadeRoot) {
      while (opts.decadeRoot.firstChild) opts.decadeRoot.removeChild(opts.decadeRoot.firstChild)
      renderDecades(payload, opts.decadeRoot)
    }
    return { payload: payload }
  }

  window.renderSectionCoverage = renderSectionCoverage
})()
