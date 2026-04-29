/* SciX V9 — Citation Intent Breakdown.
 *
 * Renders four panels from data/viz/citation_intent.json:
 *   1. Overall intent donut with the 0.27%-coverage caveat.
 *   2. Per-year stacked-area of intent mix.
 *   3. Top-25 method-cited papers as a horizontal bar list.
 *   4. Top-25 background-cited papers as a horizontal bar list.
 *   5. Communities ranked by method-cite ratio.
 *
 * Pure browser code; depends on D3 v7 already loaded by intent.html and on
 * shared.js for the navigation bar.
 */
;(function () {
  'use strict'

  var INTENT_ORDER = ['method', 'background', 'result_comparison']
  var INTENT_LABEL = {
    method: 'Method',
    background: 'Background',
    result_comparison: 'Result Comparison',
  }
  var INTENT_COLOR = {
    method: '#4292c6',
    background: '#f4a582',
    result_comparison: '#91bfdb',
  }

  var DATA_URL_CANDIDATES = [
    './citation_intent.json',
    '/viz/citation_intent.json',
    '/data/viz/citation_intent.json',
  ]

  function fmtInt(n) {
    return Number(n).toLocaleString()
  }

  function fmtPct(p, digits) {
    var d = digits == null ? 1 : digits
    return (Number(p) * 100).toFixed(d) + '%'
  }

  function setStatus(msg) {
    var el = document.getElementById('intent-status')
    if (el) el.textContent = msg
  }

  // ---------------------------------------------------------------------
  // Panel 1: donut of total intent split
  // ---------------------------------------------------------------------
  function renderDonut(totals) {
    var svg = d3.select('#intent-donut')
    svg.selectAll('*').remove()
    var width = 220
    var height = 220
    var radius = 90
    var inner = 55
    var data = INTENT_ORDER.map(function (k) {
      return { intent: k, value: Number(totals[k] || 0) }
    })
    var total = d3.sum(data, function (d) {
      return d.value
    })
    if (total === 0) return

    var pie = d3
      .pie()
      .sort(null)
      .value(function (d) {
        return d.value
      })
    var arc = d3.arc().innerRadius(inner).outerRadius(radius)

    var g = svg.append('g').attr('transform', 'translate(' + width / 2 + ',' + height / 2 + ')')
    g.selectAll('path')
      .data(pie(data))
      .enter()
      .append('path')
      .attr('d', arc)
      .attr('fill', function (d) {
        return INTENT_COLOR[d.data.intent]
      })
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .append('title')
      .text(function (d) {
        return (
          INTENT_LABEL[d.data.intent] +
          ': ' +
          fmtInt(d.data.value) +
          ' (' +
          fmtPct(d.data.value / total) +
          ')'
        )
      })

    g.append('text')
      .attr('text-anchor', 'middle')
      .attr('y', -4)
      .style('font-size', '13px')
      .style('fill', '#444')
      .text(fmtInt(total))
    g.append('text')
      .attr('text-anchor', 'middle')
      .attr('y', 12)
      .style('font-size', '11px')
      .style('fill', '#888')
      .text('classified edges')

    var legend = d3.select('#intent-donut-legend')
    legend.selectAll('*').remove()
    INTENT_ORDER.forEach(function (k) {
      var v = Number(totals[k] || 0)
      var pct = total > 0 ? v / total : 0
      var row = legend.append('div')
      row.append('span').attr('class', 'swatch').style('background', INTENT_COLOR[k])
      row.append('span').text(INTENT_LABEL[k])
      row.append('span').attr('class', 'pct').text(fmtInt(v) + ' · ' + fmtPct(pct))
    })
  }

  // ---------------------------------------------------------------------
  // Panel 2: per-year stacked bars
  //
  // Rendered as discrete bars (one per year) instead of a connected area
  // because the classified slice has gaps (e.g. 2005-2013) — a stacked area
  // would draw a misleading slope through years with zero data.
  // ---------------------------------------------------------------------
  function renderStacked(byYear) {
    var svg = d3.select('#intent-stacked')
    svg.selectAll('*').remove()
    if (!byYear || byYear.length === 0) {
      svg
        .append('text')
        .attr('x', 12)
        .attr('y', 24)
        .style('fill', '#888')
        .text('no per-year data available')
      return
    }
    var node = svg.node()
    var width = (node && node.clientWidth) || 600
    var height = 260
    svg.attr('viewBox', '0 0 ' + width + ' ' + height)
    var margin = { top: 10, right: 16, bottom: 30, left: 56 }
    var iw = width - margin.left - margin.right
    var ih = height - margin.top - margin.bottom

    var g = svg.append('g').attr('transform', 'translate(' + margin.left + ',' + margin.top + ')')

    var stack = d3
      .stack()
      .keys(INTENT_ORDER)
      .order(d3.stackOrderNone)
      .offset(d3.stackOffsetNone)
    var series = stack(byYear)

    var x = d3
      .scaleBand()
      .domain(byYear.map(function (d) {
        return d.year
      }))
      .range([0, iw])
      .padding(0.18)
    var maxTotal = d3.max(byYear, function (d) {
      return (d.method || 0) + (d.background || 0) + (d.result_comparison || 0)
    })
    var y = d3.scaleLinear().domain([0, maxTotal]).nice().range([ih, 0])

    g.selectAll('g.layer')
      .data(series)
      .enter()
      .append('g')
      .attr('class', 'layer')
      .attr('fill', function (d) {
        return INTENT_COLOR[d.key]
      })
      .selectAll('rect')
      .data(function (d) {
        return d.map(function (segment) {
          segment.intent = d.key
          return segment
        })
      })
      .enter()
      .append('rect')
      .attr('x', function (d) {
        return x(d.data.year)
      })
      .attr('y', function (d) {
        return y(d[1])
      })
      .attr('height', function (d) {
        return Math.max(0, y(d[0]) - y(d[1]))
      })
      .attr('width', x.bandwidth())
      .append('title')
      .text(function (d) {
        var v = d[1] - d[0]
        return d.data.year + ' · ' + INTENT_LABEL[d.intent] + ': ' + fmtInt(v)
      })

    var xAxis = d3.axisBottom(x).tickFormat(d3.format('d'))
    var yAxis = d3.axisLeft(y).ticks(5).tickFormat(d3.format('~s'))

    g.append('g')
      .attr('class', 'stacked-axis')
      .attr('transform', 'translate(0,' + ih + ')')
      .call(xAxis)
    g.append('g').attr('class', 'stacked-axis').call(yAxis)
    g.append('text')
      .attr('x', -ih / 2)
      .attr('y', -42)
      .attr('transform', 'rotate(-90)')
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', '#666')
      .text('classified citations')
  }

  // ---------------------------------------------------------------------
  // Panels 3 & 4: top-N papers as a bar list
  // ---------------------------------------------------------------------
  function renderTopPapers(listId, papers, intentKey) {
    var ol = document.getElementById(listId)
    if (!ol) return
    ol.innerHTML = ''
    if (!papers || papers.length === 0) {
      var li = document.createElement('li')
      li.textContent = 'no data'
      li.style.color = '#888'
      ol.appendChild(li)
      return
    }
    var maxN = papers.reduce(function (m, p) {
      return Math.max(m, p.n || 0)
    }, 0)
    papers.forEach(function (p) {
      var li = document.createElement('li')

      var bib = document.createElement('span')
      bib.className = 'bib'
      bib.textContent = p.bibcode
      bib.title = p.bibcode
      li.appendChild(bib)

      var title = document.createElement('span')
      title.className = 'title'
      var titleText = p.title || '(no title)'
      var year = p.year != null ? p.year : ''
      title.appendChild(document.createTextNode(titleText))
      if (year) {
        var yspan = document.createElement('span')
        yspan.className = 'year'
        yspan.textContent = '(' + year + ')'
        title.appendChild(yspan)
      }
      li.appendChild(title)

      var bw = document.createElement('span')
      bw.className = 'barwrap'
      var bar = document.createElement('span')
      bar.className = 'bar'
      bar.style.width = maxN > 0 ? (p.n / maxN) * 100 + '%' : '0%'
      bar.style.background = INTENT_COLOR[intentKey] || '#cfe3f4'
      bar.style.opacity = '0.55'
      bw.appendChild(bar)
      var n = document.createElement('span')
      n.className = 'n'
      n.textContent = fmtInt(p.n)
      bw.appendChild(n)
      li.appendChild(bw)

      ol.appendChild(li)
    })
  }

  // ---------------------------------------------------------------------
  // Panel 5: communities ranked by method-cite ratio
  // ---------------------------------------------------------------------
  function renderCommunities(rows, payload) {
    var ol = document.getElementById('intent-community-list')
    var sub = document.getElementById('intent-community-sub')
    if (!ol) return
    ol.innerHTML = ''
    if (sub && payload) {
      sub.textContent =
        'Resolution: ' +
        (payload.resolution || 'medium') +
        ' · minimum ' +
        fmtInt(payload.min_community_volume || 0) +
        ' classified citations to qualify · top ' +
        rows.length +
        ' shown.'
    }
    if (!rows || rows.length === 0) {
      var li = document.createElement('li')
      li.textContent = 'no qualifying communities'
      li.style.color = '#888'
      ol.appendChild(li)
      return
    }
    rows.forEach(function (r, idx) {
      var li = document.createElement('li')

      var cid = document.createElement('span')
      cid.className = 'cid'
      cid.textContent = '#' + (idx + 1)
      cid.title = 'community_id=' + r.community_id
      li.appendChild(cid)

      var terms = document.createElement('span')
      terms.className = 'terms'
      var label = (r.terms && r.terms.length ? r.terms.slice(0, 6).join(' · ') : '(no labels)')
      terms.textContent = 'c' + r.community_id + ' — ' + label
      li.appendChild(terms)

      var bar = document.createElement('span')
      bar.className = 'ratio-bar'
      var fill = document.createElement('span')
      fill.className = 'fill'
      fill.style.width = (Number(r.method_ratio) * 100).toFixed(1) + '%'
      bar.appendChild(fill)
      bar.title = fmtPct(r.method_ratio, 1) + ' method-ratio'
      li.appendChild(bar)

      var meta = document.createElement('span')
      meta.className = 'meta'
      meta.textContent = fmtPct(r.method_ratio, 1) + ' · n=' + fmtInt(r.total)
      li.appendChild(meta)

      ol.appendChild(li)
    })
  }

  // ---------------------------------------------------------------------
  // Coverage banner
  // ---------------------------------------------------------------------
  function renderCoverage(coverage, totals) {
    var el = document.getElementById('intent-coverage-banner')
    if (!el) return
    var classified = coverage && coverage.classified_edges
    var total = coverage && coverage.total_edges
    var pct = coverage && coverage.pct_classified
    if (classified == null || total == null) {
      el.textContent = 'Coverage data unavailable.'
      return
    }
    el.innerHTML =
      '<strong>Coverage caveat:</strong> ' +
      fmtInt(classified) +
      ' classified edges out of ' +
      fmtInt(total) +
      ' total citation edges (' +
      fmtPct(pct, 2) +
      '). The classified slice currently covers ~7 source-paper years (2001–2014); ' +
      'splits and rankings on this page describe that slice, not the full graph.'
  }

  // ---------------------------------------------------------------------
  // Bootstrap
  // ---------------------------------------------------------------------
  function fetchFirstAvailable(urls) {
    var fa = window.scixViz && window.scixViz.fetchFirstAvailable
    if (typeof fa === 'function') return fa(urls)
    var i = 0
    function next(lastErr) {
      if (i >= urls.length) return Promise.reject(lastErr || new Error('no data url responded'))
      var url = urls[i++]
      return fetch(url, { cache: 'no-store' })
        .then(function (resp) {
          if (!resp.ok) throw new Error(url + ' -> HTTP ' + resp.status)
          return resp.json()
        })
        .catch(function (err) {
          return next(err)
        })
    }
    return next(null)
  }

  function load() {
    setStatus('loading…')
    fetchFirstAvailable(DATA_URL_CANDIDATES)
      .then(function (data) {
        renderCoverage(data.coverage, data.totals)
        renderDonut(data.totals || {})
        renderStacked(Array.isArray(data.by_year) ? data.by_year : [])
        renderTopPapers('intent-top-method', data.top_method || [], 'method')
        renderTopPapers('intent-top-background', data.top_background || [], 'background')
        renderCommunities(data.communities_method || [], data)
        setStatus(
          'loaded · ' +
            (data.top_method || []).length +
            ' method-targets · ' +
            (data.top_background || []).length +
            ' background-targets · ' +
            (data.by_year || []).length +
            ' years · ' +
            (data.communities_method || []).length +
            ' communities',
        )
      })
      .catch(function (err) {
        setStatus('load failed: ' + (err && err.message ? err.message : err))
        var banner = document.getElementById('intent-coverage-banner')
        if (banner) {
          banner.style.background = '#ffe5e5'
          banner.style.borderColor = '#f0a0a0'
          banner.style.color = '#7a0000'
          banner.textContent = 'Failed to load citation_intent.json — run scripts/viz/build_intent_breakdown.py first.'
        }
      })
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', load)
  } else {
    load()
  }
})()
