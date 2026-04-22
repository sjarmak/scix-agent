/*
 * web/viz/ego.js — force-directed citation ego-network view.
 *
 * Exposes `window.renderEgo(bibcode, container, onDone)`:
 *   - Fetches `/viz/api/ego/{bibcode}` (up to 100 refs + 100 cites + 200 2-hop).
 *   - Renders a D3 force simulation into `container` as SVG.
 *   - Community palette is kept in sync with umap_browser.js.
 *   - Clicking a node re-centers on it (calls back into `window.scixEgoNavigate`).
 *
 * Style notes: 2-space indent, single quotes, semicolons, vanilla ES2020. The
 * only runtime dep is d3@7 from CDN — no build step.
 */
;(function () {
  'use strict'

  // 20-colour categorical palette — kept in sync with umap_browser.js so the
  // same community has the same colour across views.
  var PALETTE = [
    [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
    [148, 103, 189], [140, 86, 75], [227, 119, 194], [127, 127, 127],
    [188, 189, 34], [23, 190, 207], [78, 121, 167], [242, 142, 43],
    [225, 87, 89], [118, 183, 178], [89, 161, 79], [237, 201, 72],
    [176, 122, 161], [255, 157, 167], [156, 117, 95], [186, 176, 172],
  ]
  var FALLBACK_COLOR = 'rgb(160,160,160)'

  function _colorForCommunity(cid) {
    if (cid == null || Number.isNaN(Number(cid))) return FALLBACK_COLOR
    var idx = ((Number(cid) % PALETTE.length) + PALETTE.length) % PALETTE.length
    var rgb = PALETTE[idx]
    return 'rgb(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ')'
  }

  function _escape(s) {
    return String(s || '').replace(/[<>&]/g, function (c) {
      return c === '<' ? '&lt;' : c === '>' ? '&gt;' : '&amp;'
    })
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

  function _updatePanel(center, counts) {
    var panel = document.getElementById('ego-panel')
    if (!panel) return
    if (!center) {
      panel.innerHTML =
        '<h2>Center paper</h2><div class="row" style="color:#888">No paper loaded.</div>'
      return
    }
    panel.innerHTML =
      '<h2>Center paper</h2>' +
      '<div class="row"><strong>bibcode:</strong> <code>' +
      _escape(center.bibcode) +
      '</code></div>' +
      '<div class="row"><strong>community:</strong> ' +
      (center.community_id != null ? _escape(center.community_id) : '—') +
      '</div>' +
      '<div class="row" style="margin-top:6px">' +
      _escape(center.title || '(title unavailable)') +
      '</div>' +
      '<hr style="border:0;border-top:1px solid #eee;margin:10px 0" />' +
      '<div class="row"><strong>' +
      counts.direct_refs +
      '</strong> refs · <strong>' +
      counts.direct_cites +
      '</strong> cites · <strong>' +
      counts.second_hop +
      '</strong> 2-hop</div>' +
      '<div class="row" style="color:#888;margin-top:6px">Click a neighbor to re-center.</div>'
  }

  function _updateStats(text) {
    var el = document.getElementById('ego-stats')
    if (el) el.textContent = text
  }

  function _buildGraph(payload) {
    // Merge the three node lists into one array tagged by kind. Use the center
    // bibcode as the shared identity for any duplicate (a ref could also be a
    // cite; we keep the first occurrence's kind).
    var nodes = []
    var byBib = Object.create(null)

    function add(n, kind) {
      if (!n || !n.bibcode) return
      if (byBib[n.bibcode]) return
      var record = {
        id: n.bibcode,
        bibcode: n.bibcode,
        title: n.title || '',
        community_id: n.community_id,
        kind: kind,
        weight: n.weight || 1,
      }
      byBib[n.bibcode] = record
      nodes.push(record)
    }

    if (payload.center) add(payload.center, 'center')
    ;(payload.direct_refs || []).forEach(function (n) {
      add(n, 'ref')
    })
    ;(payload.direct_cites || []).forEach(function (n) {
      add(n, 'cite')
    })
    ;(payload.second_hop_sample || []).forEach(function (n) {
      add(n, 'hop')
    })

    // Group edges by (source, target) pair; weight = number of kinds that
    // connect them. Kind precedence: ref/cite beat hop for colouring so the
    // direct relationship wins.
    var edgeByKey = Object.create(null)
    ;(payload.edges || []).forEach(function (e) {
      if (!e || !e.source || !e.target) return
      if (!byBib[e.source] || !byBib[e.target]) return
      var key = e.source + '>' + e.target
      var existing = edgeByKey[key]
      if (existing) {
        existing.count += 1
        if (e.kind && e.kind !== 'hop') existing.kind = e.kind
      } else {
        edgeByKey[key] = {
          source: e.source,
          target: e.target,
          kind: e.kind || 'hop',
          count: 1,
        }
      }
    })
    var edges = Object.keys(edgeByKey).map(function (k) {
      return edgeByKey[k]
    })

    return { nodes: nodes, edges: edges }
  }

  function _nodeRadius(n) {
    if (n.kind === 'center') return 10
    if (n.kind === 'hop') {
      // log2(weight) dampens the difference between weight-1 and weight-20.
      return 2 + Math.log2(Math.max(1, n.weight))
    }
    return 5
  }

  function _nodeClass(n) {
    if (n.kind === 'center') return 'node-center'
    if (n.kind === 'ref') return 'node-ref'
    if (n.kind === 'cite') return 'node-cite'
    return 'node-hop'
  }

  function _edgeClass(e) {
    if (e.kind === 'ref') return 'edge-ref'
    if (e.kind === 'cite') return 'edge-cite'
    return 'edge-hop'
  }

  function _edgeWidth(e) {
    // Multiple parallel citations (e.g. a node that is both a ref AND a 2-hop
    // target via several paths) thicken the edge.
    return Math.min(4, 0.6 + Math.log2(Math.max(1, e.count)))
  }

  function _render(container, payload) {
    if (typeof d3 === 'undefined' || !d3.forceSimulation) {
      var err = document.createElement('div')
      err.style.padding = '12px'
      err.style.color = '#b00020'
      err.textContent = 'renderEgo: d3 failed to load from CDN.'
      container.appendChild(err)
      return null
    }

    var rect = container.getBoundingClientRect
      ? container.getBoundingClientRect()
      : { width: 1100, height: 600 }
    var width = rect.width > 0 ? rect.width : 1100
    var height = rect.height > 0 ? rect.height : 600

    // Preserve the tooltip element and discard any previous SVG render.
    var tooltipEl = container.querySelector('#ego-tooltip')
    Array.from(container.querySelectorAll('svg')).forEach(function (s) {
      s.remove()
    })

    var graph = _buildGraph(payload)
    if (graph.nodes.length === 0) {
      _updateStats('no nodes to render')
      return null
    }

    var svg = d3
      .select(container)
      .append('svg')
      .attr('viewBox', [0, 0, width, height])
      .attr('preserveAspectRatio', 'xMidYMid meet')

    var g = svg.append('g')

    // Pan/zoom behaviour so users can get close to dense regions.
    svg.call(
      d3.zoom().scaleExtent([0.2, 8]).on('zoom', function (event) {
        g.attr('transform', event.transform)
      }),
    )

    var linkSel = g
      .append('g')
      .attr('stroke-opacity', 0.6)
      .selectAll('line')
      .data(graph.edges)
      .join('line')
      .attr('class', _edgeClass)
      .attr('stroke-width', _edgeWidth)

    var nodeSel = g
      .append('g')
      .selectAll('circle')
      .data(graph.nodes)
      .join('circle')
      .attr('class', _nodeClass)
      .attr('r', _nodeRadius)
      .attr('fill', function (n) {
        return _colorForCommunity(n.community_id)
      })
      .style('cursor', 'pointer')
      .on('mouseenter', function (event, n) {
        var rootRect = container.getBoundingClientRect()
        _showTooltip(
          tooltipEl,
          event.clientX - rootRect.left,
          event.clientY - rootRect.top,
          n.bibcode + (n.title ? ' — ' + n.title : ''),
        )
      })
      .on('mousemove', function (event) {
        var rootRect = container.getBoundingClientRect()
        if (tooltipEl && tooltipEl.style.display !== 'none') {
          tooltipEl.style.left = event.clientX - rootRect.left + 8 + 'px'
          tooltipEl.style.top = event.clientY - rootRect.top + 8 + 'px'
        }
      })
      .on('mouseleave', function () {
        _hideTooltip(tooltipEl)
      })
      .on('click', function (event, n) {
        event.stopPropagation()
        if (n.kind === 'center') return
        if (typeof window.scixEgoNavigate === 'function') {
          window.scixEgoNavigate(n.bibcode)
        }
      })

    nodeSel.append('title').text(function (n) {
      return n.bibcode + (n.title ? '\n' + n.title : '')
    })

    // Drag behaviour: hold a node in place by fixing its position during drag.
    nodeSel.call(
      d3
        .drag()
        .on('start', function (event, n) {
          if (!event.active) sim.alphaTarget(0.3).restart()
          n.fx = n.x
          n.fy = n.y
        })
        .on('drag', function (event, n) {
          n.fx = event.x
          n.fy = event.y
        })
        .on('end', function (event, n) {
          if (!event.active) sim.alphaTarget(0)
          n.fx = null
          n.fy = null
        }),
    )

    // Force simulation. Parameters tuned for ≤500 nodes to settle in ~3s.
    var sim = d3
      .forceSimulation(graph.nodes)
      .force(
        'link',
        d3
          .forceLink(graph.edges)
          .id(function (n) {
            return n.id
          })
          .distance(function (e) {
            return e.kind === 'hop' ? 90 : 55
          })
          .strength(0.5),
      )
      .force('charge', d3.forceManyBody().strength(-120))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force(
        'collide',
        d3.forceCollide().radius(function (n) {
          return _nodeRadius(n) + 2
        }),
      )
      .alphaDecay(0.04) // ~3s to settle from alpha=1
      .on('tick', function () {
        linkSel
          .attr('x1', function (e) {
            return e.source.x
          })
          .attr('y1', function (e) {
            return e.source.y
          })
          .attr('x2', function (e) {
            return e.target.x
          })
          .attr('y2', function (e) {
            return e.target.y
          })
        nodeSel.attr('cx', function (n) {
          return n.x
        }).attr('cy', function (n) {
          return n.y
        })
      })

    // Pin the center to the canvas midpoint so it stays in frame.
    var center = graph.nodes.find(function (n) {
      return n.kind === 'center'
    })
    if (center) {
      center.fx = width / 2
      center.fy = height / 2
    }

    return sim
  }

  function renderEgo(bibcode, container, onDone) {
    _updateStats('loading ' + bibcode + '…')
    _updatePanel(null, { direct_refs: 0, direct_cites: 0, second_hop: 0 })

    var url = '/viz/api/ego/' + encodeURIComponent(bibcode)
    var started = performance.now()
    fetch(url, { cache: 'no-store' })
      .then(function (resp) {
        if (!resp.ok) {
          return resp
            .json()
            .catch(function () {
              return { detail: 'HTTP ' + resp.status }
            })
            .then(function (body) {
              var msg = body && body.detail ? body.detail : 'HTTP ' + resp.status
              throw new Error(msg)
            })
        }
        return resp.json()
      })
      .then(function (payload) {
        var fetchMs = Math.round(performance.now() - started)
        _updatePanel(payload.center, payload.counts || {
          direct_refs: (payload.direct_refs || []).length,
          direct_cites: (payload.direct_cites || []).length,
          second_hop: (payload.second_hop_sample || []).length,
        })
        _render(container, payload)
        _updateStats(
          (payload.counts
            ? payload.counts.direct_refs +
              ' refs · ' +
              payload.counts.direct_cites +
              ' cites · ' +
              payload.counts.second_hop +
              ' 2-hop · ' +
              payload.counts.edges +
              ' edges'
            : 'loaded') +
            ' · ' +
            fetchMs +
            ' ms',
        )
        if (typeof onDone === 'function') onDone(null, payload)
      })
      .catch(function (err) {
        _updateStats('error')
        if (typeof onDone === 'function') onDone(err)
      })
  }

  if (typeof window !== 'undefined') {
    window.renderEgo = renderEgo
  }
})()
