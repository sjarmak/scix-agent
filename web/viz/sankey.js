/*
 * web/viz/sankey.js — interactive temporal community Sankey.
 *
 * Exposes a single global entry point, `window.renderSankey(data, container)`,
 * expected to be called by the bootstrap script in sankey.html once the
 * dataset has been fetched. The dataset schema matches the output of
 * `scripts/viz/build_temporal_sankey_data.py`:
 *
 *   {
 *     nodes: [{ id, decade, community_id, paper_count }, ...],
 *     links: [{ source, target, value }, ...]
 *   }
 *
 * Style notes: 2-space indent, single quotes, semicolons at end of statements,
 * vanilla ES2020, no module system (relies on the d3 and d3-sankey globals
 * loaded via jsdelivr UMD bundles in sankey.html).
 */
;(function () {
  'use strict'

  // Default dimensions used when the container has no measurable width/height
  // (e.g. during early render before layout settles).
  const DEFAULT_WIDTH = 1400
  const DEFAULT_HEIGHT = 820
  const MARGIN = { top: 24, right: 160, bottom: 24, left: 70 }
  // Hide tiny nodes below this share of the max node so labels stay readable
  // on coarse + medium resolutions. Keep it conservative; no clamping for V4 demos.
  const MIN_NODE_SHARE = 0.002

  // Ordinal color scale memoized across renders so repeat community IDs keep
  // a stable color when the view re-renders (e.g. on window resize).
  const _colorScale = (function () {
    // d3-scale may not be present if d3 failed to load; guard so the helper
    // module can still be required in a test harness without d3.
    if (typeof d3 === 'undefined' || typeof d3.scaleOrdinal !== 'function') {
      return function fallbackColor() {
        return '#888'
      }
    }
    const scheme =
      (d3.schemeTableau10 && d3.schemeTableau10.slice()) ||
      (d3.schemeCategory10 && d3.schemeCategory10.slice()) ||
      ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']
    const scale = d3.scaleOrdinal(scheme)
    return function colorFor(key) {
      return scale(String(key))
    }
  })()

  function _clearNode(container) {
    while (container.firstChild) {
      container.removeChild(container.firstChild)
    }
  }

  function _mountError(container, message) {
    _clearNode(container)
    const div = document.createElement('div')
    div.className = 'sankey-error'
    div.style.padding = '12px'
    div.style.color = '#b00020'
    div.textContent = message
    container.appendChild(div)
  }

  function _resolveContainer(container) {
    if (typeof container === 'string') {
      return document.querySelector(container)
    }
    return container
  }

  // Community-label lookup loaded lazily by renderSankey. Maps community_id
  // (number) -> human-readable short label derived from the most frequent
  // title terms in that community (see scripts/viz/compute_community_labels.py).
  let _communityLabels = {}
  // Tri-state: 'idle' | 'loading' | 'loaded' | 'failed'. Prevents the render
  // path from kicking off a second fetch or looping on re-render.
  let _labelsState = 'idle'

  function _communityLabelFor(cid) {
    if (cid == null) return '?'
    const entry = _communityLabels[cid]
    if (entry && entry.terms && entry.terms.length) {
      // Use top-2 terms for a compact label; full list is in the tooltip.
      return entry.terms.slice(0, 2).join(' / ')
    }
    return 'c' + cid
  }

  function _communityTitleFor(cid) {
    if (cid == null) return ''
    const entry = _communityLabels[cid]
    if (entry && entry.terms) {
      return 'c' + cid + ': ' + entry.terms.join(', ')
    }
    return 'c' + cid
  }

  function _formatNodeLabel(node) {
    const decade = node.decade != null ? node.decade : '—'
    const community = _communityLabelFor(node.community_id)
    return decade + ' · ' + community
  }

  function _cloneData(data) {
    // Drop orphan nodes (not referenced by any link) so they don't clutter
    // the columns as label-only vertical stripes. d3-sankey mutates the
    // input, so we clone each object before handing it back.
    const referenced = new Set()
    data.links.forEach(function (l) {
      referenced.add(typeof l.source === 'object' ? l.source.id : l.source)
      referenced.add(typeof l.target === 'object' ? l.target.id : l.target)
    })
    const filtered = data.nodes.filter(function (n) {
      return referenced.has(n.id)
    })
    return {
      nodes: filtered.map(function (n) {
        return Object.assign({}, n)
      }),
      links: data.links.map(function (l) {
        return Object.assign({}, l)
      }),
    }
  }

  function _measure(container) {
    const rect = container.getBoundingClientRect ? container.getBoundingClientRect() : null
    const width = rect && rect.width > 0 ? Math.floor(rect.width) : DEFAULT_WIDTH
    const height = rect && rect.height > 200 ? Math.floor(rect.height) : DEFAULT_HEIGHT
    return { width: width, height: height }
  }

  function _loadCommunityLabels() {
    if (_labelsState === 'loaded' || _labelsState === 'loading') {
      return Promise.resolve()
    }
    _labelsState = 'loading'
    return fetch('/viz/community_labels.json', { cache: 'no-store' })
      .then(function (resp) {
        return resp.ok ? resp.json() : null
      })
      .then(function (payload) {
        if (!payload || !Array.isArray(payload.communities)) {
          _labelsState = 'failed'
          return
        }
        _communityLabels = {}
        payload.communities.forEach(function (c) {
          if (c && c.community_id != null) _communityLabels[c.community_id] = c
        })
        _labelsState = 'loaded'
      })
      .catch(function () {
        _labelsState = 'failed'
      })
  }

  function renderSankey(data, container) {
    const node = _resolveContainer(container)
    if (!node) {
      throw new Error('renderSankey: container not found')
    }
    if (!data || !Array.isArray(data.nodes) || !Array.isArray(data.links)) {
      _mountError(node, 'renderSankey: expected {nodes: [...], links: [...]}')
      return
    }
    if (typeof d3 === 'undefined' || typeof d3.sankey !== 'function') {
      _mountError(node, 'renderSankey: d3 or d3-sankey failed to load from CDN')
      return
    }

    // Labels are optional. If they haven't loaded yet, kick off the fetch
    // and re-render once on arrival; subsequent renders use the cached copy
    // so there's no infinite loop.
    if (_labelsState !== 'loaded' && _labelsState !== 'failed') {
      _loadCommunityLabels().then(function () {
        // Re-render only once labels actually arrived.
        if (_labelsState === 'loaded') renderSankey(data, container)
      })
    }

    _clearNode(node)

    const dims = _measure(node)
    const width = dims.width
    const height = dims.height

    const svg = d3
      .select(node)
      .append('svg')
      .attr('viewBox', '0 0 ' + width + ' ' + height)
      .attr('width', width)
      .attr('height', height)

    const sankeyLayout = d3
      .sankey()
      .nodeId(function (d) {
        return d.id
      })
      .nodeWidth(14)
      .nodePadding(12)
      .extent([
        [MARGIN.left, MARGIN.top],
        [width - MARGIN.right, height - MARGIN.bottom],
      ])

    const layout = sankeyLayout(_cloneData(data))
    const nodes = layout.nodes
    const links = layout.links

    const linkGroup = svg.append('g').attr('class', 'sankey-links').attr('fill', 'none')
    const nodeGroup = svg.append('g').attr('class', 'sankey-nodes')

    const linkSel = linkGroup
      .selectAll('path')
      .data(links)
      .join('path')
      .attr('class', 'sankey-link')
      .attr('d', d3.sankeyLinkHorizontal())
      .attr('stroke', function (d) {
        const sourceCommunity =
          d.source && d.source.community_id != null ? d.source.community_id : 'na'
        return _colorScale(sourceCommunity)
      })
      .attr('stroke-width', function (d) {
        return Math.max(1, d.width || 0)
      })
      .on('click', function (_event, d) {
        const target = d.target || {}
        // eslint-disable-next-line no-console
        console.log('[sankey] link click -> target community', {
          decade: target.decade,
          community_id: target.community_id,
          paper_count: target.paper_count,
          value: d.value,
          node_id: target.id,
        })
      })

    linkSel.append('title').text(function (d) {
      const s = d.source || {}
      const t = d.target || {}
      return (
        _formatNodeLabel(s) +
        '  →  ' +
        _formatNodeLabel(t) +
        '\npapers in flow: ' +
        d.value
      )
    })

    const nodeSel = nodeGroup
      .selectAll('g')
      .data(nodes)
      .join('g')
      .attr('class', 'sankey-node')

    nodeSel
      .append('rect')
      .attr('x', function (d) {
        return d.x0
      })
      .attr('y', function (d) {
        return d.y0
      })
      .attr('height', function (d) {
        return Math.max(1, (d.y1 || 0) - (d.y0 || 0))
      })
      .attr('width', function (d) {
        return Math.max(1, (d.x1 || 0) - (d.x0 || 0))
      })
      .attr('fill', function (d) {
        return _colorScale(d.community_id != null ? d.community_id : 'na')
      })

    // Only label nodes tall enough that the text won't collide with neighbours.
    const LABEL_MIN_HEIGHT = 12
    nodeSel
      .filter(function (d) {
        return (d.y1 || 0) - (d.y0 || 0) >= LABEL_MIN_HEIGHT
      })
      .append('text')
      .attr('x', function (d) {
        return d.x0 < width / 2 ? d.x1 + 6 : d.x0 - 6
      })
      .attr('y', function (d) {
        return ((d.y1 || 0) + (d.y0 || 0)) / 2
      })
      .attr('dy', '0.35em')
      .attr('text-anchor', function (d) {
        return d.x0 < width / 2 ? 'start' : 'end'
      })
      .text(_formatNodeLabel)

    // Decade axis along the bottom so viewers know what the columns mean.
    const decadesByX = new Map()
    nodes.forEach(function (n) {
      if (n.decade == null) return
      const x = (n.x0 + n.x1) / 2
      if (!decadesByX.has(n.decade) || decadesByX.get(n.decade) > x) {
        decadesByX.set(n.decade, x)
      }
    })
    const axisGroup = svg
      .append('g')
      .attr('class', 'sankey-axis')
      .attr('transform', 'translate(0, ' + (height - 6) + ')')
    decadesByX.forEach(function (x, decade) {
      axisGroup
        .append('text')
        .attr('x', x)
        .attr('y', 0)
        .attr('text-anchor', 'middle')
        .attr('font-size', '11px')
        .attr('fill', '#444')
        .text(decade + 's')
    })

    nodeSel.append('title').text(function (d) {
      const papers = d.paper_count != null ? d.paper_count.toLocaleString() : '—'
      const cid = d.community_id != null ? _communityTitleFor(d.community_id) : ''
      return (
        (d.decade != null ? d.decade + 's' : '—') +
        '\n' +
        cid +
        '\npapers: ' +
        papers
      )
    })

    // Hover-to-highlight: dim everything not incident to the hovered node.
    nodeSel
      .on('mouseenter', function (_event, hovered) {
        const incidentLinks = new Set()
        ;(hovered.sourceLinks || []).forEach(function (l) {
          incidentLinks.add(l)
        })
        ;(hovered.targetLinks || []).forEach(function (l) {
          incidentLinks.add(l)
        })
        const incidentNodes = new Set([hovered])
        incidentLinks.forEach(function (l) {
          incidentNodes.add(l.source)
          incidentNodes.add(l.target)
        })
        linkSel.classed('dim', function (l) {
          return !incidentLinks.has(l)
        })
        linkSel.classed('highlight', function (l) {
          return incidentLinks.has(l)
        })
        nodeSel.classed('dim', function (n) {
          return !incidentNodes.has(n)
        })
      })
      .on('mouseleave', function () {
        linkSel.classed('dim', false).classed('highlight', false)
        nodeSel.classed('dim', false)
      })
  }

  // Public export. Assigned to window so tests and the bootstrap script can
  // locate the symbol by name.
  if (typeof window !== 'undefined') {
    window.renderSankey = renderSankey
  }
})()
