/*
 * web/viz/agent_trace.js — agent-trace SSE overlay for the UMAP browser.
 *
 * Exposes two globals:
 *
 *   window.subscribeTraceStream(url, onEvent)
 *     Wraps ``EventSource``. Parses each ``message`` event's ``data`` field
 *     as JSON and invokes ``onEvent(payload)``. Malformed JSON is logged to
 *     the console and swallowed so the stream keeps running. Returns the
 *     underlying EventSource so callers can ``.close()`` it.
 *
 *   window.flashTraceSegments(deckInstance, bibcodes, positions)
 *     Injects a transient deck.gl LineLayer connecting the points for the
 *     given bibcodes. Removes the layer after ~1.5 s (the 1-2 s fade called
 *     for in the spec). If ``deckInstance`` is missing, ``bibcodes`` is
 *     shorter than two, or ``positions`` is empty, it is a no-op.
 *
 * Style: 2-space indent, single quotes, semicolons at end of statements,
 * vanilla ES2020, no module system. Matches the conventions in
 * umap_browser.js.
 */
;(function () {
  'use strict'

  function subscribeTraceStream(url, onEvent) {
    if (typeof EventSource === 'undefined') {
      // eslint-disable-next-line no-console
      console.error('subscribeTraceStream: EventSource is not available')
      return null
    }
    var es = new EventSource(url)
    es.addEventListener('message', function (ev) {
      try {
        var payload = JSON.parse(ev.data)
        if (typeof onEvent === 'function') {
          onEvent(payload)
        }
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error('subscribeTraceStream: bad JSON payload', err)
      }
    })
    es.addEventListener('error', function (err) {
      // eslint-disable-next-line no-console
      console.warn('subscribeTraceStream: EventSource error', err)
    })
    return es
  }

  function flashTraceSegments(deckInstance, bibcodes, positions) {
    if (!deckInstance || typeof deckInstance.setProps !== 'function') return
    if (!Array.isArray(bibcodes) || !positions || typeof deck === 'undefined') return

    var hits = []
    for (var i = 0; i < bibcodes.length; i += 1) {
      var pos = positions[bibcodes[i]]
      if (pos) hits.push({ bibcode: bibcodes[i], pos: pos })
    }
    if (!hits.length) return

    var stamp = Date.now() + '-' + Math.floor(Math.random() * 1e6)
    var newLayers = []

    // Highlight every touched point, even for single-bibcode events.
    newLayers.push(
      new deck.ScatterplotLayer({
        id: 'trace-hit-' + stamp,
        data: hits,
        getPosition: function (d) {
          return [Number(d.pos[0]) || 0, Number(d.pos[1]) || 0]
        },
        getRadius: 7,
        radiusUnits: 'pixels',
        getFillColor: [255, 64, 64, 230],
        stroked: true,
        getLineColor: [255, 255, 255, 255],
        lineWidthMinPixels: 1,
      }),
    )

    // Connect consecutive hits when there are >= 2.
    if (hits.length >= 2) {
      var segs = []
      for (var j = 0; j < hits.length - 1; j += 1) {
        segs.push({ from: hits[j].pos, to: hits[j + 1].pos })
      }
      newLayers.push(
        new deck.LineLayer({
          id: 'trace-link-' + stamp,
          data: segs,
          getSourcePosition: function (d) {
            return [Number(d.from[0]) || 0, Number(d.from[1]) || 0]
          },
          getTargetPosition: function (d) {
            return [Number(d.to[0]) || 0, Number(d.to[1]) || 0]
          },
          getColor: [255, 64, 64, 200],
          getWidth: 2,
          widthUnits: 'pixels',
        }),
      )
    }

    var current = (deckInstance.props && deckInstance.props.layers) || []
    deckInstance.setProps({ layers: current.concat(newLayers) })
    var addedIds = newLayers.map(function (l) {
      return l.id
    })
    setTimeout(function () {
      var remaining = ((deckInstance.props && deckInstance.props.layers) || []).filter(
        function (layer) {
          return layer && addedIds.indexOf(layer.id) === -1
        },
      )
      deckInstance.setProps({ layers: remaining })
    }, 1500)
  }

  // Public exports. Assigned to window so the bootstrap script in
  // agent_trace.html and tests can locate the symbols by name.
  if (typeof window !== 'undefined') {
    window.subscribeTraceStream = subscribeTraceStream
    window.flashTraceSegments = flashTraceSegments
  }
})()
