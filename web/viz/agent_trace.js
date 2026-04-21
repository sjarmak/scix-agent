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
    if (!Array.isArray(bibcodes) || bibcodes.length < 2) return
    if (!positions || typeof deck === 'undefined') return

    var segs = []
    for (var i = 0; i < bibcodes.length - 1; i += 1) {
      var a = positions[bibcodes[i]]
      var b = positions[bibcodes[i + 1]]
      if (!a || !b) continue
      segs.push({ from: a, to: b })
    }
    if (!segs.length) return

    var layerId = 'trace-flash-' + Date.now() + '-' + Math.floor(Math.random() * 1e6)
    var line = new deck.LineLayer({
      id: layerId,
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
    })

    // Merge into existing layers and schedule removal after the fade window.
    var current = (deckInstance.props && deckInstance.props.layers) || []
    deckInstance.setProps({ layers: current.concat([line]) })
    setTimeout(function () {
      var remaining = ((deckInstance.props && deckInstance.props.layers) || []).filter(
        function (layer) {
          return layer && layer.id !== layerId
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
