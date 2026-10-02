import { useEffect, useRef } from 'react'
// Plotly is ~4 MB, so it is loaded on demand: the overview and trading pages
// never pay for it, only the charts do.
let plotlyPromise: Promise<any> | null = null
const loadPlotly = () => {
  if (!plotlyPromise) plotlyPromise = import('plotly.js-dist-min')
  return plotlyPromise
}

/**
 * Thin Plotly wrapper.
 *
 * Plotly is used for the equity/drawdown curves and the distribution charts.
 * It is imported client-side only (the bundle is served by the backend), and
 * the chart never receives broker credentials - only series data.
 */
export default function Plot({
  data,
  layout,
  config,
  height = 360,
}: {
  data: any[]
  layout?: any
  config?: any
  height?: number
}) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!ref.current) return
    let plotly: any = null
    let disposed = false
    const merged = {
      height,
      margin: { l: 55, r: 20, t: 24, b: 40 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#e6edf7', size: 12 },
      xaxis: { gridcolor: '#1f2b45', zerolinecolor: '#1f2b45' },
      yaxis: { gridcolor: '#1f2b45', zerolinecolor: '#1f2b45' },
      legend: { orientation: 'h', y: -0.2 },
      ...layout,
    }
    loadPlotly().then((mod) => {
      plotly = mod.default ?? mod
      if (disposed || !ref.current) return
      plotly.react(ref.current, data, merged, {
        displayModeBar: false,
        responsive: true,
        ...config,
      })
    })
    return () => {
      disposed = true
      if (ref.current && plotly) plotly.purge(ref.current)
    }
  }, [data, layout, config, height])

  return <div ref={ref} style={{ width: '100%' }} />
}
