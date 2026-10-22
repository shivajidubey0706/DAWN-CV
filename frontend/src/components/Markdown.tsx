import katex from 'katex'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

export function Markdown({ text }: { text: string }) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]}
      components={{ a: ({ href, children }) => <a href={href} target="_blank" rel="noreferrer">{children}</a> }}>
      {text}
    </ReactMarkdown>
  )
}

export function Equation({ tex, display = true }: { tex: string; display?: boolean }) {
  let html: string
  try {
    html = katex.renderToString(tex, { displayMode: display, throwOnError: false, output: 'html' })
  } catch {
    html = tex
  }
  return <div className="eq" dangerouslySetInnerHTML={{ __html: html }} />
}
