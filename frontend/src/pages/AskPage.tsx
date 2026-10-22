import { ChatBox } from '../components/AskAI'
import { PageHead, Reveal } from '../components/ui'

export default function AskPage() {
  return (
    <>
      <PageHead eyebrow="Ask SG-ERUP AI" title="Questions about the method, data and results">
        <p>Answers are grounded in the paper text, the project documentation and the measured result files. With an
          <code> ANTHROPIC_API_KEY</code> configured on the server it uses Claude; otherwise it runs in a clearly labelled local
          fallback mode (ranked excerpt search). Keys never reach the browser.</p>
      </PageHead>
      <Reveal>
        <div className="card chat-full" style={{ padding: 0 }}>
          <ChatBox />
        </div>
      </Reveal>
    </>
  )
}
