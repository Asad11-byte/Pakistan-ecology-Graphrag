import { useState } from 'react'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'

const API_URL = import.meta.env.VITE_API_URL || '/api'

function App() {
  const [status, setStatus] = useState('')
  const [ingesting, setIngesting] = useState(false)
  const [comparison, setComparison] = useState(null)

  const [schemaMode, setSchemaMode] = useState('predefined')
  const [retrievalMode, setRetrievalMode] = useState('traversal')
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState([])
  const [asking, setAsking] = useState(false)

  const runIngest = async () => {
    setIngesting(true)
    setStatus('Ingesting dataset with both schema strategies... this can take a minute.')
    try {
      const res = await axios.post(`${API_URL}/ingest/all`)
      setStatus(`Ingestion complete.\nPredefined: ${res.data.predefined.node_writes} nodes, ${res.data.predefined.relationship_writes} relationships.\nLLM-inferred: ${res.data.llm_inferred.node_writes} nodes, ${res.data.llm_inferred.relationship_writes} relationships.`)
      await rebuildCommunities()
    } catch (err) {
      setStatus(`Ingestion failed: ${err.response?.data?.detail || err.message}`)
    } finally {
      setIngesting(false)
    }
  }

  const rebuildCommunities = async () => {
    try {
      await axios.post(`${API_URL}/query/community/rebuild?schema_mode=predefined`)
    } catch (err) {
      console.error('Community rebuild failed', err)
    }
  }

  const fetchComparison = async () => {
    try {
      const res = await axios.get(`${API_URL}/ingest/compare`)
      setComparison(res.data)
    } catch (err) {
      setStatus(`Comparison failed: ${err.response?.data?.detail || err.message}`)
    }
  }

  const resetGraph = async () => {
    if (!confirm('This deletes all nodes and relationships in Neo4j. Continue?')) return
    try {
      await axios.delete(`${API_URL}/ingest/reset`)
      setStatus('Graph cleared.')
      setComparison(null)
    } catch (err) {
      setStatus(`Reset failed: ${err.response?.data?.detail || err.message}`)
    }
  }

  const askQuestion = async () => {
    if (!question.trim()) return
    const userMsg = { role: 'user', text: question }
    setMessages(prev => [...prev, userMsg])
    setAsking(true)
    const q = question
    setQuestion('')

    try {
      const endpoint = retrievalMode === 'traversal' ? '/query/traversal' : '/query/community'
      const res = await axios.post(`${API_URL}${endpoint}`, {
        question: q,
        schema_mode: schemaMode,
      })
      const assistantMsg = {
        role: 'assistant',
        text: res.data.answer,
        meta: retrievalMode === 'traversal'
          ? `anchor: ${res.data.anchor || 'n/a'} · schema: ${schemaMode}`
          : `communities used: ${(res.data.communities_used || []).join(', ') || 'n/a'} · schema: ${schemaMode}`,
      }
      setMessages(prev => [...prev, assistantMsg])
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        text: `Error: ${err.response?.data?.detail || err.message}`,
      }])
    } finally {
      setAsking(false)
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Pakistan Ecology Graph RAG</h1>
        <p>LangChain + Neo4j + Groq + Jina — predefined vs. LLM-inferred schema extraction, traversal vs. community retrieval.</p>
      </header>

      <div className="panel">
        <h2>1. Ingest & Compare Schemas</h2>
        <div className="row">
          <button onClick={runIngest} disabled={ingesting}>
            {ingesting ? 'Ingesting...' : 'Run Ingestion (both schemas)'}
          </button>
          <button className="secondary" onClick={fetchComparison}>Fetch Comparison Stats</button>
          <button className="secondary" onClick={resetGraph}>Reset Graph</button>
        </div>
        {status && <div className="status-line">{status}</div>}

        {comparison && (
          <table className="compare-table" style={{ marginTop: 12 }}>
            <thead>
              <tr>
                <th>Metric</th>
                <th>Predefined Schema</th>
                <th>LLM-Inferred Schema</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Node count</td>
                <td>{comparison.predefined.node_count}</td>
                <td>{comparison.llm_inferred.node_count}</td>
              </tr>
              <tr>
                <td>Relationship count</td>
                <td>{comparison.predefined.relationship_count}</td>
                <td>{comparison.llm_inferred.relationship_count}</td>
              </tr>
              <tr>
                <td>Unique node labels</td>
                <td>{comparison.predefined.unique_node_labels.join(', ')}</td>
                <td>{comparison.llm_inferred.unique_node_labels.join(', ')}</td>
              </tr>
              <tr>
                <td>Unique relationship types</td>
                <td>{comparison.predefined.unique_relationship_types.join(', ')}</td>
                <td>{comparison.llm_inferred.unique_relationship_types.join(', ')}</td>
              </tr>
            </tbody>
          </table>
        )}
      </div>

      <div className="panel">
        <h2>2. Ask a Question</h2>
        <div className="row" style={{ marginBottom: 12 }}>
          <label>Schema:&nbsp;
            <select value={schemaMode} onChange={e => setSchemaMode(e.target.value)}>
              <option value="predefined">Predefined</option>
              <option value="llm_inferred">LLM-Inferred</option>
            </select>
          </label>
          <label>Retrieval:&nbsp;
            <select value={retrievalMode} onChange={e => setRetrievalMode(e.target.value)}>
              <option value="traversal">Traversal (multi-hop)</option>
              <option value="community">Community Summary (global)</option>
            </select>
          </label>
        </div>

        <div className="chat-log">
          {messages.map((m, i) => (
            <div key={i} className={`msg ${m.role}`}>
              <ReactMarkdown>{m.text}</ReactMarkdown>
              {m.meta && <div className="meta">{m.meta}</div>}
            </div>
          ))}
        </div>

        <div className="row">
          <input
            type="text"
            placeholder="e.g. How does glacial melt affect the Indus River Dolphin?"
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && askQuestion()}
          />
          <button onClick={askQuestion} disabled={asking}>
            {asking ? 'Thinking...' : 'Ask'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default App
