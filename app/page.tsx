'use client';

import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from 'react';
import { AlertTriangle, ArrowUp, Check, ChevronDown, CircleHelp, Clock3, Coffee, Film, Moon, Plus, ShieldCheck, Sparkles, Sun, Utensils, X } from 'lucide-react';

type Device = 'lighting' | 'curtains' | 'television' | 'audio' | 'dnd' | 'thermostat' | 'room_service' | 'alarm' | 'wake_up_service' | string;
type Step = { step_id: number; source: string; text: string; device: Device | null; action: string | null; confidence: number; value?: string; result?: string; result_message?: string };
type Escalation = { priority: string; reason: string };
type Event = { kind: 'user' | 'agent'; text: string; status?: string; steps?: Step[]; escalations?: Escalation[]; guard?: string | null; time: string };

const examples = [
  { label: 'Set up a movie', text: 'I want to relax and watch a movie', icon: Film },
  { label: 'Get ready for bed', text: 'Get me ready for bed', icon: Moon },
  { label: 'Order a snack', text: 'Could I get some snacks?', icon: Utensils },
  { label: 'Wake me at 7', text: 'Wake me up at 7', icon: Clock3 },
  { label: 'Make it warmer', text: "I'm feeling cold", icon: Sun },
  { label: 'Get a coffee', text: 'Could I get a coffee?', icon: Coffee },
];

const currentTime = () => new Date().toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
const welcome: Event = { kind: 'agent', text: 'Hi, I’m your hotel concierge. Tell me what you’d like to do, and I’ll plan the steps and take care of them.', time: 'Now' };

function deviceName(device: Device | null) {
  const names: Record<string, string> = { lighting: 'Lighting', curtains: 'Curtains', television: 'Television', audio: 'Audio', dnd: 'Do not disturb', thermostat: 'Thermostat', room_service: 'Room service', alarm: 'Alarm', wake_up_service: 'Wake-up call' };
  return names[device || ''] || 'Request';
}

function statusLabel(status: string) { return status.replaceAll('_', ' '); }

function Trace({ event }: { event: Event }) {
  const count = event.steps?.length || 0;
  return <details className="trace">
    <summary><span className="trace-summary-icon"><Clock3 size={14}/></span><span>Plan &amp; activity</span><span className="trace-count">{count} {count === 1 ? 'step' : 'steps'}</span><ChevronDown size={15} className="trace-chevron"/></summary>
    <div className="trace-content">
      {event.guard && <div className={`guard-note ${event.guard}`}><ShieldCheck size={15}/><span>Safety check: {event.guard === 'emergency' ? 'possible emergency escalated' : event.guard === 'injection' ? 'request declined and flagged' : 'please rephrase the request'}</span></div>}
      {count > 0 && <ol className="step-list">{event.steps?.map((step, index) => <li className="step" key={`${step.step_id}-${index}`}>
        <span className={`step-marker ${step.result || 'pending'}`}>{step.result === 'success' ? <Check size={13}/> : step.result === 'failed' || step.result === 'escalated' ? <AlertTriangle size={13}/> : index + 1}</span>
        <div className="step-copy"><div className="step-heading"><strong>{deviceName(step.device)}</strong><span className={`step-status ${step.result || 'pending'}`}>{step.result ? statusLabel(step.result) : 'planned'}</span></div>
          <p>{step.action?.replaceAll('_', ' ')}{step.value ? ` · ${step.value}` : ''}</p>
          {step.result_message && <small>{step.result_message}</small>}
        </div>
      </li>)}</ol>}
      {!count && !event.guard && <p className="no-steps">No device actions were needed.</p>}
      {event.escalations?.map((escalation, index) => <div className="escalation" key={index}><AlertTriangle size={15}/><div><strong>Staff follow-up · {escalation.priority}</strong><p>{escalation.reason}</p></div></div>)}
    </div>
  </details>;
}

export default function Home() {
  const [guestId, setGuestId] = useState('guest_001');
  const [events, setEvents] = useState<Event[]>([welcome]);
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState('');
  const composerRef = useRef<HTMLTextAreaElement>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }); }, [events, isSending]);

  function startNewConversation() {
    setEvents([{ ...welcome, time: 'Now' }]);
    setInput('');
    setError('');
    composerRef.current?.focus();
  }

  async function sendRequest(raw = input) {
    const text = raw.trim();
    if (!text || isSending) return;
    setIsSending(true);
    setError('');
    try {
      const response = await fetch('/api/concierge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ guestId: guestId.trim() || 'demo_guest', text }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'The concierge could not complete that request.');
      setEvents(previous => [...previous,
        { kind: 'user', text, time: currentTime() },
        { kind: 'agent', text: result.guest_message, status: result.status, steps: result.steps, escalations: result.escalations, guard: result.guard_flag, time: currentTime() },
      ]);
      setInput('');
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'The concierge could not complete that request.');
    } finally {
      setIsSending(false);
      composerRef.current?.focus();
    }
  }

  function onSubmit(event: FormEvent<HTMLFormElement>) { event.preventDefault(); void sendRequest(); }
  function onComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void sendRequest(); }
  }

  return <main className="chat-app">
    <aside className="sidebar">
      <a className="brand" href="#conversation" aria-label="Atelier Concierge home"><span className="brand-mark"><Sparkles size={18}/></span><span><strong>ATELIER</strong><small>HOTEL CONCIERGE</small></span></a>
      <div className="sidebar-divider"/>

      <section className="session-section">
        <label htmlFor="guest-id">Guest ID</label>
        <input id="guest-id" value={guestId} onChange={event => setGuestId(event.target.value)} placeholder="e.g. guest_001" autoComplete="off"/>
        <p>Used to apply the guest’s saved preferences.</p>
      </section>

      <section className="examples-section">
        <div className="section-label">TRY A REQUEST</div>
        <div className="example-list">{examples.map(({ label, text, icon: Icon }) => <button className="example" key={label} onClick={() => { setInput(text); composerRef.current?.focus(); }}>
          <span className="example-icon"><Icon size={15}/></span><span>{label}</span>
        </button>)}</div>
      </section>

      <div className="sidebar-bottom"><div className="safety-note"><ShieldCheck size={16}/><span><strong>Safety checks are on</strong><small>Room actions are simulated</small></span></div></div>
    </aside>

    <section className="conversation" id="conversation">
      <header className="chat-header"><div><span className="header-title">Concierge</span><span className="connection"><i/> {isSending ? 'Working' : 'Ready'}</span></div><button className="header-new" onClick={startNewConversation} aria-label="Start a new conversation"><Plus size={17}/><span>New chat</span></button></header>

      <div className="message-scroll">
        <div className="messages">
          {events.map((event, index) => event.kind === 'user' ? <article className="message user-message" key={`${index}-${event.time}`}>
            <div className="message-meta"><span>You</span><time>{event.time}</time></div><div className="user-bubble">{event.text}</div>
          </article> : <article className="message assistant-message" key={`${index}-${event.time}`}>
            <div className="assistant-avatar"><Sparkles size={15}/></div><div className="assistant-content"><div className="message-meta"><span>Concierge</span><time>{event.time}</time>{event.status && <span className={`status-tag ${event.status}`}>{statusLabel(event.status)}</span>}</div>
              <p className="assistant-text">{event.text}</p>{event.status && <Trace event={event}/>}
            </div>
          </article>)}
          {isSending && <div className="thinking"><span className="thinking-mark"><Sparkles size={14}/></span><span>Planning and carrying out your request…</span><i/><i/><i/></div>}
          <div ref={endRef}/>
        </div>
      </div>

      <div className="composer-area">
        {error && <div className="error-note" role="alert"><AlertTriangle size={15}/><span>{error}</span><button onClick={() => setError('')} aria-label="Dismiss error"><X size={15}/></button></div>}
        <form className="composer" onSubmit={onSubmit}>
          <textarea ref={composerRef} id="composer" value={input} onChange={event => setInput(event.target.value)} onKeyDown={onComposerKeyDown} placeholder="Tell me what you’d like to do…" rows={1} disabled={isSending}/>
          <button className="send-button" type="submit" disabled={!input.trim() || isSending} aria-label="Send message"><ArrowUp size={18}/></button>
        </form>
        <div className="composer-hint"><span><ShieldCheck size={13}/> Requests are checked before actions are taken</span><span><kbd>Enter</kbd> to send · <kbd>Shift + Enter</kbd> for a new line</span></div>
      </div>
    </section>
  </main>;
}
