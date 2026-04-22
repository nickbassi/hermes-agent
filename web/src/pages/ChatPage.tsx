import { useEffect, useRef, useState, useCallback } from "react";
import {
  ChevronDown,
  ChevronRight,
  MessagesSquare,
  Plus,
  Send,
  Loader2,
  Volume2,
  Square,
  Copy,
  Check,
  Menu,
  X,
  Cpu,
  Sparkles,
  Trash2,
} from "lucide-react";
import { api } from "@/lib/api";
import type { SessionInfo, SessionMessage, ChatModelEntry } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Markdown } from "@/components/Markdown";
import { timeAgo } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Tool-call disclosure
// ---------------------------------------------------------------------------

function ToolCallBlock({
  toolCall,
}: {
  toolCall: { id: string; function: { name: string; arguments: string } };
}) {
  const [open, setOpen] = useState(false);
  let args = toolCall.function.arguments;
  try {
    args = JSON.stringify(JSON.parse(args), null, 2);
  } catch {
    // leave as-is
  }
  return (
    <div className="mt-2 rounded-md border border-warning/25 bg-warning/5">
      <button
        type="button"
        className="flex w-full items-center gap-2 px-3 py-1.5 text-xs text-warning cursor-pointer hover:bg-warning/10 transition-colors rounded-t-md"
        onClick={() => setOpen(!open)}
      >
        {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        <span className="font-mono font-medium">{toolCall.function.name}</span>
        <span className="text-warning/50 ml-auto text-[10px]">{toolCall.id.slice(0, 8)}</span>
      </button>
      {open && (
        <pre className="border-t border-warning/20 px-3 py-2 text-xs text-warning/80 overflow-x-auto whitespace-pre-wrap font-mono">
          {args}
        </pre>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Message bubble — more chat-like now
// ---------------------------------------------------------------------------

function MessageBubble({
  msg,
  onSpeak,
  speakingId,
}: {
  msg: SessionMessage;
  onSpeak: (text: string, id: string) => void;
  speakingId: string | null;
}) {
  const [copied, setCopied] = useState(false);
  const id = `${msg.role}-${msg.timestamp ?? Math.random()}`;

  const handleCopy = async () => {
    if (!msg.content) return;
    try {
      await navigator.clipboard.writeText(msg.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // ignore
    }
  };

  // System / tool messages — compact, monospace, collapsed look
  if (msg.role === "system" || msg.role === "tool") {
    const tone = msg.role === "tool" ? "text-warning/90" : "text-muted-foreground";
    const bg = msg.role === "tool" ? "bg-warning/5 border-warning/20" : "bg-muted/40 border-border";
    return (
      <div className={`border ${bg} rounded-md px-3 py-2`}>
        <div className="flex items-center gap-2 mb-1">
          <span className={`text-[10px] font-mono uppercase tracking-wider ${tone}`}>
            {msg.tool_name ? `tool · ${msg.tool_name}` : msg.role}
          </span>
          {msg.timestamp && (
            <span className="text-[10px] text-muted-foreground/70">{timeAgo(msg.timestamp)}</span>
          )}
        </div>
        {msg.content && (
          <div className="text-xs text-foreground/80 whitespace-pre-wrap font-mono break-words">
            {msg.content.length > 800 ? msg.content.slice(0, 800) + "\n…" : msg.content}
          </div>
        )}
      </div>
    );
  }

  const isUser = msg.role === "user";

  return (
    <div className={`flex gap-2 sm:gap-3 ${isUser ? "flex-row-reverse" : "flex-row"}`}>
      {/* Avatar */}
      <div
        className={`shrink-0 h-8 w-8 rounded-full flex items-center justify-center text-[10px] font-semibold uppercase tracking-wider ${
          isUser
            ? "bg-primary/15 text-primary"
            : "bg-success/15 text-success"
        }`}
      >
        {isUser ? "You" : "AI"}
      </div>

      {/* Bubble */}
      <div className={`flex flex-col gap-1 min-w-0 max-w-[85%] sm:max-w-[78%] ${isUser ? "items-end" : "items-start"}`}>
        <div
          className={`rounded-2xl px-3 py-2 sm:px-4 sm:py-2.5 text-sm leading-relaxed break-words ${
            isUser
              ? "bg-primary/15 text-foreground rounded-tr-sm"
              : "bg-secondary/60 text-foreground rounded-tl-sm"
          }`}
        >
          {msg.content && (
            isUser ? (
              <div className="whitespace-pre-wrap">{msg.content}</div>
            ) : (
              <Markdown content={msg.content} />
            )
          )}
          {msg.tool_calls && msg.tool_calls.length > 0 && (
            <div className="mt-2 space-y-1">
              {msg.tool_calls.map((tc) => (
                <ToolCallBlock key={tc.id} toolCall={tc} />
              ))}
            </div>
          )}
        </div>

        {/* Actions row */}
        {msg.content && (
          <div className={`flex items-center gap-1 px-1 text-[10px] text-muted-foreground ${isUser ? "flex-row-reverse" : ""}`}>
            {msg.timestamp && <span>{timeAgo(msg.timestamp)}</span>}
            <button
              type="button"
              onClick={handleCopy}
              className="p-1 hover:bg-secondary rounded transition-colors"
              title="Copy"
            >
              {copied ? <Check className="h-3 w-3 text-success" /> : <Copy className="h-3 w-3" />}
            </button>
            {!isUser && (
              <button
                type="button"
                onClick={() => onSpeak(msg.content!, id)}
                className="p-1 hover:bg-secondary rounded transition-colors"
                title={speakingId === id ? "Stop" : "Read aloud"}
              >
                {speakingId === id ? (
                  <Square className="h-3 w-3 text-primary" />
                ) : (
                  <Volume2 className="h-3 w-3" />
                )}
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ChatPage
// ---------------------------------------------------------------------------

export default function ChatPage() {
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [messages, setMessages] = useState<SessionMessage[]>([]);
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [toolStatus, setToolStatus] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Model picker
  const [availableModels, setAvailableModels] = useState<ChatModelEntry[]>([]);
  const [chosenModel, setChosenModel] = useState<ChatModelEntry | null>(null);
  const [modelMenuOpen, setModelMenuOpen] = useState(false);

  // TTS
  const [speakingId, setSpeakingId] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [autoSpeak, setAutoSpeak] = useState(false);

  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const loadSessions = useCallback(async () => {
    try {
      const resp = await api.getSessions(50, 0);
      setSessions(resp.sessions);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  const loadMessages = useCallback(async (sid: string) => {
    setLoadingMessages(true);
    try {
      const resp = await api.getSessionMessages(sid);
      setMessages(resp.messages);
    } catch (e) {
      setError(String(e));
      setMessages([]);
    } finally {
      setLoadingMessages(false);
    }
  }, []);

  const loadModels = useCallback(async () => {
    try {
      const resp = await api.listChatModels();
      setAvailableModels(resp.models);
      const current = resp.models.find((m) => m.is_current);
      if (current) setChosenModel(current);
    } catch {
      // Non-fatal — picker just won't have options
    }
  }, []);

  useEffect(() => {
    loadSessions();
    loadModels();
  }, [loadSessions, loadModels]);

  useEffect(() => {
    if (selectedId) {
      loadMessages(selectedId);
    } else {
      setMessages([]);
    }
  }, [selectedId, loadMessages]);

  // Auto-scroll on new messages / streaming deltas
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, streamingText, toolStatus]);

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 200) + "px";
  }, [input]);

  const stopAudio = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = "";
      audioRef.current = null;
    }
    setSpeakingId(null);
  }, []);

  const handleSpeak = useCallback(
    async (text: string, id: string) => {
      if (speakingId === id) {
        stopAudio();
        return;
      }
      stopAudio();
      setSpeakingId(id);
      try {
        const blob = await api.speakText(text);
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audioRef.current = audio;
        audio.onended = () => {
          URL.revokeObjectURL(url);
          setSpeakingId(null);
        };
        audio.onerror = () => {
          URL.revokeObjectURL(url);
          setSpeakingId(null);
          setError("TTS playback failed");
        };
        await audio.play();
      } catch (e) {
        setSpeakingId(null);
        setError(`TTS: ${e}`);
      }
    },
    [speakingId, stopAudio],
  );

  useEffect(() => () => stopAudio(), [stopAudio]);

  const handleNewChat = () => {
    setSelectedId(null);
    setMessages([]);
    setStreamingText("");
    setToolStatus("");
    setError(null);
    setSidebarOpen(false);
    textareaRef.current?.focus();
  };

  const handleSelect = (sid: string) => {
    if (streaming) return;
    setSelectedId(sid);
    setStreamingText("");
    setToolStatus("");
    setError(null);
    setSidebarOpen(false);
  };

  const handleDeleteSession = async (sid: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm("Delete this session? This cannot be undone.")) return;
    try {
      await api.deleteSession(sid);
      if (selectedId === sid) {
        setSelectedId(null);
        setMessages([]);
      }
      await loadSessions();
    } catch (err) {
      setError(String(err));
    }
  };

  const handleSend = async () => {
    const text = input.trim();
    if (!text || streaming) return;
    setInput("");
    setError(null);

    const userMsg: SessionMessage = { role: "user", content: text, timestamp: Date.now() / 1000 };
    setMessages((prev) => [...prev, userMsg]);
    setStreamingText("");
    setToolStatus("");
    setStreaming(true);

    const req: { session_id?: string; message: string; model?: string; provider?: string } = {
      message: text,
    };
    if (selectedId) req.session_id = selectedId;
    if (chosenModel && !chosenModel.is_current) {
      req.model = chosenModel.model;
      if (chosenModel.provider) req.provider = chosenModel.provider;
    }

    let finalSessionId: string | null = selectedId;
    let accumulated = "";
    try {
      for await (const evt of api.chatStream(req)) {
        if (evt.type === "delta") {
          accumulated += evt.text;
          setStreamingText((prev) => prev + evt.text);
        } else if (evt.type === "tool_start") {
          setToolStatus(`⚙ ${evt.name}…`);
        } else if (evt.type === "tool_end") {
          setToolStatus(`✓ ${evt.name}`);
        } else if (evt.type === "done") {
          finalSessionId = evt.session_id;
        } else if (evt.type === "error") {
          setError(evt.message);
        }
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setStreaming(false);
      setStreamingText("");
      setToolStatus("");
      if (finalSessionId) {
        if (finalSessionId !== selectedId) setSelectedId(finalSessionId);
        await loadMessages(finalSessionId);
      }
      loadSessions();

      if (autoSpeak && accumulated) {
        handleSpeak(accumulated, `auto-${Date.now()}`);
      }
    }
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const activeTitle = selectedId
    ? sessions.find((s) => s.id === selectedId)?.title ?? `Session ${selectedId.slice(0, 12)}…`
    : "New Chat";

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  return (
    <div className="flex gap-3 h-[calc(100vh-9rem)] relative">
      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/40 z-30 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar -------------------------------------------------------- */}
      <aside
        className={`
          ${sidebarOpen ? "translate-x-0" : "-translate-x-full"}
          md:translate-x-0 fixed md:static inset-y-0 left-0 z-40
          w-72 md:w-64 shrink-0 border-r md:border border-border
          flex flex-col bg-background md:bg-background/40 transition-transform
        `}
      >
        <div className="p-2 border-b border-border flex items-center gap-2">
          <Button
            onClick={handleNewChat}
            variant="outline"
            className="flex-1 justify-start gap-2"
            disabled={streaming}
          >
            <Plus className="h-4 w-4" />
            New Chat
          </Button>
          <Button
            variant="ghost"
            className="md:hidden px-2"
            onClick={() => setSidebarOpen(false)}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
        <div className="flex-1 overflow-y-auto">
          {sessions.length === 0 && (
            <div className="p-3 text-xs text-muted-foreground">No sessions yet.</div>
          )}
          {sessions.map((s) => {
            const active = s.id === selectedId;
            const title = s.title && s.title !== "Untitled"
              ? s.title
              : s.preview || s.id.slice(0, 8);
            return (
              <div
                key={s.id}
                className={`group relative border-b border-border/50 transition-colors ${
                  active
                    ? "bg-primary/10"
                    : "hover:bg-secondary/40"
                }`}
              >
                <button
                  type="button"
                  onClick={() => handleSelect(s.id)}
                  className={`block w-full text-left px-3 py-2 pr-8 ${
                    active ? "text-foreground" : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <div className="text-xs font-medium truncate">{title}</div>
                  <div className="text-[10px] text-muted-foreground/80 flex items-center gap-2 mt-0.5">
                    <span>{s.source ?? "cli"}</span>
                    <span>·</span>
                    <span>{s.message_count} msg</span>
                    <span className="ml-auto">{timeAgo(s.last_active)}</span>
                  </div>
                </button>
                <button
                  type="button"
                  onClick={(e) => handleDeleteSession(s.id, e)}
                  className="absolute right-1 top-1/2 -translate-y-1/2 p-1.5 rounded opacity-0 group-hover:opacity-100 hover:bg-destructive/10 hover:text-destructive transition-all"
                  title="Delete session"
                >
                  <Trash2 className="h-3 w-3" />
                </button>
              </div>
            );
          })}
        </div>
      </aside>

      {/* Main pane ------------------------------------------------------ */}
      <section className="flex-1 flex flex-col border border-border bg-background/40 min-w-0 rounded-sm">
        <header className="px-3 sm:px-4 py-2 border-b border-border flex items-center gap-2 flex-wrap">
          <Button
            variant="ghost"
            className="md:hidden px-2"
            onClick={() => setSidebarOpen(true)}
          >
            <Menu className="h-4 w-4" />
          </Button>
          <MessagesSquare className="h-4 w-4 text-muted-foreground shrink-0" />
          <h2 className="text-sm font-display tracking-wider uppercase truncate flex-1 min-w-0">
            {activeTitle}
          </h2>
          {loadingMessages && <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />}

          {/* Model picker */}
          <div className="relative">
            <button
              type="button"
              onClick={() => setModelMenuOpen((v) => !v)}
              className="flex items-center gap-1.5 px-2 py-1 text-[11px] font-mono border border-border hover:bg-secondary/60 rounded transition-colors"
              disabled={streaming}
            >
              <Cpu className="h-3 w-3 text-primary" />
              <span className="truncate max-w-[140px] sm:max-w-[220px]">
                {chosenModel?.model ?? "default"}
              </span>
              <ChevronDown className="h-3 w-3 opacity-60" />
            </button>
            {modelMenuOpen && (
              <>
                <div
                  className="fixed inset-0 z-40"
                  onClick={() => setModelMenuOpen(false)}
                />
                <div className="absolute right-0 top-full mt-1 z-50 w-72 max-h-80 overflow-y-auto border border-border bg-popover shadow-lg rounded">
                  {availableModels.length === 0 && (
                    <div className="px-3 py-2 text-xs text-muted-foreground">
                      No models configured. Add some under <code>model_aliases</code> in config.
                    </div>
                  )}
                  {availableModels.map((m) => {
                    const active = chosenModel?.model === m.model && chosenModel?.provider === m.provider;
                    return (
                      <button
                        key={`${m.alias}-${m.model}`}
                        type="button"
                        onClick={() => {
                          setChosenModel(m);
                          setModelMenuOpen(false);
                        }}
                        className={`block w-full text-left px-3 py-2 text-xs hover:bg-secondary/60 transition-colors ${
                          active ? "bg-primary/10" : ""
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          {active && <Check className="h-3 w-3 text-success" />}
                          <span className="font-medium truncate">{m.alias}</span>
                          {m.is_current && (
                            <span className="ml-auto text-[9px] uppercase tracking-wider text-success/80">
                              default
                            </span>
                          )}
                        </div>
                        <div className="text-[10px] text-muted-foreground font-mono truncate mt-0.5">
                          {m.provider ? `${m.provider} · ` : ""}{m.model}
                        </div>
                      </button>
                    );
                  })}
                </div>
              </>
            )}
          </div>

          {/* Auto-TTS toggle */}
          <button
            type="button"
            onClick={() => {
              if (autoSpeak) stopAudio();
              setAutoSpeak((v) => !v);
            }}
            className={`flex items-center gap-1 px-2 py-1 text-[11px] font-mono border rounded transition-colors ${
              autoSpeak
                ? "border-primary/50 bg-primary/10 text-primary"
                : "border-border hover:bg-secondary/60"
            }`}
            title={autoSpeak ? "Auto-speak ON" : "Auto-speak replies"}
          >
            <Volume2 className="h-3 w-3" />
            <span className="hidden sm:inline">TTS</span>
          </button>
        </header>

        {/* Messages */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto p-3 sm:p-5 flex flex-col gap-4">
          {messages.length === 0 && !streamingText && !selectedId && (
            <div className="text-center text-muted-foreground text-sm py-16 flex flex-col items-center gap-2">
              <Sparkles className="h-8 w-8 text-primary/40" />
              <div className="font-display tracking-wider uppercase text-xs">
                Start a new conversation
              </div>
              <div className="text-xs">
                Type a message below. Enter to send, Shift+Enter for newline.
              </div>
            </div>
          )}
          {messages.map((m, i) => (
            <MessageBubble
              key={i}
              msg={m}
              onSpeak={handleSpeak}
              speakingId={speakingId}
            />
          ))}

          {/* Streaming bubble */}
          {(streamingText || toolStatus) && (
            <div className="flex gap-2 sm:gap-3">
              <div className="shrink-0 h-8 w-8 rounded-full bg-success/15 text-success flex items-center justify-center text-[10px] font-semibold uppercase tracking-wider">
                AI
              </div>
              <div className="flex flex-col gap-1 min-w-0 max-w-[85%] sm:max-w-[78%]">
                <div className="rounded-2xl rounded-tl-sm bg-secondary/60 px-3 py-2 sm:px-4 sm:py-2.5 text-sm leading-relaxed">
                  {streamingText && <Markdown content={streamingText} />}
                  {toolStatus && (
                    <div className="text-xs text-muted-foreground mt-1 font-mono flex items-center gap-1.5">
                      <Loader2 className="h-3 w-3 animate-spin" />
                      {toolStatus}
                    </div>
                  )}
                  {!streamingText && !toolStatus && (
                    <Loader2 className="h-3 w-3 animate-spin text-success" />
                  )}
                </div>
              </div>
            </div>
          )}

          {error && (
            <div className="bg-destructive/10 border border-destructive/30 text-destructive text-xs p-3 font-mono rounded">
              {error}
            </div>
          )}
        </div>

        {/* Composer */}
        <div className="border-t border-border p-2 sm:p-3">
          <div className="flex gap-2 items-end">
            <textarea
              ref={textareaRef}
              className="flex-1 resize-none bg-background border border-border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-primary/50 min-h-[2.5rem] max-h-48"
              placeholder="Type a message…"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              rows={1}
              disabled={streaming}
            />
            <Button
              onClick={handleSend}
              disabled={streaming || !input.trim()}
              className="gap-1 h-10"
            >
              {streaming ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
              <span className="hidden sm:inline">Send</span>
            </Button>
          </div>
        </div>
      </section>
    </div>
  );
}
