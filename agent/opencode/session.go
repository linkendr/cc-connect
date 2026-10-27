package opencode

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/chenhg5/cc-connect/core"
)

// opencodeSession manages multi-turn conversations with the OpenCode CLI.
// Each Send() launches a new `opencode run --format json` process
// with --session for conversation continuity.
type opencodeSession struct {
	cmd               string
	extraArgs         []string // extra args from cmd, prepended before opencode args
	workDir           string
	model             string
	mode              string
	agentName         string
	extraEnv          []string
	events            chan core.Event
	chatID            atomic.Value // stores string — OpenCode session ID
	ctx               context.Context
	cancel            context.CancelFunc
	wg                sync.WaitGroup
	alive             atomic.Bool
	expectingContinue atomic.Bool // true when compaction_continue received, waiting for next step
	resultSent        atomic.Bool // true when EventResult has been sent for this turn

	// Process lifecycle owned by the session: the turns whose readers are
	// still running. Each turn's process group is killed at turn end or on
	// Close, always while the CLI (the group leader) is not yet reaped, so
	// the group ID cannot have been reused by an unrelated process.
	procMu sync.Mutex
	procs  map[*opencodeProc]struct{}
}

// opencodeProc is one `opencode run` turn: its process group and the read
// ends of its stdout/stderr pipes, which the session owns.
type opencodeProc struct {
	cmd     *exec.Cmd
	stdoutR *os.File
	stderrR *os.File

	mu         sync.Mutex
	started    bool // cmd.Start succeeded
	reaped     bool // the leader is being or has been reaped: never signal its group again
	killedByUs bool // one of our kills was sent while the leader was still running
}

// markOurKill records, before a kill (p.mu held), whether the leader is
// still running, so a SIGKILL exit can be told apart from an external one.
// An unreadable state counts as running (our kill).
func (p *opencodeProc) markOurKill() {
	if p.cmd.Process == nil {
		return
	}
	if exited, known := leaderExited(p.cmd.Process.Pid); !known || !exited {
		p.killedByUs = true
	}
}

// endedByOurKill reports whether a SIGKILL exit came from our own kill.
func (p *opencodeProc) endedByOurKill() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.killedByUs
}

// kill SIGKILLs the turn's process group (the child alone on Windows), but
// only while the leader is unreaped: an exited-but-unreaped leader keeps the
// group ID reserved, so the signal can only reach this turn's processes.
func (p *opencodeProc) kill() {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.started && !p.reaped {
		p.markOurKill()
		killProcessGroup(p.cmd)
	}
}

// reap ends the turn's process tree and collects the CLI's exit status:
// the group is killed (twice, for a child forked during the first signal)
// before Wait, and no kill is sent after reaping starts. Idempotent.
func (p *opencodeProc) reap() error {
	p.mu.Lock()
	if p.reaped {
		p.mu.Unlock()
		return nil
	}
	p.markOurKill()
	killProcessGroup(p.cmd)
	killProcessGroup(p.cmd)
	p.reaped = true
	p.mu.Unlock()
	return p.cmd.Wait()
}

// pipeWaitDelay bounds how long a turn waits for stderr to reach EOF after
// stdout did (a descendant outside the group may hold it open), and
// cmd.WaitDelay for the stdin copy.
var pipeWaitDelay = 2 * time.Second

// lockedBuffer is a bytes.Buffer safe for one writer and concurrent readers.
type lockedBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *lockedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *lockedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

func (s *opencodeSession) addProc(p *opencodeProc) {
	s.procMu.Lock()
	defer s.procMu.Unlock()
	if s.procs == nil {
		s.procs = make(map[*opencodeProc]struct{})
	}
	s.procs[p] = struct{}{}
}

// finishProc drops a turn once both of its readers have finished and its
// leader is reaped.
func (s *opencodeSession) finishProc(p *opencodeProc) {
	s.procMu.Lock()
	defer s.procMu.Unlock()
	delete(s.procs, p)
}

func newOpencodeSession(ctx context.Context, cmd string, extraArgs []string, workDir, model, mode, agentName, resumeID string, extraEnv []string) (*opencodeSession, error) {
	sessionCtx, cancel := context.WithCancel(ctx)

	s := &opencodeSession{
		cmd:       cmd,
		extraArgs: extraArgs,
		workDir:   workDir,
		model:     model,
		mode:      mode,
		agentName: agentName,
		extraEnv:  extraEnv,
		events:    make(chan core.Event, 64),
		ctx:       sessionCtx,
		cancel:    cancel,
	}
	s.alive.Store(true)

	if resumeID != "" && resumeID != core.ContinueSession {
		s.chatID.Store(resumeID)
	}

	return s, nil
}

func (s *opencodeSession) Send(prompt string, messageID string, images []core.ImageAttachment, files []core.FileAttachment) error {
	if len(files) > 0 {
		filePaths := core.SaveFilesToDisk(s.workDir, messageID, files)
		prompt = core.AppendFileRefs(prompt, filePaths)
	}
	prompt, imagePaths, err := s.stageImages(prompt, images)
	if err != nil {
		return err
	}
	if !s.alive.Load() {
		return fmt.Errorf("session is closed")
	}

	s.resultSent.Store(false)
	s.expectingContinue.Store(false)

	chatID := s.CurrentSessionID()
	isResume := chatID != ""

	args := s.buildRunArgs(prompt, imagePaths, chatID)

	slog.Debug("opencodeSession: launching", "resume", isResume, "args", core.RedactArgs(args))

	cmd := exec.CommandContext(s.ctx, s.cmd, args...)
	prepareProcessGroup(cmd)
	proc := &opencodeProc{cmd: cmd}
	// Context cancellation (Close, engine shutdown) kills through proc, so
	// the kill is skipped once the leader is being reaped.
	cmd.Cancel = func() error {
		proc.kill()
		return nil
	}
	cmd.WaitDelay = pipeWaitDelay // stdin copy only; stdout/stderr are our own pipes
	cmd.Dir = s.workDir
	env := os.Environ()
	if len(s.extraEnv) > 0 {
		env = core.MergeEnv(env, s.extraEnv)
	}
	cmd.Env = env

	// The session owns both ends of the output pipes (instead of exec's
	// StdoutPipe / Stderr copy), so Close can always close the read ends
	// and unblock the readers, whatever a descendant does with its copies.
	stdoutR, stdoutW, err := os.Pipe()
	if err != nil {
		return fmt.Errorf("opencodeSession: stdout pipe: %w", err)
	}
	stderrR, stderrW, err := os.Pipe()
	if err != nil {
		_ = stdoutR.Close()
		_ = stdoutW.Close()
		return fmt.Errorf("opencodeSession: stderr pipe: %w", err)
	}
	cmd.Stdout = stdoutW
	cmd.Stderr = stderrW
	cmd.Stdin = strings.NewReader(prompt)

	proc.mu.Lock()
	startErr := cmd.Start()
	proc.started = startErr == nil
	proc.mu.Unlock()
	// The child has its own copies now; ours must close so EOF can arrive.
	_ = stdoutW.Close()
	_ = stderrW.Close()
	if startErr != nil {
		_ = stdoutR.Close()
		_ = stderrR.Close()
		return fmt.Errorf("opencodeSession: start: %w", startErr)
	}

	proc.stdoutR, proc.stderrR = stdoutR, stderrR
	s.addProc(proc)
	stderrBuf := &lockedBuffer{}
	stderrDone := make(chan struct{})
	s.wg.Add(2)
	go func() {
		defer s.wg.Done()
		defer close(stderrDone)
		_, _ = io.Copy(stderrBuf, stderrR)
		_ = stderrR.Close()
	}()
	go s.readLoop(cmd, proc, stderrBuf, stderrDone)

	return nil
}

func (s *opencodeSession) stageImages(prompt string, images []core.ImageAttachment) (string, []string, error) {
	if len(images) == 0 {
		return prompt, nil, nil
	}

	imgDir := filepath.Join(s.workDir, ".cc-connect", "images")
	if err := os.MkdirAll(imgDir, 0o755); err != nil {
		return "", nil, fmt.Errorf("opencodeSession: create image dir: %w", err)
	}

	imagePaths := make([]string, 0, len(images))
	for i, img := range images {
		ext := opencodeImageExt(img.MimeType)
		fname := fmt.Sprintf("img_%d_%d%s", time.Now().UnixMilli(), i, ext)
		fpath := filepath.Join(imgDir, fname)
		if err := os.WriteFile(fpath, img.Data, 0o644); err != nil {
			return "", nil, fmt.Errorf("opencodeSession: save image: %w", err)
		}
		imagePaths = append(imagePaths, fpath)
	}

	if prompt == "" {
		prompt = "Please analyze the attached image(s)."
	}

	return prompt, imagePaths, nil
}

func opencodeImageExt(mimeType string) string {
	switch mimeType {
	case "image/jpeg":
		return ".jpg"
	case "image/gif":
		return ".gif"
	case "image/webp":
		return ".webp"
	default:
		return ".png"
	}
}

func (s *opencodeSession) buildRunArgs(prompt string, imagePaths []string, chatID string) []string {
	args := append(append([]string{}, s.extraArgs...), "run", "--format", "json")

	if chatID != "" {
		args = append(args, "--session", chatID)
	}
	if s.agentName != "" {
		args = append(args, "--agent", s.agentName)
	}
	if s.model != "" {
		args = append(args, "--model", s.model)
	}
	// The working directory is set as the process cwd in Send (cmd.Dir), like
	// the other opencode invocations. `opencode run` has no --dir flag in
	// opencode v2 and rejects it ("Unrecognized flag: --dir").

	// Enable thinking blocks.
	args = append(args, "--thinking")

	// In yolo/auto mode, skip permission prompts entirely so headless
	// runs don't get stuck with auto-rejected external-directory ops.
	if s.mode == "yolo" {
		args = append(args, "--dangerously-skip-permissions")
	}

	for _, imagePath := range imagePaths {
		if imagePath == "" {
			continue
		}
		args = append(args, "--file", imagePath)
	}

	return args
}

func (s *opencodeSession) readLoop(cmd *exec.Cmd, proc *opencodeProc, stderrBuf *lockedBuffer, stderrDone <-chan struct{}) {
	defer s.wg.Done()
	stdout := proc.stdoutR
	// endTurn finishes the turn on every path: stderr gets a bounded time
	// to reach EOF after stdout did, then our read end is closed and its
	// reader joined; the process group is killed before the leader is
	// reaped; the record is dropped only after both readers are done.
	ended := false
	var exitErr error
	endTurn := func() error {
		if ended {
			return exitErr
		}
		ended = true
		_ = stdout.Close()
		select {
		case <-stderrDone:
		case <-time.After(pipeWaitDelay):
		case <-s.ctx.Done():
		}
		_ = proc.stderrR.Close()
		<-stderrDone
		exitErr = proc.reap()
		s.finishProc(proc)
		return exitErr
	}
	defer func() { _ = endTurn() }()

	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 0, 64*1024), 10*1024*1024)

	sawStdout := false
	for scanner.Scan() {
		line := scanner.Text()
		sawStdout = true
		if line == "" {
			continue
		}

		var raw map[string]any
		if err := json.Unmarshal([]byte(line), &raw); err != nil {
			slog.Debug("opencodeSession: non-JSON line", "line", line)
			continue
		}

		s.handleEvent(raw)
	}

	if err := scanner.Err(); err != nil {
		slog.Error("opencodeSession: scanner error", "error", err)
		evt := core.Event{Type: core.EventError, Error: fmt.Errorf("read stdout: %w", err)}
		select {
		case s.events <- evt:
		case <-s.ctx.Done():
			return
		}
		return
	}

	// stdout is at EOF (normal completion closes stdout and stderr with the
	// CLI's exit): end the turn and collect the exit status.
	turnErr := endTurn()
	stderrMsg := stderrBuf.String()
	// Non-zero exit is decided from the exit status, not from Wait's
	// error: a clean exit (code 0) stays main's empty-result path, and so
	// does a CLI that our own kill ended while it was still running. An
	// external SIGKILL is a non-zero exit.
	ourKill := killedBySIGKILL(cmd.ProcessState) && proc.endedByOurKill()
	nonZeroExit := cmd.ProcessState != nil && cmd.ProcessState.ExitCode() != 0 && !ourKill
	if stderrMsg == "" && nonZeroExit && !sawStdout && s.ctx.Err() == nil {
		// The process failed without a word (crash, kill, bad flag that
		// prints nothing): report an error instead of synthesizing an empty
		// successful result. Unclassified = provider-side for the agent
		// fallback chain.
		slog.Error("opencodeSession: process exited without output", "error", turnErr)
		evt := core.Event{Type: core.EventError, Error: fmt.Errorf("opencode exited without output: %v", turnErr)}
		select {
		case s.events <- evt:
		case <-s.ctx.Done():
		}
		return
	}
	if stderrMsg != "" {
		slog.Error("opencodeSession: process error", "stderr", truncate(stderrMsg, 500))
		if strings.Contains(stderrMsg, "Session not found") {
			s.chatID.Store("")
			slog.Warn("opencodeSession: cleared stale session ID")
		}
		evt := core.Event{Type: core.EventError, Error: fmt.Errorf("%s", stderrMsg), ErrorKind: classifyOpencodeText(stderrMsg)}
		select {
		case s.events <- evt:
		case <-s.ctx.Done():
		}
		return
	}

	// Check if we received compaction_continue before readLoop ended.
	// If so, OpenCode will continue with a new turn - do NOT send EventResult.
	// The subsequent process will send its own EventResult when it finishes.
	if s.expectingContinue.Load() {
		slog.Info("opencodeSession: readLoop ended after compaction_continue, skipping EventResult", "session_id", s.CurrentSessionID())
		s.expectingContinue.Store(false)
		return
	}

	slog.Debug("opencodeSession: readLoop complete, sending fallback EventResult", "session_id", s.CurrentSessionID())
	s.sendEventResult()
}

// OpenCode NDJSON event structure:
//
//	{ "type": "text|tool_use|reasoning|step_start|step_finish",
//	  "part": { "type": "text|tool|reasoning|step-start|step-finish", ... } }
func (s *opencodeSession) handleEvent(raw map[string]any) {
	eventType, _ := raw["type"].(string)

	switch eventType {
	case "text":
		s.handleText(raw)
	case "tool_use":
		s.handleToolUse(raw)
	case "reasoning":
		s.handleReasoning(raw)
	case "step_start":
		s.handleStepStart(raw)
	case "step_finish":
		s.handleStepFinish(raw)
	case "error":
		s.handleError(raw)
	default:
		b, _ := json.Marshal(raw)
		slog.Debug("opencodeSession: unhandled event", "type", eventType, "raw", string(b))
	}
}

func (s *opencodeSession) handleText(raw map[string]any) {
	part, _ := raw["part"].(map[string]any)
	if part == nil {
		return
	}
	text, _ := part["text"].(string)

	// Extract metadata and synthetic flags to identify compaction_continue
	metadata, _ := part["metadata"].(map[string]any)
	synthetic, _ := part["synthetic"].(bool)

	// Check for compaction_continue: this is OpenCode's auto-continuation signal.
	// When received, we should NOT send EventText to engine, but mark that we expect
	// a continuation (next step_start will start a new turn without EventResult).
	if synthetic && metadata != nil {
		if cc, ok := metadata["compaction_continue"].(bool); ok && cc {
			slog.Info("opencodeSession: compaction_continue detected, marking expectingContinue", "session_id", s.CurrentSessionID())
			s.expectingContinue.Store(true)
			// Do NOT send EventText - this is internal continuation signal
			return
		}
	}

	if text != "" {
		evt := core.Event{Type: core.EventText, Content: text, Metadata: metadata, Synthetic: synthetic}
		select {
		case s.events <- evt:
		case <-s.ctx.Done():
			return
		}
	}
}

func (s *opencodeSession) handleToolUse(raw map[string]any) {
	part, _ := raw["part"].(map[string]any)
	if part == nil {
		return
	}

	toolName, _ := part["tool"].(string)

	state, _ := part["state"].(map[string]any)
	status := ""
	if state != nil {
		status, _ = state["status"].(string)
	}

	// Extract tool input summary for display
	input := extractToolInput(state)

	if status == "completed" {
		// OpenCode bundles call + result in one event; emit both for UI.
		useEvt := core.Event{Type: core.EventToolUse, ToolName: toolName, ToolInput: input}
		select {
		case s.events <- useEvt:
		case <-s.ctx.Done():
			return
		}

		output, _ := state["output"].(string)
		resultEvt := core.Event{Type: core.EventToolResult, ToolName: toolName, Content: truncate(output, 500)}
		select {
		case s.events <- resultEvt:
		case <-s.ctx.Done():
			return
		}
	} else {
		evt := core.Event{Type: core.EventToolUse, ToolName: toolName, ToolInput: input}
		select {
		case s.events <- evt:
		case <-s.ctx.Done():
			return
		}

		// When a tool call is rejected (e.g. permission denied in default mode),
		// opencode exits without generating any follow-up text. Surface the rejection
		// reason so the engine has something meaningful to send rather than "(空响应)".
		// This covers the common case where the user has not configured tool permissions
		// and needs guidance to use mode="yolo" or update opencode settings.
		if status == "error" && state != nil {
			errMsg, _ := state["error"].(string)
			if errMsg != "" {
				slog.Info("opencodeSession: tool rejected, surfacing error as text", "tool", toolName, "error", errMsg)
				errEvt := core.Event{Type: core.EventText, Content: errMsg}
				select {
				case s.events <- errEvt:
				case <-s.ctx.Done():
					return
				}
			}
		}
	}
}

func extractToolInput(state map[string]any) string {
	if state == nil {
		return ""
	}
	// Prefer title as a concise description (e.g. "List files in current directory")
	if title, ok := state["title"].(string); ok && title != "" {
		return title
	}
	switch input := state["input"].(type) {
	case string:
		return input
	case map[string]any:
		// Use "description" or "command" fields if available
		if desc, ok := input["description"].(string); ok && desc != "" {
			return desc
		}
		if cmd, ok := input["command"].(string); ok && cmd != "" {
			return cmd
		}
		b, _ := json.Marshal(input)
		return truncate(string(b), 200)
	}
	return ""
}

func (s *opencodeSession) handleReasoning(raw map[string]any) {
	part, _ := raw["part"].(map[string]any)
	if part == nil {
		return
	}
	text, _ := part["text"].(string)
	if text != "" {
		evt := core.Event{Type: core.EventThinking, Content: text}
		select {
		case s.events <- evt:
		case <-s.ctx.Done():
			return
		}
	}
}

func (s *opencodeSession) handleError(raw map[string]any) {
	errMsg := extractErrorMessage(raw)
	kind := classifyOpencodeError(raw, errMsg)
	slog.Error("opencodeSession: agent error", "error", errMsg, "kind", kind)
	evt := core.Event{Type: core.EventError, Error: fmt.Errorf("%s", errMsg), ErrorKind: kind}
	select {
	case s.events <- evt:
	case <-s.ctx.Done():
		return
	}
}

// classifyOpencodeError marks the OpenCode errors that are user-caused or
// terminal, so an agent fallback chain does not hand them to the next agent:
// an aborted message (MessageAbortedError) and a prompt that exceeds the
// model's context window. Everything else stays unclassified, which the
// chain treats as provider-side (auth, quota, rate limit, API errors).
// Without a fallback chain the engine ignores ErrorKind on EventError.
func classifyOpencodeError(raw map[string]any, msg string) core.ErrorKind {
	name := ""
	if errObj, ok := raw["error"].(map[string]any); ok {
		name, _ = errObj["name"].(string)
	}
	switch name {
	case "MessageAbortedError":
		return core.ErrorKindCancelled
	case "ContextOverflowError", "ContextWindowExceededError":
		return core.ErrorKindContextTooLong
	}
	return classifyOpencodeText(msg)
}

// classifyOpencodeText classifies free error text (the formatted JSON
// error, or stderr of a failed `opencode run`): an aborted message and a
// context overflow are terminal; everything else, including auth and quota
// errors, stays unclassified (provider-side).
func classifyOpencodeText(text string) core.ErrorKind {
	if strings.Contains(text, "MessageAbortedError") {
		return core.ErrorKindCancelled
	}
	if strings.Contains(text, "ContextOverflowError") || strings.Contains(text, "ContextWindowExceededError") {
		return core.ErrorKindContextTooLong
	}
	low := strings.ToLower(text)
	for _, p := range []string{"context_length_exceeded", "maximum context length", "context window", "prompt is too long"} {
		if strings.Contains(low, p) {
			return core.ErrorKindContextTooLong
		}
	}
	return core.ErrorKindUnknown
}

// extractErrorMessage tries to pull a human-readable message from various
// OpenCode error JSON shapes.
func extractErrorMessage(raw map[string]any) string {
	// Shape: {"error": {"data": {"message": "..."}, "name": "..."}}
	if errObj, ok := raw["error"].(map[string]any); ok {
		if data, ok := errObj["data"].(map[string]any); ok {
			if msg, ok := data["message"].(string); ok && msg != "" {
				name, _ := errObj["name"].(string)
				if name != "" {
					return name + ": " + msg
				}
				return msg
			}
		}
		if msg, ok := errObj["message"].(string); ok && msg != "" {
			return msg
		}
		if name, ok := errObj["name"].(string); ok && name != "" {
			return name
		}
	}
	// Shape: {"error": "string message"}
	if errStr, ok := raw["error"].(string); ok && errStr != "" {
		return errStr
	}
	// Shape: {"part": {"error": "...", "message": "..."}}
	if part, ok := raw["part"].(map[string]any); ok {
		if msg, ok := part["error"].(string); ok && msg != "" {
			return msg
		}
		if msg, ok := part["message"].(string); ok && msg != "" {
			return msg
		}
	}
	if msg, ok := raw["message"].(string); ok && msg != "" {
		return msg
	}
	b, _ := json.Marshal(raw)
	return string(b)
}

func (s *opencodeSession) handleStepStart(raw map[string]any) {
	sessionID, _ := raw["sessionID"].(string)
	if sessionID == "" {
		part, _ := raw["part"].(map[string]any)
		if part != nil {
			sessionID, _ = part["sessionID"].(string)
		}
	}
	if sessionID != "" {
		s.chatID.Store(sessionID)
		slog.Debug("opencodeSession: session started", "session_id", sessionID)
	}
}

func (s *opencodeSession) handleStepFinish(raw map[string]any) {
	part, _ := raw["part"].(map[string]any)
	reason := ""
	if part != nil {
		reason, _ = part["reason"].(string)
	}
	slog.Debug("opencodeSession: step finished", "reason", reason, "session_id", s.CurrentSessionID())

	if reason == "stop" {
		s.sendEventResult()
	}
}

func (s *opencodeSession) sendEventResult() {
	if s.resultSent.Load() {
		slog.Debug("opencodeSession: EventResult already sent, skipping", "session_id", s.CurrentSessionID())
		return
	}
	s.resultSent.Store(true)
	sid := s.CurrentSessionID()
	evt := core.Event{Type: core.EventResult, SessionID: sid, Done: true}
	select {
	case s.events <- evt:
	case <-s.ctx.Done():
	}
}

// RespondPermission is a no-op — OpenCode handles permissions internally.
func (s *opencodeSession) RespondPermission(_ string, _ core.PermissionResult) error {
	return nil
}

func (s *opencodeSession) Events() <-chan core.Event {
	return s.events
}

func (s *opencodeSession) CurrentSessionID() string {
	v, _ := s.chatID.Load().(string)
	return v
}

func (s *opencodeSession) Alive() bool {
	return s.alive.Load()
}

func (s *opencodeSession) Close() error {
	s.alive.Store(false)
	s.cancel()
	// Kill every running turn's process group (only while its leader is
	// unreaped) and close our read ends, so the readers return whatever a
	// stray descendant holds; each turn then reaps its leader and exits.
	s.procMu.Lock()
	procs := make([]*opencodeProc, 0, len(s.procs))
	for p := range s.procs {
		procs = append(procs, p)
	}
	s.procMu.Unlock()
	for _, p := range procs {
		p.kill()
		_ = p.stdoutR.Close()
		_ = p.stderrR.Close()
	}
	done := make(chan struct{})
	go func() {
		s.wg.Wait()
		close(done)
	}()
	select {
	case <-done:
		close(s.events)
	case <-time.After(closeWaitTimeout):
		slog.Warn("opencodeSession: close timed out, abandoning wg.Wait")
		// Report it: a reader is still running, so the engine must not
		// resume this conversation into a new process (it marks the key
		// unsafe).
		return fmt.Errorf("opencode: read loop did not exit within %s", closeWaitTimeout)
	}
	return nil
}

// closeWaitTimeout bounds how long Close waits for the read loop. A var
// only so tests can shorten it.
var closeWaitTimeout = 8 * time.Second

func truncate(s string, maxRunes int) string {
	if utf8.RuneCountInString(s) <= maxRunes {
		return s
	}
	return string([]rune(s)[:maxRunes]) + "..."
}
