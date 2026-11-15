/**
 * Copyright 2026 The ODML Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import type {ChatInterface, Conversation, ConversationConfig, Engine, Message, MessageLike} from '@litert-lm/core';

import {devHooks, type LlmChatStateController} from '../state_controller.js';
import type {LocalDirectoryService} from '../stores/local_directory_service.js';
import {getModelKey, ModelLoaderService} from '../stores/model_loader_service.js';
import {MODELS, type ModelSettings} from '../stores/settings_store.js';

import {MemoryStorage} from './memory_storage.js';

/** Supported modes for `?demo=<mode>`. */
type DemoMode = 'ready'|'empty'|'loading'|'error';

/** Size shown for the simulated model's cached weights. */
const DEMO_MODEL_BYTES = 2_300_000_000;

interface DemoTurnPlan {
  thought?: string;
  toolCode?: string;
  reply: string;
}

function extractPromptText(message: MessageLike): string {
  if (typeof message === 'string') return message;
  if (typeof message.content === 'string') return message.content;
  if (Array.isArray(message.content)) {
    return message.content
        .map(part => {
          if (part.type === 'text') return part.text;
          return `[${part.type} attached]`;
        })
        .join(' ');
  }
  return '';
}

function buildDemoTurnPlan(
    promptText: string, enableThinking: boolean,
    canRunJsTool = true): DemoTurnPlan {
  const lower = promptText.toLowerCase();

  if (lower.includes('particle')) {
    return {
      thought: enableThinking ?
          'Designing a self-contained HTML5 canvas particle constellation with pointer repulsion, velocity damping, and DPR scaling so it streams smoothly and runs interactively in Preview HTML.' :
          undefined,
      reply: `Here is a self-contained interactive HTML5 canvas particle effect you can launch directly with **Preview HTML**:

\`\`\`html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: radial-gradient(circle at 50% 35%, #131c2e 0%, #090d16 75%);
      font-family: system-ui, -apple-system, sans-serif;
      color: #e2e8f0;
    }
    canvas { display: block; width: 100%; height: 100%; }
    .hud {
      position: fixed;
      left: 20px;
      bottom: 20px;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(15, 23, 42, 0.78);
      border: 1px solid rgba(148, 163, 184, 0.22);
      font-size: 13px;
      pointer-events: none;
    }
  </style>
</head>
<body>
  <canvas id="stage"></canvas>
  <div class="hud">Move pointer to attract &amp; click to burst particles</div>
  <script>
    const canvas = document.getElementById('stage');
    const ctx = canvas.getContext('2d');
    const pointer = { x: -9999, y: -9999 };
    const particles = [];
    const COUNT = 90;

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = window.innerWidth * dpr;
      canvas.height = window.innerHeight * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    window.addEventListener('resize', resize);
    resize();

    for (let i = 0; i < COUNT; i++) {
      particles.push({
        x: Math.random() * window.innerWidth,
        y: Math.random() * window.innerHeight,
        vx: (Math.random() - 0.5) * 1.4,
        vy: (Math.random() - 0.5) * 1.4,
        r: 1.6 + Math.random() * 2.2,
      });
    }

    window.addEventListener('pointermove', (e) => {
      pointer.x = e.clientX;
      pointer.y = e.clientY;
    });
    window.addEventListener('pointerdown', (e) => {
      for (const p of particles) {
        const dx = p.x - e.clientX;
        const dy = p.y - e.clientY;
        const d = Math.hypot(dx, dy) || 1;
        if (d < 220) {
          p.vx += (dx / d) * 4.5;
          p.vy += (dy / d) * 4.5;
        }
      }
    });

    function frame() {
      ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        p.x = (p.x + p.vx + window.innerWidth) % window.innerWidth;
        p.y = (p.y + p.vy + window.innerHeight) % window.innerHeight;
        p.vx *= 0.995;
        p.vy *= 0.995;
        if (Math.hypot(p.vx, p.vy) < 0.35) {
          p.vx += (Math.random() - 0.5) * 0.15;
          p.vy += (Math.random() - 0.5) * 0.15;
        }
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = '#14d4aa';
        ctx.fill();

        for (let j = i + 1; j < particles.length; j++) {
          const q = particles[j];
          const dist = Math.hypot(p.x - q.x, p.y - q.y);
          if (dist < 115) {
            ctx.strokeStyle = 'rgba(20, 212, 170, ' + (1 - dist / 115) * 0.28 + ')';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(q.x, q.y);
            ctx.stroke();
          }
        }
      }
      requestAnimationFrame(frame);
    }
    frame();
  </script>
</body>
</html>
\`\`\`

Click **Preview HTML** in the code header above to interact with the live canvas constellation.`,
    };
  }

  if (lower.includes('button') || lower.includes('html')) {
    return {
      thought: enableThinking ?
          'Planning a self-contained HTML snippet with smooth CSS transitions and interactive canvas animation so it can be launched directly in the sandboxed HTML preview.' :
          undefined,
      reply: `Here is a self-contained interactive HTML snippet you can test right here with **Preview HTML**:

\`\`\`html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background: #0f1115;
      color: #f3f4f6;
      font-family: system-ui, sans-serif;
    }
    .card {
      padding: 32px 40px;
      border-radius: 16px;
      background: #181c24;
      border: 1px solid rgba(255, 255, 255, 0.1);
      text-align: center;
    }
    button {
      margin-top: 16px;
      padding: 12px 24px;
      border-radius: 999px;
      border: none;
      background: #12d6ad;
      color: #051b15;
      font-weight: 600;
      font-size: 15px;
      cursor: pointer;
      transition: transform 0.15s ease, opacity 0.15s ease;
    }
    button:hover { transform: translateY(-1px); opacity: 0.92; }
  </style>
</head>
<body>
  <div class="card">
    <h2>LiteRT-LM Sandbox Preview</h2>
    <p id="count">Clicked 0 times</p>
    <button id="btn">Trigger Interaction</button>
  </div>
  <script>
    let n = 0;
    document.getElementById('btn').onclick = () => {
      document.getElementById('count').textContent =
          'Clicked ' + (++n) + ' times';
    };
  </script>
</body>
</html>
\`\`\`

Click **Preview HTML** in the code header above to launch it in the sandboxed iframe.`,
    };
  }

  if (lower.includes('calculate') || lower.includes('math') ||
      lower.includes('prime') || /\d+\s*[*+/^%-]\s*\d+/.test(lower)) {
    if (!canRunJsTool) {
      return {
        thought: enableThinking ?
            'JavaScript sandbox tool is disabled in settings; deriving the closed-form sum directly.' :
            undefined,
        reply: `Using the closed-form identity directly (with \`run_javascript\` disabled):

$$\\sum_{k=1}^{n} k^2 = \\frac{n(n + 1)(2n + 1)}{6}$$

For $n = 12$, evaluating $\\frac{12 \\cdot 13 \\cdot 25}{6}$ yields **650**.`,
      };
    }
    return {
      thought: enableThinking ?
          'This prompt benefits from exact arithmetic verification. Invoking `run_javascript` in the isolated sandbox first, then formatting the derivation with LaTeX.' :
          undefined,
      toolCode:
          'const sum = Array.from({length: 12}, (_, i) => (i + 1) ** 2).reduce((a, b) => a + b, 0); console.log("Sum of squares 1..12:", sum); sum;',
      reply: `Using the built-in \`run_javascript\` sandbox, we verified the closed-form identity:

$$\\sum_{k=1}^{n} k^2 = \\frac{n(n + 1)(2n + 1)}{6}$$

For $n = 12$, evaluating $\\frac{12 \\cdot 13 \\cdot 25}{6}$ yields **650** with zero rounding error.`,
    };
  }

  return {
    thought: enableThinking ?
        'Structuring a concise explanation of on-device WebGPU inference, highlighting memory locality, KV-cache reuse, and multimodal pipelines in LiteRT-LM.' :
        undefined,
    reply: `**WebGPU** gives browsers low-overhead access to modern GPU compute shaders, allowing **LiteRT-LM** to run quantized Gemma models entirely on your device:

1. **Zero Server Round-Trips:** Prompts, images, and audio stay in local memory with full offline capability once cached.
2. **Fused Compute Kernels:** Matrix-vector decode steps execute directly in WGSL shaders while keeping the KV cache resident in VRAM.
3. **Multimodal & Agentic Ready:** Vision/audio encoders and sandboxed JavaScript tool calls compose in a single streaming session.

| Stage | Typical Bound | Key Metric |
| :--- | :--- | :--- |
| **Prefill** | Compute-bound | Prompt tokens / sec |
| **Decode** | Memory-bandwidth-bound | Output tokens / sec |

For an attention head dimension $d_k$, scaled dot-product attention computes $\\text{Attention}(Q, K, V) = \\text{softmax}\\!\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$.`,
  };
}

class DemoConversation implements ChatInterface {
  private history: Message[] = [];
  private cancelled = false;
  private lastPromptTokens = 28;
  private lastDecodeTokens = 96;
  private pendingToolFollowup: string|null = null;
  private readonly enableThinking: boolean;
  private readonly canRunJsTool: boolean;
  private readonly stepDelayMs: number;

  constructor(config?: ConversationConfig, stepDelayMs = 18) {
    this.enableThinking =
        Boolean(config?.preface?.extra_context?.['enable_thinking'] ?? true);
    this.canRunJsTool = Boolean(config?.preface?.tools?.some(
        t => 'function' in t && t.function.name === 'run_javascript'));
    this.stepDelayMs = stepDelayMs;
    if (config?.preface?.messages) {
      this.history = [...(config.preface.messages as Message[])];
    }
  }

  async sendMessage(message: MessageLike): Promise<Message> {
    const stream = this.sendMessageStreaming(message);
    const reader = stream.getReader();
    let text = '';
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      if (typeof value.content === 'string') {
        text += value.content;
      }
    }
    return {role: 'assistant', content: text};
  }

  sendMessageStreaming(message: MessageLike): ReadableStream<Message> {
    this.cancelled = false;
    const isToolResultTurn = typeof message === 'object' && message !== null &&
        'role' in message && (message as Message).role === 'tool';

    const promptText = extractPromptText(message);
    this.history.push(
        typeof message === 'string' ?
            ({role: 'user', content: message} as Message) :
            (message as Message));

    const plan = isToolResultTurn && this.pendingToolFollowup ?
        {reply: this.pendingToolFollowup} :
        buildDemoTurnPlan(promptText, this.enableThinking, this.canRunJsTool);

    this.lastPromptTokens = Math.max(12, Math.round(promptText.length / 3.6));
    const replyTokens = plan.reply.match(/\S+\s*/g) ?? [plan.reply];
    this.lastDecodeTokens = replyTokens.length;

    return new ReadableStream<Message>({
      start: async (controller) => {
        const sleep = (ms: number) =>
            new Promise<void>(resolve => setTimeout(resolve, ms));

        try {
          if (plan.thought) {
            const thoughtChunks =
                plan.thought.match(/.{1,14}/g) ?? [plan.thought];
            for (const chunk of thoughtChunks) {
              if (this.cancelled) return;
              await sleep(this.stepDelayMs);
              if (this.cancelled) return;
              controller.enqueue({
                role: 'assistant',
                content: '',
                channels: {'thought': chunk},
              });
            }
          }

          if (plan.toolCode && !isToolResultTurn) {
            this.pendingToolFollowup = plan.reply;
            await sleep(this.stepDelayMs * 2);
            if (this.cancelled) return;
            const toolCall: Message = {
              role: 'assistant',
              content: '',
              tool_calls: [{
                type: 'function',
                function: {
                  name: 'run_javascript',
                  arguments: {'code': plan.toolCode},
                },
              }],
            };
            this.history.push(toolCall);
            controller.enqueue(toolCall);
            controller.close();
            return;
          }

          this.pendingToolFollowup = null;
          for (let i = 0; i < replyTokens.length; i += 2) {
            if (this.cancelled) return;
            await sleep(this.stepDelayMs);
            if (this.cancelled) return;
            const piece = replyTokens.slice(i, i + 2).join('');
            controller.enqueue({
              role: 'assistant',
              content: piece,
            });
          }

          if (!this.cancelled) {
            this.history.push({role: 'assistant', content: plan.reply});
            controller.close();
          }
        } catch (err) {
          if (!this.cancelled) {
            controller.error(err);
          }
        }
      },
      cancel: () => {
        this.cancelled = true;
      },
    });
  }

  cancel() {
    this.cancelled = true;
  }

  getHistory(): Message[] {
    return [...this.history];
  }

  async getTokenCount(): Promise<number> {
    return this.lastPromptTokens + this.lastDecodeTokens;
  }

  async getBenchmarkInfo() {
    return {
      lastPrefillTokensPerSecond: 612.4,
      lastPrefillTokenCount: this.lastPromptTokens,
      lastDecodeTokensPerSecond: 54.8,
      lastDecodeTokenCount: this.lastDecodeTokens,
      timeToFirstTokenInSecond: 0.08,
      mtpRounds: 0,
      mtpDraftedTokens: 0,
      mtpAcceptedTokens: 0,
      mtpEmittedTokens: 0,
      plainDecodeSteps: this.lastDecodeTokens,
    };
  }

  async delete(): Promise<void> {}
}

/** Creates a lightweight simulated `Engine` for `?demo=...` verification. */
export function createDemoEngine(stepDelayMs = 18): Engine {
  return {
    createConversation: async (config?: ConversationConfig) =>
        new DemoConversation(config, stepDelayMs) as unknown as Conversation,
    delete: async () => {},
  } as unknown as Engine;
}

const FALLBACK_PNG_BYTES = new Uint8Array([
  0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
  0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
  0x08, 0x06, 0x00, 0x00, 0x00, 0x1f, 0x15, 0xc4, 0x89, 0x00, 0x00, 0x00,
  0x0a, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0x00, 0x01, 0x00, 0x00,
  0x05, 0x00, 0x01, 0x0d, 0x0a, 0x2d, 0xb4, 0x00, 0x00, 0x00, 0x00, 0x49,
  0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
]);

async function createSampleDiagramPngBlob(): Promise<Blob> {
  if (typeof document === 'undefined') {
    return new Blob([FALLBACK_PNG_BYTES], {type: 'image/png'});
  }
  const canvas = document.createElement('canvas');
  canvas.width = 520;
  canvas.height = 172;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    return new Blob([FALLBACK_PNG_BYTES], {type: 'image/png'});
  }

  // Background panel
  ctx.fillStyle = '#131720';
  ctx.beginPath();
  ctx.roundRect(0, 0, 520, 172, 16);
  ctx.fill();

  const drawBox = (
      x: number, w: number, border: string, title: string, sub: string,
      subColor = '#94a3b8') => {
    ctx.fillStyle = '#1d2433';
    ctx.strokeStyle = border;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(x, 44, w, 84, 12);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#f1f5f9';
    ctx.font = '600 14px system-ui, -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(title, x + w / 2, 80);

    ctx.fillStyle = subColor;
    ctx.font = '500 12px ui-monospace, SFMono-Regular, monospace';
    ctx.fillText(sub, x + w / 2, 104);
  };

  const drawArrow = (x1: number, x2: number) => {
    ctx.strokeStyle = '#14d4aa';
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.moveTo(x1, 86);
    ctx.lineTo(x2, 86);
    ctx.moveTo(x2 - 6, 81);
    ctx.lineTo(x2, 86);
    ctx.lineTo(x2 - 6, 91);
    ctx.stroke();
  };

  drawBox(22, 138, '#14d4aa', 'Prompt + Media', 'Text · Image · WAV');
  drawArrow(166, 190);
  drawBox(196, 138, '#60a5fa', 'LiteRT-LM WASM', 'WebGPU Shaders');
  drawArrow(340, 364);
  drawBox(370, 128, '#14d4aa', 'Token Stream', '56.4 tk/s', '#14d4aa');

  return new Promise<Blob>((resolve) => {
    canvas.toBlob((blob) => {
      resolve(blob ?? new Blob([FALLBACK_PNG_BYTES], {type: 'image/png'}));
    }, 'image/png');
  });
}

/** A model loader that "loads" a simulated engine instead of real weights. */
export class DemoModelLoader extends ModelLoaderService {
  /** Resolves when the demo state is set up. */
  ready = Promise.resolve();

  private readonly demoCachedModels = new Map<string, number>();

  constructor(
      private readonly onUpdate: () => void,
      updateStatus: (msg: string) => void,
      localDirService: LocalDirectoryService) {
    super(onUpdate, updateStatus, localDirService);
  }

  /** Marks a model's weights as cached. */
  setCached(modelPath: string) {
    this.demoCachedModels.set(getModelKey(modelPath), DEMO_MODEL_BYTES);
    this.cachedModels = new Map(this.demoCachedModels);
  }

  override async updateCacheSize() {
    this.cachedModels = new Map(this.demoCachedModels);
    this.onUpdate();
  }

  override async loadModelWeights(
      settings: ModelSettings, onModelLoaded: () => Promise<void>) {
    // Leave a simulated (?demo=loading) download, if any.
    this.isModelLoading = false;
    this.downloadProgresses.clear();
    this.downloadSpeeds.clear();
    this.downloadAbortController = null;
    await this.engine?.delete();
    this.engine = createDemoEngine();
    this.loadedSettings = structuredClone(settings);
    this.setCached(settings.selectedModelPath);
    this.metricLoadTime = '0.38s';
    await onModelLoaded();
    this.onUpdate();
  }
}

/**
 * Sets up the `?demo=<mode>` state on a controller whose model loader is a
 * `DemoModelLoader`.
 */
function startDemo(mode: DemoMode, controller: LlmChatStateController) {
  const {settings, chatSession} = controller;
  const modelLoader = controller.modelLoader as DemoModelLoader;
  const activeModel =
      MODELS.find(m => m.path === settings.selectedModelPath) ?? MODELS[0]!;
  const modelKey = getModelKey(activeModel.path);

  if (mode === 'loading') {
    modelLoader.isModelLoading = true;
    modelLoader.downloadProgresses.set(modelKey, 64);
    modelLoader.downloadSpeeds.set(
        modelKey, '1472.0 MB / 2300.0 MB · 48.2 MB/s');
    const abortController = new AbortController();
    abortController.signal.addEventListener('abort', () => {
      modelLoader.isModelLoading = false;
      modelLoader.downloadProgresses.delete(modelKey);
      modelLoader.downloadSpeeds.delete(modelKey);
      modelLoader.downloadAbortController = null;
      controller.statusText = 'Model download cancelled by user.';
      controller.requestUpdate();
    });
    modelLoader.downloadAbortController = abortController;
    controller.statusText = `Downloading weights (${activeModel.filename})...`;
    controller.requestUpdate();
    return;
  }

  if (mode === 'error') {
    const errorMessage =
        'Failed to load model: WebGPU adapter unavailable in this browser environment. Enable hardware acceleration or switch to a WebGPU-capable browser.';
    controller.statusText = errorMessage;
    controller.errorMessage = errorMessage;
    controller.requestUpdate();
    return;
  }

  // 'ready' and 'empty' start with the simulated engine already loaded.
  modelLoader.setCached(activeModel.path);
  modelLoader.metricLoadTime = '0.42s';
  modelLoader.engine = createDemoEngine();
  modelLoader.loadedSettings = structuredClone(settings.modelSettings);

  modelLoader.ready = (async () => {
    await chatSession.startNewConversation();
    if (mode === 'ready') {
      const diagram = chatSession.createMedia(
          'image', 'pipeline-overview.png', await createSampleDiagramPngBlob());
      const plan = buildDemoTurnPlan('html', true);
      const mathPlan = buildDemoTurnPlan('Explain WebGPU', true);
      const id = 'demo-showcase';
      controller.storage.setItem(
          'litertlm-conversations-list', JSON.stringify([{
            id,
            title: 'On-device WebGPU & interactive preview',
            createdAt: Date.now(),
            modelPath: activeModel.path,
          }]));
      controller.storage.setItem(`litertlm-chat-history-${id}`, JSON.stringify([
        {
          role: 'user',
          text:
              'Walk through this architecture diagram [image: pipeline-overview.png] and summarize how on-device WebGPU inference works.',
          parts: [
            {type: 'text', text: 'Walk through this architecture diagram '},
            {type: 'image', id: diagram.id, name: diagram.name},
            {
              type: 'text',
              text: ' and summarize how on-device WebGPU inference works.',
            },
          ],
          senderName: 'User',
          tokensCount: '42',
        },
        {
          role: 'assistant',
          text: mathPlan.reply,
          thoughtText: mathPlan.thought,
          senderName: activeModel.name,
          prefillSpeed: '648.2 tk/s',
          decodeSpeed: '56.4 tk/s',
          tokensCount: '138',
        },
        {
          role: 'user',
          text: 'Create an interactive HTML card I can test in the preview.',
          senderName: 'User',
          tokensCount: '16',
        },
        {
          role: 'assistant',
          text: plan.reply,
          thoughtText: plan.thought,
          senderName: activeModel.name,
          prefillSpeed: '590.5 tk/s',
          decodeSpeed: '53.9 tk/s',
          tokensCount: '164',
        },
      ]));
      chatSession.loadSavedConversationsIndex();
      await chatSession.selectConversation(id);
    }
    await chatSession.createConversationSession();
    controller.statusText = 'Model loaded and ready.';
    controller.requestUpdate();
  })();
}

/**
 * Installs the dev hooks for `?demo=<mode>` (`1`/`ready`, `empty`, `loading`,
 * or `error`): deterministic states with a simulated engine, whose
 * conversations and settings stay in memory. Returns whether demo mode is on.
 */
export function installDemoMode(search: string): boolean {
  const raw = new URLSearchParams(search).get('demo')?.trim();
  if (!raw) return false;
  const mode: DemoMode =
      raw === 'loading' || raw === 'error' || raw === 'empty' ? raw : 'ready';
  devHooks.storage ??= new MemoryStorage();
  devHooks.createModelLoader = (...args) => new DemoModelLoader(...args);
  devHooks.onConnected = controller => {
    startDemo(mode, controller);
    return true;
  };
  return true;
}
